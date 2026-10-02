//! Training spread over several machines.
//!
//! One coordinator (the always-on server) holds the master weights. Each worker trains
//! its own copy for a couple of minutes, sends back only the weights that moved, then
//! pulls every other worker's changes, so all machines keep learning on nearly the same
//! net. Workers talk HTTP through the system `curl` (shipped with Windows 10+ too), which
//! keeps TLS out of this dependency-free crate; nginx terminates TLS for the coordinator.

use crate::board::Rng;
use crate::ntuple::{self, key_parts, NTuple, Pool};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_JOB: &str = "alpha=0.0015625\nrestart=0\nsecs=120\nsend_mb=40\ntc=0\npause=0\n";
/// Deltas the coordinator keeps for workers that fall behind; older ones need a full download.
const LOG_BUDGET: usize = 1 << 30;
const LOG_MAX_AGE: Duration = Duration::from_secs(6 * 3600);
const SAVE_EVERY: Duration = Duration::from_secs(600);
/// Chunk reports per worker that the status page averages over.
const RECENT: usize = 20;
/// Boards the coordinator keeps per restart-pool key.
const POOL_CAP: usize = 100_000;

/// Bytes per entry on the wire, roughly: a 1-3 byte index gap plus a bf16 value.
const BYTES_PER_ENTRY: f64 = 3.5;

/// A delta as: entry count (u32), then per entry the varint gap from the previous index
/// and the change as bf16 (the top half of an f32; the rounding error stays behind as residual).
pub fn encode(delta: &[(u32, f32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + delta.len() * 4);
    out.extend((delta.len() as u32).to_le_bytes());
    let mut prev = 0u32;
    for &(i, d) in delta {
        let mut gap = i - prev;
        prev = i;
        loop {
            let byte = (gap & 0x7F) as u8;
            gap >>= 7;
            if gap == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
        out.extend(to_bf16(d).to_le_bytes());
    }
    out
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub fn decode(bytes: &[u8]) -> Option<Vec<(u32, f32)>> {
    let n = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
    let mut out = Vec::with_capacity(n);
    let (mut pos, mut idx) = (4, 0u32);
    for _ in 0..n {
        let mut gap = 0u32;
        for shift in (0..35).step_by(7) {
            let byte = *bytes.get(pos)?;
            pos += 1;
            gap |= ((byte & 0x7F) as u32) << shift;
            if byte & 0x80 == 0 {
                break;
            }
        }
        idx += gap;
        out.push((idx, from_bf16(u16::from_le_bytes(bytes.get(pos..pos + 2)?.try_into().ok()?))));
        pos += 2;
    }
    Some(out)
}

/// The largest changes that fit in `budget_bytes`, rounded to what the wire carries.
/// Whatever is left out stays in the caller's residual and goes in a later chunk.
fn pick_largest(mut raw: Vec<(u32, f32)>, budget_bytes: f64) -> Vec<(u32, f32)> {
    let keep = (budget_bytes / BYTES_PER_ENTRY) as usize;
    if raw.len() > keep && keep > 0 {
        raw.select_nth_unstable_by(keep - 1, |a, b| b.1.abs().total_cmp(&a.1.abs()));
        raw.truncate(keep);
        raw.sort_unstable_by_key(|e| e.0);
    }
    raw.into_iter().map(|(i, d)| (i, from_bf16(to_bf16(d)))).filter(|e| e.1 != 0.0).collect()
}

/// Job settings as `key=value` lines.
fn job_text<'a>(job: &'a str, key: &str) -> Option<&'a str> {
    job.lines().find_map(|l| Some(l.strip_prefix(key)?.strip_prefix('=')?.trim()))
}

fn job_value(job: &str, key: &str) -> Option<f64> {
    job.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('=')?.trim().parse().ok())
}

/// Totals of one training chunk. Score and `reached` (2048 .. 65536) are over the fresh
/// games, `best` and `top` are the chunk's highest fresh score and tile rank, and
/// `restarts[s]` counts the episodes restarted at stage s+1 (16384, 32768, 65536 boards)
/// and how many of them built the next tile.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Chunk {
    secs: f64,
    episodes: u64,
    fresh: u64,
    score: u64,
    reached: [u64; 6],
    best: u64,
    top: u8,
    restarts: [[u64; 2]; 3],
}

impl Chunk {
    fn to_query(self) -> String {
        let r = self.reached.map(|x| x.to_string()).join(",");
        let mut q = format!("secs={:.1}&episodes={}&fresh={}&score={}&reached={r}&best={}&top={}", self.secs, self.episodes, self.fresh, self.score, self.best, self.top);
        for (i, [n, p]) in self.restarts.iter().enumerate() {
            q += &format!("&r{}={n},{p}", i + 1);
        }
        q
    }

    fn from_query(q: &HashMap<String, String>) -> Chunk {
        let num = |k: &str| q.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        let list = |k: &str, slots: &mut [u64]| {
            for (slot, v) in slots.iter_mut().zip(q.get(k).map_or("", |s| s).split(',')) {
                *slot = v.parse().unwrap_or(0);
            }
        };
        let mut reached = [0; 6];
        list("reached", &mut reached);
        let mut restarts = [[0; 2]; 3];
        for (i, r) in restarts.iter_mut().enumerate() {
            list(&format!("r{}", i + 1), r);
        }
        Chunk { secs: num("secs"), episodes: num("episodes") as u64, fresh: num("fresh") as u64, score: num("score") as u64, reached, best: num("best") as u64, top: num("top") as u8, restarts }
    }

    fn add(&mut self, o: &Chunk) {
        self.secs += o.secs;
        self.episodes += o.episodes;
        self.fresh += o.fresh;
        self.score += o.score;
        for (a, b) in self.reached.iter_mut().zip(o.reached) {
            *a += b;
        }
        self.best = self.best.max(o.best);
        self.top = self.top.max(o.top);
        for (a, b) in self.restarts.iter_mut().zip(o.restarts) {
            a[0] += b[0];
            a[1] += b[1];
        }
    }

    fn rate(&self) -> f64 {
        if self.secs > 0.0 { self.episodes as f64 / self.secs } else { 0.0 }
    }
}

// ---------------------------------------------------------------- coordinator

struct Delta {
    seq: u64,
    from: String,
    bytes: Arc<Vec<u8>>,
    at: Instant,
}

struct WorkerInfo {
    last_seen: Instant,
    recent: VecDeque<Chunk>,
    total_episodes: u64,
    /// The latest heartbeat: what the worker is doing plus its machine's load.
    beat: Option<(Instant, HashMap<String, String>)>,
}

impl WorkerInfo {
    fn new() -> WorkerInfo {
        WorkerInfo { last_seen: Instant::now(), recent: VecDeque::new(), total_episodes: 0, beat: None }
    }
}

struct Coord {
    net: NTuple,
    path: String,
    seq: u64,
    saved_seq: u64,
    saved_at: Instant,
    log: VecDeque<Delta>,
    log_bytes: usize,
    job: String,
    workers: HashMap<String, WorkerInfo>,
    /// Training games since the net was created, kept in `path.episodes` across restarts.
    episodes: u64,
    /// Best training game so far: (score, tile rank, machine, unix time), kept in `path.best`.
    record: Option<(u64, u8, String, u64)>,
    /// Restart boards harvested by every worker, kept in `path.pool`.
    pool: Pool,
    rng: Rng,
}

impl Coord {
    fn save(&mut self) {
        if self.pool.dirty() {
            if let Err(e) = self.pool.save(&format!("{}.pool", self.path)) {
                eprintln!("saving the pool: {e}");
            }
        }
        if self.seq == self.saved_seq {
            return;
        }
        let t = Instant::now();
        if let Err(e) = self.net.save(&self.path) {
            eprintln!("saving master: {e}");
            return;
        }
        let _ = std::fs::write(format!("{}.seq", self.path), self.seq.to_string());
        let _ = std::fs::write(format!("{}.episodes", self.path), self.episodes.to_string());
        self.saved_seq = self.seq;
        self.saved_at = Instant::now();
        // Only deltas the saved file already holds may go, and only when over budget.
        while let Some(d) = self.log.front() {
            if d.seq > self.saved_seq || (self.log_bytes <= LOG_BUDGET && d.at.elapsed() < LOG_MAX_AGE) {
                break;
            }
            self.log_bytes -= d.bytes.len();
            self.log.pop_front();
        }
        eprintln!("saved master at seq {} in {:.1}s", self.seq, t.elapsed().as_secs_f64());
    }

    /// Episodes played in the current stage (from `stage_start`) and the stage's goal.
    fn progress(&self) -> (f64, f64) {
        let since = (self.episodes as f64 - job_value(&self.job, "stage_start").unwrap_or(0.0)).max(0.0);
        (since, job_value(&self.job, "goal").unwrap_or(100e6))
    }

    /// The job as workers see it. With `schedule=otd`, alpha and TC follow the OTD recipe
    /// over the stage's `goal` episodes: per-weight alpha 0.1/64, cut 10x at 50% and again at
    /// 75%. `net_stages` is always the master's stage count, so a worker whose copy has a
    /// different shape knows to download the master again.
    fn effective_job(&self) -> String {
        let mut out: String = self.job.lines().filter(|l| !l.starts_with("net_stages=")).map(|l| format!("{l}\n")).collect();
        if job_text(&self.job, "schedule") == Some("otd") {
            let (since, goal) = self.progress();
            // No TC phase: its accumulators live on each worker, start at zero and reset on every
            // restart, so each weight's first TC step is the full 1.0 rate. On 2026-09-27 that took
            // the farm's mean score from 302k to 35k within an hour.
            let (alpha, tc) = match since / goal {
                d if d < 0.5 => (0.1, 0),
                d if d < 0.75 => (0.01, 0),
                _ => (0.001, 0),
            };
            out = out.lines().filter(|l| !l.starts_with("alpha=") && !l.starts_with("tc=")).map(|l| format!("{l}\n")).collect();
            out += &format!("alpha={}\ntc={tc}\n", alpha / 64.0);
        }
        out += &format!("net_stages={}\n", self.net.stages());
        out
    }

    /// Writes the delta log next to the net, for `load_log` after a restart.
    fn save_log(&self) -> std::io::Result<()> {
        let path = format!("{}.log", self.path);
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{path}.tmp"))?);
        for d in &self.log {
            f.write_all(&d.seq.to_le_bytes())?;
            f.write_all(&(d.from.len() as u32).to_le_bytes())?;
            f.write_all(d.from.as_bytes())?;
            f.write_all(&(d.bytes.len() as u32).to_le_bytes())?;
            f.write_all(&d.bytes)?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(format!("{path}.tmp"), path)
    }

    fn status(&self, color: bool) -> String {
        let paint = |code: &str, text: String| if color { format!("\x1b[{code}m{text}\x1b[0m") } else { text };
        let effective = self.effective_job();
        let job = |k: &str| job_value(&effective, k);
        let mut s = String::new();

        // Headline: progress towards the episode goal of this training stage.
        let live = |w: &WorkerInfo| w.last_seen.elapsed() < Duration::from_secs(600);
        let rate: f64 = self.workers.values().filter(|w| live(w)).map(recent_rate).sum();
        let (since, goal) = self.progress();
        let done = (since / goal).min(1.0);
        let bar = (done * 30.0).round() as usize;
        let eta = if rate > 0.0 { duration((goal - since).max(0.0) / rate) } else { "-".into() };
        s += &paint("1", "2048 TRAINING FARM".into());
        s += &format!("   alpha {}{}   TC {}   stages {} (frozen below {}), restarts from stage {}{}\n",
            job("alpha").unwrap_or(0.0),
            if job_text(&self.job, "schedule") == Some("otd") { " (auto: OTD schedule)" } else { "" },
            if job("tc").unwrap_or(0.0) > 0.0 { "on" } else { "off" },
            self.net.stages(),
            job("freeze").unwrap_or(0.0),
            job("restart_stage").unwrap_or(1.0),
            if job("pause").unwrap_or(0.0) > 0.0 { paint("33", "   PAUSED".into()) } else { String::new() });
        s += &format!("progress  {}{}  {:.1}%  {} / {} games this stage ({} in all)  ETA {eta}\n",
            paint("32", "#".repeat(bar)), paint("2", ".".repeat(30 - bar)), done * 100.0,
            millions(since), millions(goal), millions(self.episodes as f64));
        if let Some((score, rank, who, at)) = &self.record {
            let ago = duration(unix_now().saturating_sub(*at) as f64);
            s += &format!("best game {}  ({} tile)  by {who}, {ago} ago\n", paint("1;32", thousands(*score as f64)), 1u64 << rank);
        }
        s += "\n";

        // Machines: what each is doing and how hard it is working.
        let mut names: Vec<_> = self.workers.keys().collect();
        names.sort();
        s += &paint("1", format!("{:<12}{:<15}{:>6}{:>8}{:>20}{:>9}", "MACHINE", "STATUS", "CPU", "TEMP", "RAM used/total", "g2048"));
        s += "\n";
        for name in &names {
            let w = &self.workers[*name];
            let (state, cpu, temp, ram, rss) = match &w.beat {
                Some((at, b)) if at.elapsed() < Duration::from_secs(60) => {
                    let num = |k: &str| b.get(k).and_then(|v| v.parse::<f64>().ok());
                    let state = b.get("state").cloned().unwrap_or_default();
                    let state = match state.as_str() {
                        "training" => paint("32", format!("{:<15}", "* training")),
                        _ => paint("33", format!("{:<15}", format!("* {state}"))),
                    };
                    let cpu = num("cpu").map_or(format!("{:>6}", "-"), |c| format!("{:>5.0}%", c));
                    let temp = match num("temp") {
                        Some(t) => paint(if t >= 95.0 { "31" } else if t >= 85.0 { "33" } else { "32" }, format!("{:>7.0}C", t)),
                        None => format!("{:>8}", "n/a"),
                    };
                    let ram = match (num("ram"), num("ramtot")) {
                        (Some(u), Some(t)) => {
                            let text = format!("{:>20}", format!("{:.1} / {:.1} GB", u / 1024.0, t / 1024.0));
                            if u / t > 0.9 { paint("31", text) } else { text }
                        }
                        _ => format!("{:>20}", "-"),
                    };
                    let rss = num("rss").map_or(format!("{:>9}", "-"), |r| format!("{:>9}", format!("{:.1} GB", r / 1024.0)));
                    (state, cpu, temp, ram, rss)
                }
                Some((at, _)) => (paint("31", format!("{:<15}", format!("OFFLINE {}", duration(at.elapsed().as_secs_f64())))), String::new(), String::new(), String::new(), String::new()),
                None => (paint("33", format!("{:<15}", "old worker")), format!("{:>6}", "?"), format!("{:>8}", "?"), format!("{:>20}", "update the exe"), String::new()),
            };
            s += &format!("{:<12}{state}{cpu}{temp}{ram}{rss}\n", name);
        }

        // Training results, from each worker's recent chunks.
        s += "\n";
        s += &paint("1", format!("{:<12}{:>9}{:>12}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>12}", "RESULTS", "games/s", "mean score", "2048", "4096", "8192", "16384", "32768", "65536", "best"));
        s += "\n";
        let row = |label: &str, c: &Chunk, rate: f64| {
            let pct = |x: u64| if c.fresh == 0 { 0.0 } else { 100.0 * x as f64 / c.fresh as f64 };
            let r = c.reached;
            let mean = if c.fresh == 0 { 0.0 } else { c.score as f64 / c.fresh as f64 };
            format!("{label:<12}{:>9}{:>12}{:>7.1}%{:>7.1}%{:>7.1}%{:>7.2}%{:>7.2}%{:>7.3}%{:>12}\n",
                thousands(rate), thousands(mean), pct(r[0]), pct(r[1]), pct(r[2]), pct(r[3]), pct(r[4]), pct(r[5]), thousands(c.best as f64))
        };
        let mut sums: Vec<(&String, Chunk, bool)> = Vec::new();
        let mut total = Chunk::default();
        for name in &names {
            let w = &self.workers[*name];
            if w.recent.is_empty() {
                continue;
            }
            let mut sum = Chunk::default();
            w.recent.iter().for_each(|c| sum.add(c));
            s += &row(name, &sum, if live(w) { sum.rate() } else { 0.0 });
            if live(w) {
                total.add(&sum);
            }
            sums.push((name, sum, live(w)));
        }
        s += &paint("1", row("ALL", &total, rate));

        // Restart episodes: how often a game restarted at a stage built the stage's next tile.
        if total.restarts.iter().any(|r| r[0] > 0) {
            s += "\n";
            s += &paint("1", format!("{:<12}{:>12}{:>10}{:>12}{:>10}{:>12}{:>10}", "RESTARTS", "from 16384", "-> 32768", "from 32768", "-> 65536", "from 65536", "-> next"));
            s += "\n";
            let rrow = |label: &str, c: &Chunk| {
                let mut line = format!("{label:<12}");
                for [n, p] in c.restarts {
                    line += &if n == 0 { format!("{:>12}{:>10}", "-", "-") } else { format!("{:>12}{:>9.2}%", thousands(n as f64), 100.0 * p as f64 / n as f64) };
                }
                line + "\n"
            };
            for (name, sum, _) in &sums {
                s += &rrow(name, sum);
            }
            s += &paint("1", rrow("ALL", &total));
        }

        // The restart pool, one line per largest tile (per chain state is ~15 lines per tile,
        // too many for `farm watch`).
        let summary = self.pool.summary();
        if !summary.is_empty() {
            let mut rows: Vec<(u8, usize, usize, u64)> = Vec::new();
            for (k, n, seen) in summary {
                let (m, _, _) = key_parts(k);
                match rows.last_mut() {
                    Some(r) if r.0 == m => { r.1 += 1; r.2 += n; r.3 += seen; }
                    _ => rows.push((m, 1, n, seen)),
                }
            }
            s += "\n";
            s += &paint("1", format!("{:<12}{:<10}{:>10}{:>12}{:>12}", "POOL", "largest", "states", "boards", "seen"));
            s += "\n";
            for (m, states, n, seen) in rows {
                let tile = if m == 0 { "-".to_string() } else { (1u64 << m).to_string() };
                s += &format!("{:<12}{:<10}{:>10}{:>12}{:>12}\n", "", tile, states, thousands(n as f64), thousands(seen as f64));
            }
        }
        s += &paint("2", format!(
            "\nmaster: update {}, saved {} ago, {} updates held ({} MB), {} pool boards. Scores and best are over each machine's last {RECENT} chunks.\n",
            self.seq, duration(self.saved_at.elapsed().as_secs_f64()), self.log.len(), self.log_bytes >> 20, thousands(self.pool.total() as f64)));
        s
    }
}

fn duration(secs: f64) -> String {
    let m = (secs / 60.0) as u64;
    if m >= 60 { format!("{}h {:02}m", m / 60, m % 60) } else if m > 0 { format!("{m}m") } else { format!("{}s", secs as u64) }
}

fn millions(x: f64) -> String {
    if x >= 1e6 { format!("{:.1}M", x / 1e6) } else { format!("{:.0}k", x / 1e3) }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Reads a log written by `save_log`. It is used only if it runs right up to `seq`, the
/// saved net's update number: a log with a gap would let a worker skip updates unnoticed.
/// The file is removed either way, so a stale log can never be read by a later start.
fn load_log(path: &str, seq: u64) -> (VecDeque<Delta>, usize) {
    let Ok(data) = std::fs::read(path) else { return (VecDeque::new(), 0) };
    let _ = std::fs::remove_file(path);
    let mut log = VecDeque::new();
    let mut bytes = 0;
    let mut at = 0;
    let mut take = |n: usize| {
        let s = data.get(at..at + n)?;
        at += n;
        Some(s)
    };
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
    while let Some(s) = take(8) {
        let d_seq = u64::from_le_bytes(s.try_into().unwrap());
        let Some(from) = take(4).map(u32_at).and_then(|n| take(n)) else { break };
        let from = String::from_utf8_lossy(from).to_string();
        let Some(body) = take(4).map(u32_at).and_then(|n| take(n)) else { break };
        bytes += body.len();
        log.push_back(Delta { seq: d_seq, from, bytes: Arc::new(body.to_vec()), at: Instant::now() });
    }
    if log.back().map(|d| d.seq) != Some(seq) {
        return (VecDeque::new(), 0);
    }
    (log, bytes)
}

fn thousands(x: f64) -> String {
    let digits = format!("{:.0}", x);
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn recent_rate(w: &WorkerInfo) -> f64 {
    let mut c = Chunk::default();
    w.recent.iter().for_each(|x| c.add(x));
    c.rate()
}

/// This worker's share of the games all live workers play per second.
fn share(workers: &HashMap<String, WorkerInfo>, name: &str) -> f32 {
    let live = |w: &&WorkerInfo| w.last_seen.elapsed() < Duration::from_secs(600);
    let total: f64 = workers.values().filter(live).map(recent_rate).sum();
    let mine = workers.get(name).map_or(0.0, recent_rate);
    if total > 0.0 { (mine / total).clamp(0.05, 1.0) as f32 } else { 1.0 }
}

/// A delta times `scale`, rounded to what the wire carries so both ends agree exactly.
fn scaled(delta: &[(u32, f32)], scale: f32) -> Vec<(u32, f32)> {
    delta.iter().map(|&(i, d)| (i, from_bf16(to_bf16(d * scale)))).filter(|e| e.1 != 0.0).collect()
}

/// Removes the entries below `start`, the first trainable weight under the job's freeze,
/// and returns how many went. A finished stage cannot be written by any worker this way.
fn drop_frozen(delta: &mut Vec<(u32, f32)>, start: usize) -> usize {
    let before = delta.len();
    delta.retain(|&(i, _)| i as usize >= start);
    before - delta.len()
}

/// What a worker adds to its own net after the master took `taken` of what it `sent`
/// (both sorted by index, `taken` a scaled and filtered subset): the difference, so the
/// local copy holds exactly what the master holds and the rest returns to the residual.
fn settle(sent: &[(u32, f32)], taken: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut back = Vec::with_capacity(sent.len());
    let mut t = taken.iter().peekable();
    for &(i, d) in sent {
        let a = if t.peek().is_some_and(|x| x.0 == i) { t.next().unwrap().1 } else { 0.0 };
        back.push((i, a - d));
    }
    back
}

fn parse_query(target: &str) -> (String, HashMap<String, String>) {
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    let map = q.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
    (path.to_string(), map)
}

/// Deletes the download links of `path` except the newest two (`keep` and the one before).
fn prune_links(path: &str, keep: u64) {
    let p = std::path::Path::new(path);
    let (dir, name) = (p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new(".")), p.file_name().unwrap_or_default().to_string_lossy());
    let prefix = format!("{name}.dl-");
    let mut seqs: Vec<u64> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_prefix(&prefix).and_then(|s| s.parse().ok()))
        .collect();
    seqs.sort_unstable();
    seqs.retain(|&s| s != keep);
    for s in seqs.iter().rev().skip(1) {
        let _ = std::fs::remove_file(dir.join(format!("{prefix}{s}")));
    }
}

fn respond(stream: &mut TcpStream, code: u16, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        410 => "Gone",
        416 => "Range Not Satisfiable",
        _ => "Error",
    };
    let head = format!("HTTP/1.1 {code} {reason}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    let _ = stream.write_all(head.as_bytes()).and_then(|_| stream.write_all(body));
}

fn handle(mut stream: TcpStream, coord: &Mutex<Coord>, token: &str) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut words = line.split_whitespace();
    let (method, target) = (words.next().unwrap_or("").to_string(), words.next().unwrap_or("").to_string());
    let (mut len, mut authed, mut from) = (0usize, false, 0u64);
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? <= 2 {
            break;
        }
        let (k, v) = h.split_once(':').unwrap_or(("", ""));
        match k.trim().to_ascii_lowercase().as_str() {
            "content-length" => len = v.trim().parse().unwrap_or(0),
            "authorization" => authed = v.trim() == format!("Bearer {token}"),
            // Only the "bytes=N-" form curl -C sends.
            "range" => from = v.trim().strip_prefix("bytes=").and_then(|r| r.strip_suffix('-')).and_then(|n| n.parse().ok()).unwrap_or(0),
            _ => {}
        }
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    if !authed {
        respond(&mut stream, 401, b"bad token\n");
        return Ok(());
    }
    let (path, q) = parse_query(&target);
    match (method.as_str(), path.as_str()) {
        ("GET", "/job") => {
            let job = coord.lock().unwrap().effective_job();
            respond(&mut stream, 200, job.as_bytes());
        }
        ("POST", "/job") => {
            let mut c = coord.lock().unwrap();
            c.job = String::from_utf8_lossy(&body).to_string();
            let _ = std::fs::write(format!("{}.job", c.path), &c.job);
            respond(&mut stream, 200, c.job.as_bytes());
        }
        ("GET", "/status") => {
            let s = coord.lock().unwrap().status(q.contains_key("color"));
            respond(&mut stream, 200, s.as_bytes());
        }
        ("POST", "/beat") => {
            let name = q.get("name").cloned().unwrap_or_default();
            let mut c = coord.lock().unwrap();
            c.workers.entry(name).or_insert_with(WorkerInfo::new).beat = Some((Instant::now(), q));
            respond(&mut stream, 200, b"ok\n");
        }
        ("GET", "/net") => {
            // Every download is served from a hard link of the saved file named by its seq,
            // so a broken transfer can resume (Range + ?seq=) from the same bytes after later
            // saves have replaced the master. Only the newest two links are kept.
            let resume: Option<u64> = q.get("seq").and_then(|s| s.parse().ok()).filter(|_| from > 0);
            let (mut file, seq) = {
                let mut c = coord.lock().unwrap();
                let base = c.path.clone();
                let link = |seq: u64| format!("{base}.dl-{seq}");
                match resume {
                    Some(seq) => match std::fs::File::open(link(seq)) {
                        Ok(f) => (f, seq),
                        Err(_) => {
                            drop(c);
                            respond(&mut stream, 416, b"that net is gone, download it again\n");
                            return Ok(());
                        }
                    },
                    None => {
                        if c.saved_seq != c.seq && c.saved_at.elapsed() > Duration::from_secs(60) {
                            c.save();
                        }
                        let seq = c.saved_seq;
                        if std::fs::metadata(link(seq)).is_err() {
                            std::fs::hard_link(&base, link(seq))?;
                        }
                        prune_links(&base, seq);
                        (std::fs::File::open(link(seq))?, seq)
                    }
                }
            };
            let size = file.metadata()?.len() + 8;
            if from >= size {
                respond(&mut stream, 416, b"range past the end\n");
                return Ok(());
            }
            let head = if resume.is_some() {
                format!("HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes {from}-{}/{size}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", size - 1, size - from)
            } else {
                format!("HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n")
            };
            stream.write_all(head.as_bytes())?;
            if resume.is_some() {
                // The body is 8 seq bytes then the file; a resume always starts inside the file
                // (the worker only resumes once it holds the seq).
                file.seek(std::io::SeekFrom::Start(from.saturating_sub(8)))?;
            } else {
                stream.write_all(&seq.to_le_bytes())?;
            }
            std::io::copy(&mut file, &mut stream)?;
        }
        ("GET", "/deltas") => {
            let since: u64 = q.get("since").and_then(|s| s.parse().ok()).unwrap_or(0);
            let me = q.get("me").cloned().unwrap_or_default();
            let (seq, picked) = {
                let c = coord.lock().unwrap();
                let first = c.log.front().map_or(c.seq + 1, |d| d.seq);
                if since < c.seq && first > since + 1 {
                    drop(c);
                    respond(&mut stream, 410, b"too far behind, download the net again\n");
                    return Ok(());
                }
                let picked: Vec<_> = c.log.iter().filter(|d| d.seq > since && d.from != me).map(|d| d.bytes.clone()).collect();
                (c.seq, picked)
            };
            let mut out = Vec::new();
            out.extend(seq.to_le_bytes());
            out.extend((picked.len() as u32).to_le_bytes());
            for d in &picked {
                out.extend((d.len() as u32).to_le_bytes());
                out.extend(d.iter());
            }
            respond(&mut stream, 200, &out);
        }
        ("POST", "/delta") => {
            let Some(delta) = decode(&body) else {
                respond(&mut stream, 500, b"bad delta\n");
                return Ok(());
            };
            let me = q.get("me").cloned().unwrap_or_default();
            let name = q.get("name").cloned().unwrap_or_else(|| me.clone());
            let chunk = Chunk::from_query(&q);
            let mut c = coord.lock().unwrap();
            c.episodes += chunk.episodes;
            if chunk.best > c.record.as_ref().map_or(0, |r| r.0) {
                c.record = Some((chunk.best, chunk.top, name.clone(), unix_now()));
                let (score, rank, who, at) = c.record.as_ref().unwrap();
                let _ = std::fs::write(format!("{}.best", c.path), format!("{score} {rank} {who} {at}\n"));
            }
            let w = c.workers.entry(name.clone()).or_insert_with(WorkerInfo::new);
            w.last_seen = Instant::now();
            w.total_episodes += chunk.episodes;
            w.recent.push_back(chunk);
            if w.recent.len() > RECENT {
                w.recent.pop_front();
            }
            // Workers all learn the common positions at once, so summing their changes would
            // move those weights several times over. Each change is weighted by the worker's
            // share of games instead, which averages the machines' nets.
            let scale = share(&c.workers, &name);
            let mut delta = scaled(&delta, scale);
            // Frozen stages are final: nothing a worker sends for them is taken.
            let frozen_end = (job_value(&c.job, "freeze").unwrap_or(0.0) as usize).min(c.net.stages()) * c.net.stage_size();
            let dropped = drop_frozen(&mut delta, frozen_end);
            if dropped > 0 {
                eprintln!("{name}: dropped {dropped} changes to frozen weights");
            }
            c.net.apply(&delta);
            c.seq += 1;
            let seq = c.seq;
            let bytes = encode(&delta);
            c.log_bytes += bytes.len();
            c.log.push_back(Delta { seq, from: me, bytes: Arc::new(bytes), at: Instant::now() });
            drop(c);
            respond(&mut stream, 200, format!("{seq} {scale} {frozen_end}").as_bytes());
        }
        ("POST", "/positions") => {
            // A worker's harvest of restart boards since its last chunk.
            let Some(entries) = Pool::decode(&body) else {
                respond(&mut stream, 400, b"bad positions\n");
                return Ok(());
            };
            let mut c = coord.lock().unwrap();
            let c = &mut *c;
            c.pool.merge(&entries, &mut c.rng);
            respond(&mut stream, 200, b"ok\n");
        }
        ("GET", "/positions") => {
            // A sample of the pool for a worker's restarts: `n` boards from keys at stage
            // `min_stage` or above, keys drawn uniformly.
            let n: usize = q.get("n").and_then(|s| s.parse().ok()).unwrap_or(4000).min(100_000);
            let min_stage: usize = q.get("min_stage").and_then(|s| s.parse().ok()).unwrap_or(1);
            let body = {
                let mut guard = coord.lock().unwrap();
                let c = &mut *guard;
                Pool::encode(&c.pool.sample(n, min_stage, &mut c.rng))
            };
            respond(&mut stream, 200, &body);
        }
        ("POST", "/shutdown") => {
            // Save the net and the delta log, then exit; systemd starts the new binary, and
            // workers carry on from the restored log instead of downloading the net again.
            let mut c = coord.lock().unwrap();
            c.save();
            match c.save_log() {
                Ok(()) => {
                    respond(&mut stream, 200, format!("saved at seq {}, restarting\n", c.seq).as_bytes());
                    std::process::exit(0);
                }
                Err(e) => respond(&mut stream, 500, format!("saving the delta log: {e}\n").as_bytes()),
            }
        }
        _ => respond(&mut stream, 404, b"no such endpoint\n"),
    }
    Ok(())
}

/// `g2048 coord --net MASTER --token-file F [--port 20490]`
pub fn coord(net_path: String, token: String, port: u16) {
    let net = NTuple::load(&net_path).unwrap_or_else(|e| panic!("loading {net_path}: {e}"));
    let seq: u64 = std::fs::read_to_string(format!("{net_path}.seq")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    let job = std::fs::read_to_string(format!("{net_path}.job")).unwrap_or_else(|_| DEFAULT_JOB.to_string());
    let episodes: u64 = std::fs::read_to_string(format!("{net_path}.episodes")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    let record = std::fs::read_to_string(format!("{net_path}.best")).ok().and_then(|s| {
        let f: Vec<&str> = s.split_whitespace().collect();
        Some((f.first()?.parse().ok()?, f.get(1)?.parse().ok()?, f.get(2)?.to_string(), f.get(3)?.parse().ok()?))
    });
    let (log, log_bytes) = load_log(&format!("{net_path}.log"), seq);
    eprintln!("restored {} deltas ({} MB) of the log", log.len(), log_bytes >> 20);
    let pool = Pool::load(&format!("{net_path}.pool"), POOL_CAP);
    eprintln!("restart pool: {} boards in {} chain states; master has {} stages", pool.total(), pool.summary().len(), net.stages());
    let coord = Arc::new(Mutex::new(Coord {
        net,
        path: net_path,
        seq,
        saved_seq: seq,
        saved_at: Instant::now(),
        log,
        log_bytes,
        job,
        workers: HashMap::new(),
        episodes,
        record,
        pool,
        rng: Rng(unix_now() | 1),
    }));
    let saver = coord.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(SAVE_EVERY);
        saver.lock().unwrap().save();
    });
    let token = Arc::new(token);
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).expect("binding port");
    eprintln!("coordinator on 127.0.0.1:{port}, master seq {seq}");
    for stream in listener.incoming().flatten() {
        let (coord, token) = (coord.clone(), token.clone());
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &coord, &token) {
                eprintln!("request failed: {e}");
            }
        });
    }
}

// ---------------------------------------------------------------- worker

#[derive(Clone)]
struct Client {
    url: String,
    auth: String,
    tmp: std::path::PathBuf,
    /// Seconds before curl gives up; None for the big transfers, which may take minutes.
    max_time: Option<u32>,
}

impl Client {
    /// Runs curl, returning the HTTP status and the path the response body landed in.
    fn call(&self, method: &str, path: &str, body: Option<&std::path::Path>) -> Result<(u16, std::path::PathBuf), String> {
        let out = self.tmp.join(format!("resp-{}", path.split('?').next().unwrap_or("x").trim_matches('/')));
        let url = format!("{}{path}", self.url);
        let mut cmd = std::process::Command::new("curl");
        cmd.args(["-sS", "--connect-timeout", "20", "-X", method, "-H", &self.auth, "-o"]).arg(&out).args(["-w", "%{http_code}"]);
        if let Some(t) = self.max_time {
            cmd.args(["--max-time", &t.to_string()]);
        }
        if let Some(b) = body {
            cmd.args(["-H", "Content-Type: application/octet-stream", "--data-binary"]).arg(format!("@{}", b.display()));
        }
        let r = cmd.arg(&url).output().map_err(|e| format!("running curl: {e}"))?;
        if !r.status.success() {
            return Err(format!("curl {method} {path}: {}", String::from_utf8_lossy(&r.stderr).trim()));
        }
        let code = String::from_utf8_lossy(&r.stdout).trim().parse().unwrap_or(0);
        Ok((code, out))
    }

    fn text(&self, method: &str, path: &str, body: Option<&std::path::Path>) -> Result<String, String> {
        match self.call(method, path, body)? {
            (200, p) => std::fs::read_to_string(p).map_err(|e| e.to_string()),
            (code, p) => Err(format!("{method} {path}: HTTP {code} {}", std::fs::read_to_string(p).unwrap_or_default().trim())),
        }
    }
}

/// Downloads the master net: 8-byte seq, then the weights file. A broken transfer leaves
/// its partial file behind, and the next call resumes it from the same saved version.
fn download_net(c: &Client) -> Result<(NTuple, u64), String> {
    let path = c.tmp.join("net.part");
    let have = std::fs::metadata(&path).map_or(0, |m| m.len());
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-sS", "--connect-timeout", "20", "-H", &c.auth, "-o"]).arg(&path).args(["-w", "%{http_code}"]);
    let url = if have >= 8 {
        let mut seq = [0u8; 8];
        std::fs::File::open(&path).and_then(|mut f| f.read_exact(&mut seq)).map_err(|e| e.to_string())?;
        let seq = u64::from_le_bytes(seq);
        eprintln!("resuming the master net download at {} MB (update {seq})...", have >> 20);
        cmd.args(["-C", &have.to_string()]);
        format!("{}/net?seq={seq}", c.url)
    } else {
        eprintln!("downloading the master net (large, one time)...");
        let _ = std::fs::remove_file(&path);
        format!("{}/net", c.url)
    };
    let r = cmd.arg(&url).output().map_err(|e| format!("running curl: {e}"))?;
    let code: u16 = String::from_utf8_lossy(&r.stdout).trim().parse().unwrap_or(0);
    // 416: the file is already whole (a curl that outlived its worker finished it), or its
    // version is gone from the coordinator. Use it if it loads, else start over.
    let whole = code == 416;
    if !whole && !r.status.success() {
        return Err(format!("curl GET /net: {} (keeping the partial file to resume)", String::from_utf8_lossy(&r.stderr).trim()));
    }
    if !whole && code != 200 && code != 206 {
        let _ = std::fs::remove_file(&path);
        return Err(format!("GET /net: HTTP {code}"));
    }
    let mut f = BufReader::new(std::fs::File::open(&path).map_err(|e| e.to_string())?);
    let mut seq = [0u8; 8];
    f.read_exact(&mut seq).map_err(|e| e.to_string())?;
    let net = NTuple::load_from(f);
    let _ = std::fs::remove_file(&path);
    match net {
        Ok(n) => Ok((n, u64::from_le_bytes(seq))),
        Err(_) if whole => Err("GET /net: the partial download is stale, starting over".into()),
        Err(e) => Err(e.to_string()),
    }
}

/// A worker's copy of the master as it last knew it, from the first trainable weight on
/// (frozen stages never change, so they are not kept twice). net - mirror is training not
/// yet sent.
struct Mirror {
    start: usize,
    w: Vec<f32>,
}

impl Mirror {
    /// Taken right after a sync, when the net holds exactly what the master holds. The
    /// net's freeze must already be the job's.
    fn of(net: &NTuple) -> Mirror {
        let start = net.frozen_end();
        Mirror { start, w: net.snapshot_from(start) }
    }

    fn add(&mut self, delta: &[(u32, f32)]) {
        for &(i, d) in delta {
            if let Some(x) = (i as usize).checked_sub(self.start).and_then(|j| self.w.get_mut(j)) {
                *x += d;
            }
        }
    }
}

/// Applies every other worker's deltas since `since`; None means too far behind (re-download).
fn pull(c: &Client, net: &NTuple, mirror: &mut Mirror, since: u64, me: &str) -> Result<Option<u64>, String> {
    let (code, path) = c.call("GET", &format!("/deltas?since={since}&me={me}"), None)?;
    if code == 410 {
        return Ok(None);
    }
    if code != 200 {
        return Err(format!("GET /deltas: HTTP {code}"));
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let bad = || "malformed deltas response".to_string();
    let seq = u64::from_le_bytes(bytes.get(..8).ok_or_else(bad)?.try_into().unwrap());
    let n = u32::from_le_bytes(bytes.get(8..12).ok_or_else(bad)?.try_into().unwrap());
    let mut pos = 12;
    for _ in 0..n {
        let len = u32::from_le_bytes(bytes.get(pos..pos + 4).ok_or_else(bad)?.try_into().unwrap()) as usize;
        pos += 4;
        let delta = decode(bytes.get(pos..pos + len).ok_or_else(bad)?).ok_or_else(bad)?;
        net.apply(&delta);
        mirror.add(&delta);
        pos += len;
    }
    Ok(Some(seq))
}

/// Fetches a sample of the master's restart pool into `pool`.
fn fetch_positions(c: &Client, pool: &Pool, n: usize, min_stage: usize, rng: &mut Rng) -> Result<usize, String> {
    let (code, path) = c.call("GET", &format!("/positions?n={n}&min_stage={min_stage}"), None)?;
    if code != 200 {
        return Err(format!("GET /positions: HTTP {code}"));
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let entries = Pool::decode(&bytes).ok_or("malformed positions response")?;
    let got = entries.iter().map(|e| e.2.len()).sum();
    pool.merge(&entries, rng);
    Ok(got)
}

/// Machine load for the status page, as query pairs: cpu (%), temp (C), ram, ramtot and
/// rss (this process) in MiB. Anything the OS won't tell us is left out.
#[cfg_attr(windows, allow(dead_code))]
struct SysSampler {
    last_cpu: Option<(u64, u64)>,
}

impl SysSampler {
    #[cfg(not(windows))]
    fn sample(&mut self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        // cpu: busy share of all jiffies since the previous sample.
        if let Some(line) = std::fs::read_to_string("/proc/stat").ok().and_then(|s| s.lines().next().map(String::from)) {
            let v: Vec<u64> = line.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
            let total: u64 = v.iter().take(8).sum();
            let idle = v.get(3).unwrap_or(&0) + v.get(4).unwrap_or(&0);
            if let Some((t0, i0)) = self.last_cpu.replace((total, idle)) {
                if total > t0 {
                    out.push(("cpu", format!("{:.0}", 100.0 * (1.0 - (idle - i0) as f64 / (total - t0) as f64))));
                }
            }
        }
        let kb = |file: &str, key: &str| -> Option<f64> {
            let s = std::fs::read_to_string(file).ok()?;
            s.lines().find_map(|l| l.strip_prefix(key)?.split_whitespace().next()?.parse().ok())
        };
        if let (Some(t), Some(a)) = (kb("/proc/meminfo", "MemTotal:"), kb("/proc/meminfo", "MemAvailable:")) {
            out.push(("ram", format!("{:.0}", (t - a) / 1024.0)));
            out.push(("ramtot", format!("{:.0}", t / 1024.0)));
        }
        if let Some(r) = kb("/proc/self/status", "VmRSS:") {
            out.push(("rss", format!("{:.0}", r / 1024.0)));
        }
        // temp: the CPU package sensor, else the hottest thermal zone.
        let mut cpu_temp = None;
        let mut hottest: Option<f64> = None;
        for dir in std::fs::read_dir("/sys/class/hwmon").into_iter().flatten().flatten() {
            let name = std::fs::read_to_string(dir.path().join("name")).unwrap_or_default();
            let Some(t) = std::fs::read_to_string(dir.path().join("temp1_input")).ok().and_then(|v| v.trim().parse::<f64>().ok()) else { continue };
            match name.trim() {
                "k10temp" | "coretemp" | "zenpower" | "cpu_thermal" => cpu_temp = Some(t / 1000.0),
                "acpitz" => hottest = Some(hottest.map_or(t / 1000.0, |h| h.max(t / 1000.0))),
                _ => {}
            }
        }
        if let Some(t) = cpu_temp.or(hottest) {
            out.push(("temp", format!("{t:.0}")));
        }
        out
    }

    /// Windows: one PowerShell call. Temperature needs admin rights, so it is often missing.
    #[cfg(windows)]
    fn sample(&mut self) -> Vec<(&'static str, String)> {
        let script = format!(
            "$o=Get-CimInstance Win32_OperatingSystem;\
             $c=(Get-CimInstance Win32_Processor|Measure-Object LoadPercentage -Average).Average;\
             $t=try{{(Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature -EA Stop|Measure-Object CurrentTemperature -Maximum).Maximum/10-273.15}}catch{{''}};\
             $p=(Get-Process -Id {}).WorkingSet64;\
             \"$c;$($o.TotalVisibleMemorySize);$($o.FreePhysicalMemory);$t;$p\"",
            std::process::id()
        );
        // A hung WMI query must not stall the heartbeat: give up after 20 seconds.
        let Ok(mut child) = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        else {
            return Vec::new();
        };
        let start = Instant::now();
        while child.try_wait().ok().flatten().is_none() {
            if start.elapsed() > Duration::from_secs(20) {
                let _ = child.kill();
                let _ = child.wait();
                return Vec::new();
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let mut text = String::new();
        let _ = child.stdout.take().map(|mut o| o.read_to_string(&mut text));
        let f: Vec<Option<f64>> = text.trim().split(';').map(|x| x.trim().parse().ok()).collect();
        let get = |i: usize| f.get(i).copied().flatten();
        let mut out = Vec::new();
        if let Some(c) = get(0) {
            out.push(("cpu", format!("{c:.0}")));
        }
        if let (Some(t), Some(free)) = (get(1), get(2)) {
            out.push(("ram", format!("{:.0}", (t - free) / 1024.0)));
            out.push(("ramtot", format!("{:.0}", t / 1024.0)));
        }
        if let Some(t) = get(3) {
            out.push(("temp", format!("{t:.0}")));
        }
        if let Some(p) = get(4) {
            out.push(("rss", format!("{:.0}", p / 1048576.0)));
        }
        out
    }
}

/// Reports what this worker is doing and its machine's load every 10 seconds.
fn heartbeat(c: Client, name: String, threads: Arc<Mutex<usize>>, state: Arc<Mutex<&'static str>>) {
    let mut sampler = SysSampler { last_cpu: None };
    let empty = c.tmp.join("beat.bin");
    let _ = std::fs::write(&empty, b"");
    loop {
        let mut q = format!("/beat?name={name}&threads={}&state={}", threads.lock().unwrap(), state.lock().unwrap());
        for (k, v) in sampler.sample() {
            q += &format!("&{k}={v}");
        }
        let _ = c.call("POST", &q, Some(&empty));
        std::thread::sleep(Duration::from_secs(10));
    }
}

fn machine_name() -> String {
    for var in ["G2048_NAME", "COMPUTERNAME", "HOSTNAME"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                return v.to_lowercase();
            }
        }
    }
    std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "worker".into())
}

/// Workers older than the job's `version=` restart into the new build on their own.
pub const BUILD: u32 = 10;
pub const CHILD_ENV: &str = "G2048_WORKER_CHILD";
/// The exit code a worker uses to ask its supervisor for the new build.
const UPDATE_EXIT: i32 = 42;
/// The exit code for `stop.NAME=1` in the job: the supervisor exits too.
const STOP_EXIT: i32 = 43;

/// Runs the worker as a child process and restarts it when it exits: after a crash, or
/// with the new build when the job asks for one. On Windows it first swaps in the exe the
/// server publishes (a running exe can be renamed, not overwritten). On Linux it swaps in
/// the published `g2048-linux-<arch>` when that is a newer build than the binary on disk,
/// so machines that don't build from source (the OptiPlex) update themselves; machines
/// that rebuild in place, or reach no /files/, keep the binary on disk.
pub fn supervise(url: &str) {
    let exe = std::env::current_exe().expect("finding own exe");
    let args: Vec<String> = std::env::args().skip(1).collect();
    loop {
        let started = Instant::now();
        let code = std::process::Command::new(&exe).args(&args).env(CHILD_ENV, "1").status().ok().and_then(|s| s.code());
        match code {
            Some(STOP_EXIT) => {
                eprintln!("stopped from the server");
                return;
            }
            Some(UPDATE_EXIT) => {
                // Asked again right after an update: the new build isn't published yet.
                if started.elapsed() < Duration::from_secs(300) {
                    std::thread::sleep(Duration::from_secs(300));
                }
                if cfg!(windows) {
                    eprintln!("downloading the new version...");
                    let new = exe.with_extension("new.exe");
                    let got = std::process::Command::new("curl")
                        .args(["-sSf", "--connect-timeout", "20", "-o"])
                        .arg(&new)
                        .arg(format!("{}/files/g2048.exe", url.trim_end_matches('/')))
                        .status()
                        .is_ok_and(|s| s.success());
                    // This supervisor keeps running from the exe it renamed in an earlier
                    // update, and Windows won't delete or replace that file, so each swap
                    // gets a fresh name and old ones are cleared once nothing runs them.
                    let dir = exe.parent().map(|d| d.to_path_buf()).unwrap_or_default();
                    for f in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                        if f.file_name().to_string_lossy().starts_with("g2048.old") {
                            let _ = std::fs::remove_file(f.path());
                        }
                    }
                    let old = exe.with_extension(format!("old-{}.exe", unix_now()));
                    if !got || std::fs::rename(&exe, &old).is_err() || std::fs::rename(&new, &exe).is_err() {
                        eprintln!("update failed, restarting the current version in 60s");
                        std::thread::sleep(Duration::from_secs(60));
                    }
                } else {
                    update_linux(&exe, url);
                }
            }
            _ => {
                eprintln!("worker stopped ({code:?}), restarting in 10s");
                std::thread::sleep(Duration::from_secs(10));
            }
        }
    }
}

/// The build number a g2048 binary reports with `g2048 build`, or 0 if it doesn't run.
fn build_of(exe: &std::path::Path) -> u32 {
    std::process::Command::new(exe)
        .arg("build")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Downloads the published Linux build and renames it over `exe` (allowed while it runs)
/// when it reports a newer build than the binary on disk.
fn update_linux(exe: &std::path::Path, url: &str) {
    let new = exe.with_extension("new");
    let file = format!("{}/files/g2048-linux-{}", url.trim_end_matches('/'), std::env::consts::ARCH);
    let got = std::process::Command::new("curl").args(["-sSf", "--connect-timeout", "20", "-o"]).arg(&new).arg(&file).status().is_ok_and(|s| s.success());
    if !got {
        let _ = std::fs::remove_file(&new);
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&new, std::fs::Permissions::from_mode(0o755));
    }
    let (have, offered) = (build_of(exe), build_of(&new));
    if offered > have && std::fs::rename(&new, exe).is_ok() {
        eprintln!("updated build {have} -> {offered}");
    } else {
        let _ = std::fs::remove_file(&new);
    }
}

/// `g2048 worker --url URL --token T [--name N] [--threads N] [--cache DIR]`: trains forever.
pub fn worker(url: String, token: String, name: Option<String>, threads: usize, cache: std::path::PathBuf) {
    std::fs::create_dir_all(&cache).expect("creating cache dir");
    let name = name.unwrap_or_else(machine_name);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    // Unique per run, so a restarted worker never skips deltas it didn't make itself.
    let me = format!("{name}-{:08x}", (nanos ^ std::process::id() as u64) as u32);
    let c = Client { url: url.trim_end_matches('/').to_string(), auth: format!("Authorization: Bearer {token}"), tmp: cache.clone(), max_time: None };
    let cached = cache.join("net.bin");
    let cached_seq = cache.join("net.seq");
    let pool_file = cache.join("pool.bin");
    eprintln!("worker {me} using {threads} threads");
    let status = Arc::new(Mutex::new("starting"));
    let cores = Arc::new(Mutex::new(threads));
    {
        let cores = cores.clone();
        let c = Client { max_time: Some(20), ..c.clone() };
        let (name, status) = (name.clone(), status.clone());
        std::thread::spawn(move || heartbeat(c, name, cores, status));
    }
    let set = |s: &'static str| *status.lock().unwrap() = s;

    // The local net, the master as this worker last knew it (`mirror`, taken once the job's
    // freeze is known) and the master's seq. While one chunk trains, a background thread
    // sends the previous chunk's changes and pulls everyone else's, so no core sits idle;
    // the mirror travels with that thread and is None here meanwhile.
    let mut net: Option<Arc<NTuple>> = None;
    let mut mirror: Option<Mirror> = None;
    let mut seq = 0;
    if let Some(n) = NTuple::load(cached.to_str().unwrap()).ok() {
        if let Some(s) = std::fs::read_to_string(&cached_seq).ok().and_then(|s| s.trim().parse().ok()) {
            eprintln!("resuming from the cached net at seq {s}");
            (net, seq) = (Some(Arc::new(n)), s);
        }
    }
    let mut syncing: Option<std::thread::JoinHandle<(Mirror, Option<u64>)>> = None;
    // Restart boards: this machine's harvest plus samples of the master's pool, kept
    // across restarts.
    let pool = Arc::new(Pool::load(pool_file.to_str().unwrap(), 2000));
    eprintln!("restart pool: {} boards", pool.total());
    let mut last_cache_save = Instant::now();
    let mut seed = nanos;
    #[cfg(windows)]
    let mut priority = String::new();
    let backoff = || {
        set("retrying");
        std::thread::sleep(Duration::from_secs(30));
    };
    let save_pool = |pool: &Pool| {
        if let Err(e) = pool.save(pool_file.to_str().unwrap()) {
            eprintln!("saving the pool: {e}");
        }
    };

    loop {
        let job = match c.text("GET", "/job", None) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("{e}; retrying in 30s");
                backoff();
                continue;
            }
        };
        if job_value(&job, "version").unwrap_or(0.0) > BUILD as f64 {
            set("updating");
            eprintln!("a new version is out; finishing the last sync and restarting");
            if let Some(h) = syncing.take() {
                let (_, s) = h.join().expect("sync thread panicked");
                // Keep the net so the new build resumes without the big download.
                if let (Some(n), Some(s)) = (&net, s) {
                    if n.save(cached.to_str().unwrap()).is_ok() {
                        let _ = std::fs::write(&cached_seq, s.to_string());
                    }
                }
            }
            save_pool(&pool);
            std::process::exit(UPDATE_EXIT);
        }
        if job_value(&job, &format!("stop.{name}")).unwrap_or(0.0) > 0.0 {
            set("stopped");
            eprintln!("stopped from the server");
            if let Some(h) = syncing.take() {
                let _ = h.join();
            }
            save_pool(&pool);
            std::process::exit(STOP_EXIT);
        }
        if job_value(&job, "pause").unwrap_or(0.0) > 0.0 {
            set("paused");
            std::thread::sleep(Duration::from_secs(60));
            continue;
        }
        let freeze = job_value(&job, "freeze").unwrap_or(0.0) as usize;
        // No net yet (or too far behind): download the master and catch up before training.
        if net.is_none() {
            set("downloading net");
            let n = match download_net(&c) {
                Ok((n, s)) => {
                    seq = s;
                    n
                }
                Err(e) => {
                    eprintln!("{e}; retrying in 30s");
                    backoff();
                    continue;
                }
            };
            n.set_frozen(freeze);
            let mut m = Mirror::of(&n);
            set("syncing");
            match pull(&c, &n, &mut m, seq, &me) {
                Ok(Some(s)) => seq = s,
                Ok(None) => continue,
                Err(e) => {
                    eprintln!("{e}; retrying in 30s");
                    backoff();
                    continue;
                }
            }
            net = Some(Arc::new(n));
            mirror = Some(m);
        }
        let n = net.clone().unwrap();
        n.set_frozen(freeze);
        // The master grew a stage (or this copy is from another net): deltas for the new
        // stage would be dropped here unnoticed, so download the master again.
        if job_value(&job, "net_stages").is_some_and(|k| k as usize != n.stages()) {
            eprintln!("the master has {} stages, this copy {}; downloading it again", job_value(&job, "net_stages").unwrap_or(0.0), n.stages());
            if let Some(h) = syncing.take() {
                let _ = h.join();
            }
            (net, mirror) = (None, None);
            continue;
        }
        if syncing.is_none() {
            match &mirror {
                // Resumed from the cache: the cached net is the master as last known.
                None => mirror = Some(Mirror::of(&n)),
                Some(m) if m.start != n.frozen_end() => {
                    eprintln!("freeze changed; downloading the master again");
                    (net, mirror) = (None, None);
                    continue;
                }
                _ => {}
            }
        }

        // Train one chunk. `freeze=N` keeps stages below N fixed while later stages learn.
        let alpha = job_value(&job, "alpha").unwrap_or(0.00015625) as f32;
        let restart = job_value(&job, "restart").unwrap_or(0.5) as f32;
        let restart_stage = job_value(&job, "restart_stage").unwrap_or(1.0) as usize;
        let pool_n = job_value(&job, "pool_n").unwrap_or(4000.0) as usize;
        pool.set_cap(job_value(&job, "pool_cap").unwrap_or(2000.0) as usize);
        let secs = job_value(&job, "secs").unwrap_or(120.0);
        let send_mb = job_value(&job, "send_mb").unwrap_or(16.0);
        // `priority.NAME=high` etc. sets a Windows machine's priority class. Below normal (the
        // default) still uses every core when idle but lets the desktop go first.
        #[cfg(windows)]
        {
            // Only a known class name ever reaches PowerShell.
            const CLASSES: [&str; 6] = ["Idle", "BelowNormal", "Normal", "AboveNormal", "High", "RealTime"];
            let asked = job_text(&job, &format!("priority.{name}")).unwrap_or("BelowNormal");
            let want = CLASSES.iter().find(|c| c.eq_ignore_ascii_case(asked)).unwrap_or(&"BelowNormal").to_string();
            if want != priority {
                let _ = std::process::Command::new("powershell")
                    .args(["-NoProfile", "-NonInteractive", "-Command", &format!("(Get-Process -Id {}).PriorityClass='{want}'", std::process::id())])
                    .status();
                priority = want;
            }
        }
        // `threads.NAME=N` in the job caps one machine's cores.
        let threads = job_value(&job, &format!("threads.{name}")).map_or(threads, |t| (t as usize).clamp(1, threads));
        *cores.lock().unwrap() = threads;
        // [episodes, fresh, score, reached 2048..65536 (6), best, top, restarts from stage 1..3 as (count, progressed)]
        let counters: Vec<AtomicU64> = (0..17).map(|_| AtomicU64::new(0)).collect();
        let start = Instant::now();
        set("training");
        seed = seed.wrapping_add(0x9E37_79B9);
        ntuple::train_parallel(&n, &pool, alpha, restart, restart_stage, seed, threads, u64::MAX, Some(start + Duration::from_secs_f64(secs)), &|_, e| {
            counters[0].fetch_add(1, Relaxed);
            if e.fresh {
                counters[1].fetch_add(1, Relaxed);
                counters[2].fetch_add(e.score, Relaxed);
                for k in 0..6 {
                    if e.max_rank >= 11 + k as u8 {
                        counters[3 + k].fetch_add(1, Relaxed);
                    }
                }
                counters[9].fetch_max(e.score, Relaxed);
                counters[10].fetch_max(e.max_rank as u64, Relaxed);
            } else {
                let s = (e.start_rank as usize).saturating_sub(13).clamp(1, 3);
                counters[11 + 2 * (s - 1)].fetch_add(1, Relaxed);
                counters[12 + 2 * (s - 1)].fetch_add(e.progressed() as u64, Relaxed);
            }
        });
        let v: Vec<u64> = counters.iter().map(|a| a.load(Relaxed)).collect();
        let chunk = Chunk {
            secs: start.elapsed().as_secs_f64(),
            episodes: v[0],
            fresh: v[1],
            score: v[2],
            reached: [v[3], v[4], v[5], v[6], v[7], v[8]],
            best: v[9],
            top: v[10] as u8,
            restarts: [[v[11], v[12]], [v[13], v[14]], [v[15], v[16]]],
        };
        let harvest = pool.take_outbox();
        drop(n);

        // The previous chunk's sync must land before this chunk's changes are measured.
        if let Some(h) = syncing.take() {
            set("syncing");
            let (m, s) = h.join().expect("sync thread panicked");
            mirror = Some(m);
            match s {
                Some(s) => seq = s,
                None => {
                    eprintln!("too far behind the master, downloading it again");
                    (net, mirror) = (None, None);
                    continue;
                }
            }
        }
        let n = net.as_mut().unwrap();
        let m = mirror.take().expect("mirror present after sync");
        if m.start != n.frozen_end() {
            // The job's freeze moved while this chunk trained; this chunk's changes are not
            // worth sending against a copy whose trainable range no longer matches.
            eprintln!("freeze changed; downloading the master again");
            (net, mirror) = (None, None);
            continue;
        }
        // TC fine-tuning phase: its per-weight accumulators stay local to each worker.
        if job_value(&job, "tc").unwrap_or(0.0) > 0.0 && !n.tc_enabled() {
            eprintln!("switching to TC learning");
            Arc::get_mut(n).expect("net still shared").enable_tc();
        }
        if last_cache_save.elapsed() > Duration::from_secs(1800) {
            if n.save(cached.to_str().unwrap()).is_ok() {
                let _ = std::fs::write(&cached_seq, seq.to_string());
            }
            save_pool(&pool);
            last_cache_save = Instant::now();
        }
        set("preparing update");
        let unsent = n.diff_from(&m.w, m.start);
        let pending = unsent.len();
        let sent = pick_largest(unsent, send_mb * 1e6);
        let bytes = encode(&sent);
        let file = cache.join("delta.bin");
        std::fs::write(&file, &bytes).expect("writing delta");
        let positions = cache.join("positions.bin");
        let harvested: usize = harvest.iter().map(|e| e.2.len()).sum();
        if harvested > 0 {
            std::fs::write(&positions, Pool::encode(&harvest)).expect("writing positions");
        }

        let (c, n, me, name, pool, mut m) = (c.clone(), n.clone(), me.clone(), name.clone(), pool.clone(), m);
        syncing = Some(std::thread::spawn(move || {
            // Keep retrying: dropping a delta would leave this copy ahead of the master for good.
            let reply = loop {
                match c.text("POST", &format!("/delta?me={me}&name={name}&{}", chunk.to_query()), Some(&file)) {
                    Ok(r) => break r,
                    Err(e) => {
                        eprintln!("{e}; retrying in 30s");
                        std::thread::sleep(Duration::from_secs(30));
                    }
                }
            };
            // The master took `scale` of what was sent and nothing in its frozen range; keep
            // the same here so this copy matches it.
            let mut words = reply.split_whitespace().skip(1);
            let scale: f32 = words.next().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let frozen_end: usize = words.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let mut taken = scaled(&sent, scale);
            drop_frozen(&mut taken, frozen_end);
            n.apply(&settle(&sent, &taken));
            m.add(&taken);
            eprintln!(
                "{} games in {:.0}s, mean score {:.0}; sent {:.1} MB ({} of {} changed weights), share {scale:.2}, {harvested} boards harvested",
                chunk.episodes,
                chunk.secs,
                if chunk.fresh > 0 { chunk.score as f64 / chunk.fresh as f64 } else { 0.0 },
                bytes.len() as f64 / 1e6,
                sent.len(),
                pending
            );
            // The harvest is a sample; losing one is fine, so no retry loop.
            if harvested > 0 {
                if let Err(e) = c.text("POST", &format!("/positions?me={me}"), Some(&positions)) {
                    eprintln!("{e}");
                }
            }
            let s = loop {
                match pull(&c, &n, &mut m, seq, &me) {
                    Ok(s) => break s,
                    Err(e) => {
                        eprintln!("{e}; retrying in 30s");
                        std::thread::sleep(Duration::from_secs(30));
                    }
                }
            };
            let mut rng = Rng(seq.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ nanos | 1);
            match fetch_positions(&c, &pool, pool_n, restart_stage, &mut rng) {
                Ok(got) => eprintln!("pool: {got} boards fetched, {} held", pool.total()),
                Err(e) => eprintln!("{e}"),
            }
            (m, s)
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_roundtrip() {
        let d = vec![(0, 1.5), (5, -2.0), (300, 0.25), (400_000_000, 7.0)];
        assert_eq!(decode(&encode(&d)).unwrap(), d);
        assert_eq!(decode(&encode(&[])).unwrap(), vec![]);
    }

    #[test]
    fn chunk_query_roundtrip_and_old_workers() {
        let c = Chunk { secs: 12.5, episodes: 100, fresh: 60, score: 9000, reached: [50, 40, 30, 20, 3, 1], best: 700, top: 15, restarts: [[30, 4], [10, 1], [0, 0]] };
        let q = format!("/delta?me=x&{}", c.to_query());
        assert_eq!(Chunk::from_query(&parse_query(&q).1), c);
        // A build-7 worker sends five reached counts and no restart fields.
        let old = parse_query("/delta?secs=1.0&episodes=5&fresh=5&score=10&reached=1,2,3,4,5&best=9&top=13").1;
        let c = Chunk::from_query(&old);
        assert_eq!((c.reached, c.restarts), ([1, 2, 3, 4, 5, 0], [[0, 0]; 3]));
    }

    #[test]
    fn frozen_weights_are_dropped_and_settled() {
        let sent = vec![(1, 0.5), (10, -2.0), (20, 4.0)];
        let mut taken = scaled(&sent, 0.5);
        assert_eq!(drop_frozen(&mut taken, 10), 1);
        assert_eq!(taken, vec![(10, -1.0), (20, 2.0)]);
        // The worker takes back what the master did not: all of index 1, half of the rest.
        assert_eq!(settle(&sent, &taken), vec![(1, -0.5), (10, 1.0), (20, -2.0)]);
        assert_eq!(drop_frozen(&mut taken, 0), 0);
    }

    #[test]
    fn mirror_tracks_only_trainable_weights() {
        let mut net = NTuple::new(0.0, 1, &ntuple::TUPLES_4);
        net.expand_stages(2);
        net.set_frozen(1);
        let mut m = Mirror::of(&net);
        assert_eq!((m.start, m.w.len()), (net.stage_size(), net.stage_size()));
        m.add(&[(3, 1.0), (net.stage_size() as u32 + 3, 2.0)]);
        assert_eq!(m.w[3], 2.0);
        net.apply(&[(net.stage_size() as u32 + 3, 2.0)]);
        assert!(net.diff_from(&m.w, m.start).is_empty());
    }

    #[test]
    fn pick_largest_keeps_the_biggest_within_budget() {
        let raw = vec![(1, 0.001), (2, -5.0), (3, 0.5), (4, 3.0)];
        let got = pick_largest(raw, 2.0 * BYTES_PER_ENTRY);
        assert_eq!(got, vec![(2, -5.0), (4, 3.0)]);
    }
}
