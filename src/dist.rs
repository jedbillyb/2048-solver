//! Training spread over several machines.
//!
//! One coordinator (the always-on server) holds the master weights. Each worker trains
//! its own copy for a couple of minutes, sends back only the weights that moved, then
//! pulls every other worker's changes, so all machines keep learning on nearly the same
//! net. Workers talk HTTP through the system `curl` (shipped with Windows 10+ too), which
//! keeps TLS out of this dependency-free crate; nginx terminates TLS for the coordinator.

use crate::ntuple::{self, NTuple, RestartPool};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
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
#[cfg_attr(not(windows), allow(dead_code))]
fn job_text<'a>(job: &'a str, key: &str) -> Option<&'a str> {
    job.lines().find_map(|l| Some(l.strip_prefix(key)?.strip_prefix('=')?.trim()))
}

fn job_value(job: &str, key: &str) -> Option<f64> {
    job.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('=')?.trim().parse().ok())
}

/// Totals of the fresh games in one training chunk: reached counts are for 2048 .. 32768.
#[derive(Clone, Copy, Default)]
struct Chunk {
    secs: f64,
    episodes: u64,
    fresh: u64,
    score: u64,
    reached: [u64; 5],
}

impl Chunk {
    fn to_query(self) -> String {
        let r = self.reached.map(|x| x.to_string()).join(",");
        format!("secs={:.1}&episodes={}&fresh={}&score={}&reached={r}", self.secs, self.episodes, self.fresh, self.score)
    }

    fn from_query(q: &HashMap<String, String>) -> Chunk {
        let num = |k: &str| q.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        let mut reached = [0; 5];
        for (slot, v) in reached.iter_mut().zip(q.get("reached").map_or("", |s| s).split(',')) {
            *slot = v.parse().unwrap_or(0);
        }
        Chunk { secs: num("secs"), episodes: num("episodes") as u64, fresh: num("fresh") as u64, score: num("score") as u64, reached }
    }

    fn add(&mut self, o: &Chunk) {
        self.secs += o.secs;
        self.episodes += o.episodes;
        self.fresh += o.fresh;
        self.score += o.score;
        for (a, b) in self.reached.iter_mut().zip(o.reached) {
            *a += b;
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
}

impl Coord {
    fn save(&mut self) {
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

    fn status(&self, color: bool) -> String {
        let paint = |code: &str, text: String| if color { format!("\x1b[{code}m{text}\x1b[0m") } else { text };
        let job = |k: &str| job_value(&self.job, k);
        let mut s = String::new();

        // Headline: progress towards the episode goal of this training stage.
        let live = |w: &WorkerInfo| w.last_seen.elapsed() < Duration::from_secs(600);
        let rate: f64 = self.workers.values().filter(|w| live(w)).map(recent_rate).sum();
        let goal = job("goal").unwrap_or(100e6);
        let done = (self.episodes as f64 / goal).min(1.0);
        let bar = (done * 30.0).round() as usize;
        let eta = if rate > 0.0 { duration((goal - self.episodes as f64).max(0.0) / rate) } else { "-".into() };
        s += &paint("1", "2048 TRAINING FARM".into());
        s += &format!("   alpha {}   TC {}{}\n",
            job("alpha").unwrap_or(0.0),
            if job("tc").unwrap_or(0.0) > 0.0 { "on" } else { "off" },
            if job("pause").unwrap_or(0.0) > 0.0 { paint("33", "   PAUSED".into()) } else { String::new() });
        s += &format!("progress  {}{}  {:.1}%  {} / {} games  ETA {eta}\n\n",
            paint("32", "#".repeat(bar)), paint("2", ".".repeat(30 - bar)), done * 100.0,
            millions(self.episodes as f64), millions(goal));

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
        s += &paint("1", format!("{:<12}{:>9}{:>12}{:>8}{:>8}{:>8}{:>8}{:>8}", "RESULTS", "games/s", "mean score", "2048", "4096", "8192", "16384", "32768"));
        s += "\n";
        let row = |label: &str, c: &Chunk, rate: f64| {
            let pct = |x: u64| if c.fresh == 0 { 0.0 } else { 100.0 * x as f64 / c.fresh as f64 };
            let r = c.reached;
            let mean = if c.fresh == 0 { 0.0 } else { c.score as f64 / c.fresh as f64 };
            format!("{label:<12}{:>9}{:>12}{:>7.1}%{:>7.1}%{:>7.1}%{:>7.2}%{:>7.2}%\n",
                thousands(rate), thousands(mean), pct(r[0]), pct(r[1]), pct(r[2]), pct(r[3]), pct(r[4]))
        };
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
        }
        s += &paint("1", row("ALL", &total, rate));
        s += &paint("2", format!(
            "\nmaster: update {}, saved {} ago, {} updates held ({} MB). Scores are over each machine's last {RECENT} chunks.\n",
            self.seq, duration(self.saved_at.elapsed().as_secs_f64()), self.log.len(), self.log_bytes >> 20));
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

fn parse_query(target: &str) -> (String, HashMap<String, String>) {
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    let map = q.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
    (path.to_string(), map)
}

fn respond(stream: &mut TcpStream, code: u16, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        401 => "Unauthorized",
        404 => "Not Found",
        410 => "Gone",
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
    let (mut len, mut authed) = (0usize, false);
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? <= 2 {
            break;
        }
        let (k, v) = h.split_once(':').unwrap_or(("", ""));
        match k.trim().to_ascii_lowercase().as_str() {
            "content-length" => len = v.trim().parse().unwrap_or(0),
            "authorization" => authed = v.trim() == format!("Bearer {token}"),
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
            let job = coord.lock().unwrap().job.clone();
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
            // Open under the lock so the file and its seq match even if a save lands mid-download.
            let (mut file, seq) = {
                let mut c = coord.lock().unwrap();
                if c.saved_seq != c.seq && c.saved_at.elapsed() > Duration::from_secs(60) {
                    c.save();
                }
                (std::fs::File::open(&c.path)?, c.saved_seq)
            };
            let size = file.metadata()?.len() + 8;
            let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n");
            stream.write_all(head.as_bytes())?;
            stream.write_all(&seq.to_le_bytes())?;
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
            let delta = scaled(&delta, scale);
            c.net.apply(&delta);
            c.seq += 1;
            let seq = c.seq;
            let bytes = encode(&delta);
            c.log_bytes += bytes.len();
            c.log.push_back(Delta { seq, from: me, bytes: Arc::new(bytes), at: Instant::now() });
            drop(c);
            respond(&mut stream, 200, format!("{seq} {scale}").as_bytes());
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
    let coord = Arc::new(Mutex::new(Coord {
        net,
        path: net_path,
        seq,
        saved_seq: seq,
        saved_at: Instant::now(),
        log: VecDeque::new(),
        log_bytes: 0,
        job,
        workers: HashMap::new(),
        episodes,
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

/// Downloads the master net: 8-byte seq, then the weights file.
fn download_net(c: &Client) -> Result<(NTuple, u64), String> {
    eprintln!("downloading the master net (large, one time)...");
    let (code, path) = c.call("GET", "/net", None)?;
    if code != 200 {
        return Err(format!("GET /net: HTTP {code}"));
    }
    let mut f = BufReader::new(std::fs::File::open(&path).map_err(|e| e.to_string())?);
    let mut seq = [0u8; 8];
    f.read_exact(&mut seq).map_err(|e| e.to_string())?;
    let net = NTuple::load_from(f).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(path);
    Ok((net, u64::from_le_bytes(seq)))
}

/// Applies every other worker's deltas since `since`; None means too far behind (re-download).
fn pull(c: &Client, net: &NTuple, mirror: &mut [f32], since: u64, me: &str) -> Result<Option<u64>, String> {
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
        add_into(mirror, &delta);
        pos += len;
    }
    Ok(Some(seq))
}

fn add_into(v: &mut [f32], delta: &[(u32, f32)]) {
    for &(i, d) in delta {
        if let Some(x) = v.get_mut(i as usize) {
            *x += d;
        }
    }
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
const BUILD: u32 = 3;
pub const CHILD_ENV: &str = "G2048_WORKER_CHILD";
/// The exit code a worker uses to ask its supervisor for the new build.
const UPDATE_EXIT: i32 = 42;
/// The exit code for `stop.NAME=1` in the job: the supervisor exits too.
const STOP_EXIT: i32 = 43;

/// Runs the worker as a child process and restarts it when it exits: after a crash, or
/// with the new build when the job asks for one. On Windows it first swaps in the exe the
/// server publishes (a running exe can be renamed, not overwritten); elsewhere the binary
/// on disk is already the new one.
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
                    let old = exe.with_extension("old.exe");
                    let _ = std::fs::remove_file(&old);
                    if !got || std::fs::rename(&exe, &old).is_err() || std::fs::rename(&new, &exe).is_err() {
                        eprintln!("update failed, restarting the current version in 60s");
                        std::thread::sleep(Duration::from_secs(60));
                    }
                }
            }
            _ => {
                eprintln!("worker stopped ({code:?}), restarting in 10s");
                std::thread::sleep(Duration::from_secs(10));
            }
        }
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

    // The local net, the master as this worker last knew it (`mirror`), and the master's seq.
    // net - mirror is training not yet sent. While one chunk trains, a background thread
    // sends the previous chunk's changes and pulls everyone else's, so no core sits idle.
    let mut net: Option<Arc<NTuple>> = None;
    let mut mirror: Vec<f32> = Vec::new();
    let mut seq = 0;
    if let Some(n) = NTuple::load(cached.to_str().unwrap()).ok() {
        if let Some(s) = std::fs::read_to_string(&cached_seq).ok().and_then(|s| s.trim().parse().ok()) {
            eprintln!("resuming from the cached net at seq {s}");
            mirror = n.snapshot();
            (net, seq) = (Some(Arc::new(n)), s);
        }
    }
    let mut syncing: Option<std::thread::JoinHandle<(Vec<f32>, Option<u64>)>> = None;
    let mut pool: Option<RestartPool> = None;
    let mut last_cache_save = Instant::now();
    let mut seed = nanos;
    #[cfg(windows)]
    let mut priority = String::new();
    let backoff = || {
        set("retrying");
        std::thread::sleep(Duration::from_secs(30));
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
            std::process::exit(UPDATE_EXIT);
        }
        if job_value(&job, &format!("stop.{name}")).unwrap_or(0.0) > 0.0 {
            set("stopped");
            eprintln!("stopped from the server");
            if let Some(h) = syncing.take() {
                let _ = h.join();
            }
            std::process::exit(STOP_EXIT);
        }
        if job_value(&job, "pause").unwrap_or(0.0) > 0.0 {
            set("paused");
            std::thread::sleep(Duration::from_secs(60));
            continue;
        }
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
            mirror = n.snapshot();
            set("syncing");
            match pull(&c, &n, &mut mirror, seq, &me) {
                Ok(Some(s)) => seq = s,
                Ok(None) => continue,
                Err(e) => {
                    eprintln!("{e}; retrying in 30s");
                    backoff();
                    continue;
                }
            }
            net = Some(Arc::new(n));
        }
        let n = net.clone().unwrap();
        let pool = pool.get_or_insert_with(|| RestartPool::new(n.stages(), 100_000));

        // Train one chunk.
        let alpha = job_value(&job, "alpha").unwrap_or(0.00015625) as f32;
        let restart = job_value(&job, "restart").unwrap_or(0.5) as f32;
        let secs = job_value(&job, "secs").unwrap_or(120.0);
        let send_mb = job_value(&job, "send_mb").unwrap_or(16.0);
        // `priority.NAME=high` etc. sets a Windows machine's priority class. Below normal (the
        // default) still uses every core when idle but lets the desktop go first.
        #[cfg(windows)]
        {
            let want = job_text(&job, &format!("priority.{name}")).unwrap_or("BelowNormal").to_string();
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
        let counters: Vec<AtomicU64> = (0..8).map(|_| AtomicU64::new(0)).collect();
        let start = Instant::now();
        set("training");
        seed = seed.wrapping_add(0x9E37_79B9);
        ntuple::train_parallel(&n, pool, alpha, restart, seed, threads, u64::MAX, Some(start + Duration::from_secs_f64(secs)), &|_, e| {
            counters[0].fetch_add(1, Relaxed);
            if e.fresh {
                counters[1].fetch_add(1, Relaxed);
                counters[2].fetch_add(e.score, Relaxed);
                for k in 0..5 {
                    if e.max_rank >= 11 + k as u8 {
                        counters[3 + k].fetch_add(1, Relaxed);
                    }
                }
            }
        });
        let v: Vec<u64> = counters.iter().map(|a| a.load(Relaxed)).collect();
        let chunk = Chunk { secs: start.elapsed().as_secs_f64(), episodes: v[0], fresh: v[1], score: v[2], reached: [v[3], v[4], v[5], v[6], v[7]] };
        drop(n);

        // The previous chunk's sync must land before this chunk's changes are measured.
        if let Some(h) = syncing.take() {
            set("syncing");
            let (m, s) = h.join().expect("sync thread panicked");
            mirror = m;
            match s {
                Some(s) => seq = s,
                None => {
                    eprintln!("too far behind the master, downloading it again");
                    net = None;
                    continue;
                }
            }
        }
        let n = net.as_mut().unwrap();
        // TC fine-tuning phase: its per-weight accumulators stay local to each worker.
        if job_value(&job, "tc").unwrap_or(0.0) > 0.0 && !n.tc_enabled() {
            eprintln!("switching to TC learning");
            Arc::get_mut(n).expect("net still shared").enable_tc();
        }
        if last_cache_save.elapsed() > Duration::from_secs(1800) {
            if n.save(cached.to_str().unwrap()).is_ok() {
                let _ = std::fs::write(&cached_seq, seq.to_string());
            }
            last_cache_save = Instant::now();
        }
        set("preparing update");
        let unsent = n.diff(&mirror);
        let pending = unsent.len();
        let sent = pick_largest(unsent, send_mb * 1e6);
        let bytes = encode(&sent);
        let file = cache.join("delta.bin");
        std::fs::write(&file, &bytes).expect("writing delta");

        let (c, n, me, name, mut mirror_owned) = (c.clone(), n.clone(), me.clone(), name.clone(), std::mem::take(&mut mirror));
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
            // The master took `scale` of what was sent; keep the same here so this copy matches it.
            let scale: f32 = reply.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let taken = scaled(&sent, scale);
            let mut back = Vec::with_capacity(sent.len());
            let mut t = taken.iter().peekable();
            for &(i, d) in &sent {
                let a = if t.peek().is_some_and(|x| x.0 == i) { t.next().unwrap().1 } else { 0.0 };
                back.push((i, a - d));
            }
            n.apply(&back);
            add_into(&mut mirror_owned, &taken);
            eprintln!(
                "{} games in {:.0}s, mean score {:.0}; sent {:.1} MB ({} of {} changed weights), share {scale:.2}",
                chunk.episodes,
                chunk.secs,
                if chunk.fresh > 0 { chunk.score as f64 / chunk.fresh as f64 } else { 0.0 },
                bytes.len() as f64 / 1e6,
                sent.len(),
                pending
            );
            let s = loop {
                match pull(&c, &n, &mut mirror_owned, seq, &me) {
                    Ok(s) => break s,
                    Err(e) => {
                        eprintln!("{e}; retrying in 30s");
                        std::thread::sleep(Duration::from_secs(30));
                    }
                }
            };
            (mirror_owned, s)
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
    fn pick_largest_keeps_the_biggest_within_budget() {
        let raw = vec![(1, 0.001), (2, -5.0), (3, 0.5), (4, 3.0)];
        let got = pick_largest(raw, 2.0 * BYTES_PER_ENTRY);
        assert_eq!(got, vec![(2, -5.0), (4, 3.0)]);
    }
}
