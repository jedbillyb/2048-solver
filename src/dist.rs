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

const DEFAULT_JOB: &str = "alpha=0.00015625\nrestart=0.5\nsecs=120\nsend_mb=16\npause=0\n";
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

    fn line(&self, name: &str, seen: String, rate: f64) -> String {
        let pct = |x: u64| if self.fresh == 0 { 0.0 } else { 100.0 * x as f64 / self.fresh as f64 };
        let r = self.reached;
        format!(
            "{name:<22}{seen:>6}{:>9.0}{:>9}{:>9.0}{:>7.1}{:>7.1}{:>7.1}{:>7.1}{:>7.2}\n",
            rate,
            self.fresh,
            if self.fresh == 0 { 0.0 } else { self.score as f64 / self.fresh as f64 },
            pct(r[0]), pct(r[1]), pct(r[2]), pct(r[3]), pct(r[4])
        )
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

    fn status(&self) -> String {
        let mut s = format!(
            "master seq {}  saved seq {} ({}s ago)  holding {} deltas ({} MB)\njob: {}\n\n",
            self.seq,
            self.saved_seq,
            self.saved_at.elapsed().as_secs(),
            self.log.len(),
            self.log_bytes >> 20,
            self.job.trim().replace('\n', "  ")
        );
        s += &format!("{:<22}{:>6}{:>9}{:>9}{:>9}{:>7}{:>7}{:>7}{:>7}{:>7}\n", "worker", "seen", "games/s", "fresh", "mean", "2048", "4096", "8192", "16384", "32768");
        let mut names: Vec<_> = self.workers.keys().collect();
        names.sort();
        // Rates add across machines; score and reach percentages pool their fresh games.
        let (mut total, mut rate) = (Chunk::default(), 0.0);
        for name in names {
            let w = &self.workers[name];
            let mut sum = Chunk::default();
            w.recent.iter().for_each(|c| sum.add(c));
            let seen = w.last_seen.elapsed().as_secs();
            s += &sum.line(name, format!("{seen}s"), sum.rate());
            if seen < 600 {
                total.add(&sum);
                rate += sum.rate();
            }
        }
        s += &total.line("ALL (live)", String::new(), rate);
        let episodes: u64 = self.workers.values().map(|w| w.total_episodes).sum();
        s += &format!("\n{episodes} training games since the coordinator started\n");
        s
    }
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
            let s = coord.lock().unwrap().status();
            respond(&mut stream, 200, s.as_bytes());
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
            c.net.apply(&delta);
            c.seq += 1;
            let seq = c.seq;
            c.log_bytes += body.len();
            c.log.push_back(Delta { seq, from: me, bytes: Arc::new(body), at: Instant::now() });
            let w = c.workers.entry(name).or_insert(WorkerInfo { last_seen: Instant::now(), recent: VecDeque::new(), total_episodes: 0 });
            w.last_seen = Instant::now();
            w.total_episodes += chunk.episodes;
            w.recent.push_back(chunk);
            if w.recent.len() > RECENT {
                w.recent.pop_front();
            }
            drop(c);
            respond(&mut stream, 200, seq.to_string().as_bytes());
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

struct Client {
    url: String,
    auth: String,
    tmp: std::path::PathBuf,
}

impl Client {
    /// Runs curl, returning the HTTP status and the path the response body landed in.
    fn call(&self, method: &str, path: &str, body: Option<&std::path::Path>) -> Result<(u16, std::path::PathBuf), String> {
        let out = self.tmp.join(format!("resp-{}", path.split('?').next().unwrap_or("x").trim_matches('/')));
        let url = format!("{}{path}", self.url);
        let mut cmd = std::process::Command::new("curl");
        cmd.args(["-sS", "--connect-timeout", "20", "-X", method, "-H", &self.auth, "-o"]).arg(&out).args(["-w", "%{http_code}"]);
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

/// `g2048 worker --url URL --token T [--name N] [--threads N] [--cache DIR]`: trains forever.
pub fn worker(url: String, token: String, name: Option<String>, threads: usize, cache: std::path::PathBuf) {
    std::fs::create_dir_all(&cache).expect("creating cache dir");
    let name = name.unwrap_or_else(machine_name);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    // Unique per run, so a restarted worker never skips deltas it didn't make itself.
    let me = format!("{name}-{:08x}", (nanos ^ std::process::id() as u64) as u32);
    let c = Client { url: url.trim_end_matches('/').to_string(), auth: format!("Authorization: Bearer {token}"), tmp: cache.clone() };
    let cached = cache.join("net.bin");
    let cached_seq = cache.join("net.seq");
    eprintln!("worker {me} using {threads} threads");

    // The local net, the master as this worker last knew it (`mirror`), and the master's seq.
    // net - mirror is training not yet sent; each chunk sends the biggest part of it.
    let mut state: Option<(NTuple, Vec<f32>, u64)> = NTuple::load(cached.to_str().unwrap()).ok().and_then(|n| {
        let seq = std::fs::read_to_string(&cached_seq).ok()?.trim().parse().ok()?;
        eprintln!("resuming from the cached net at seq {seq}");
        let mirror = n.snapshot();
        Some((n, mirror, seq))
    });
    let mut pool: Option<RestartPool> = None;
    let mut last_cache_save = Instant::now();
    let mut seed = nanos;
    let backoff = || std::thread::sleep(Duration::from_secs(30));

    loop {
        let job = match c.text("GET", "/job", None) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("{e}; retrying in 30s");
                backoff();
                continue;
            }
        };
        if job_value(&job, "pause").unwrap_or(0.0) > 0.0 {
            std::thread::sleep(Duration::from_secs(60));
            continue;
        }
        // Get in sync with the master: cached or downloaded net, then everyone's newer deltas.
        let (net, mut mirror, seq) = match state.take() {
            Some(s) => s,
            None => match download_net(&c) {
                Ok((n, seq)) => {
                    let mirror = n.snapshot();
                    (n, mirror, seq)
                }
                Err(e) => {
                    eprintln!("{e}; retrying in 30s");
                    backoff();
                    continue;
                }
            },
        };
        let seq = match pull(&c, &net, &mut mirror, seq, &me) {
            Ok(Some(s)) => s,
            Ok(None) => {
                eprintln!("too far behind the master, downloading it again");
                continue;
            }
            Err(e) => {
                eprintln!("{e}; retrying in 30s");
                state = Some((net, mirror, seq));
                backoff();
                continue;
            }
        };
        if last_cache_save.elapsed() > Duration::from_secs(1800) {
            if net.save(cached.to_str().unwrap()).is_ok() {
                let _ = std::fs::write(&cached_seq, seq.to_string());
            }
            last_cache_save = Instant::now();
        }
        let pool = pool.get_or_insert_with(|| RestartPool::new(net.stages(), 100_000));

        // Train one chunk.
        let alpha = job_value(&job, "alpha").unwrap_or(0.00015625) as f32;
        let restart = job_value(&job, "restart").unwrap_or(0.5) as f32;
        let secs = job_value(&job, "secs").unwrap_or(120.0);
        let send_mb = job_value(&job, "send_mb").unwrap_or(16.0);
        let counters: Vec<AtomicU64> = (0..8).map(|_| AtomicU64::new(0)).collect();
        let start = Instant::now();
        seed = seed.wrapping_add(0x9E37_79B9);
        ntuple::train_parallel(&net, pool, alpha, restart, seed, threads, u64::MAX, Some(start + Duration::from_secs_f64(secs)), &|_, e| {
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
        let unsent = net.diff(&mirror);
        let pending = unsent.len();
        let sent = pick_largest(unsent, send_mb * 1e6);
        let bytes = encode(&sent);
        let file = cache.join("delta.bin");
        std::fs::write(&file, &bytes).expect("writing delta");
        // Keep retrying: dropping a delta would leave this copy ahead of the master for good.
        loop {
            match c.text("POST", &format!("/delta?me={me}&name={name}&{}", chunk.to_query()), Some(&file)) {
                Ok(_) => break,
                Err(e) => {
                    eprintln!("{e}; retrying in 30s");
                    backoff();
                }
            }
        }
        add_into(&mut mirror, &sent);
        eprintln!(
            "{} games in {:.0}s, mean score {:.0}; sent {:.1} MB ({} of {} changed weights)",
            chunk.episodes,
            chunk.secs,
            if chunk.fresh > 0 { chunk.score as f64 / chunk.fresh as f64 } else { 0.0 },
            bytes.len() as f64 / 1e6,
            sent.len(),
            pending
        );
        state = Some((net, mirror, seq));
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
