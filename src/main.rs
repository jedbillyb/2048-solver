mod ai;
mod board;
mod ntuple;

use board::*;
use ntuple::NTuple;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage:
  g2048 bench [games=16] [seed=1] [--net FILE --depth N [--endgame-depth N]]
  g2048 positions OUT_FILE [count=64] --net FILE [--depth 2]   (boards where 16384 first appears)
  g2048 endgame POS_FILE --net FILE [--depth N] [--cprob P]     (play saved boards to the end)
  g2048 serve --net FILE [--depth N] [--port 20480]
  g2048 train OUT_FILE [games=1000000] [--resume FILE] [--alpha A] [--seed S] [--tc 1] [--stages 3] [--restart 0.5] [--tuples 4|8]";

struct GameResult {
    score: u64,
    max_rank: u8,
    moves: u64,
    /// Two 32768s merged. The game stops there: that's the goal.
    won_65536: bool,
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
}

fn positional(args: &[String]) -> Vec<&String> {
    let mut out = vec![];
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += 2;
        } else {
            out.push(&args[i]);
            i += 1;
        }
    }
    out
}

fn threads(cap: u64) -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).min(cap.max(1) as usize)
}

fn play(ai: &ai::Ai, seed: u64) -> GameResult {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut b = spawn(spawn(0, &mut rng), &mut rng);
    let (mut score, mut moves, mut won_65536) = (0u64, 0u64, false);
    while let Some(d) = ai.best_move(b) {
        let (nb, s) = ai.tables().apply(b, d);
        score += s as u64;
        moves += 1;
        if made_65536(b, nb) {
            won_65536 = true;
            b = nb;
            break;
        }
        b = spawn(nb, &mut rng);
    }
    if std::env::var_os("G2048_SHOW_END").is_some() {
        eprintln!("final board (score {score}):\n{}", board::print(b));
    }
    GameResult { score, max_rank: max_rank(b), moves, won_65536 }
}

fn report(results: &[GameResult], secs: f64, threads: usize) {
    let n = results.len() as f64;
    let total_moves: u64 = results.iter().map(|r| r.moves).sum();
    let mut scores: Vec<u64> = results.iter().map(|r| r.score).collect();
    scores.sort();
    println!("\n{} games in {:.1}s ({:.0} moves/s across {} threads)", results.len(), secs, total_moves as f64 / secs, threads);
    println!("score  mean {:.0}  median {}  max {}", scores.iter().sum::<u64>() as f64 / n, scores[scores.len() / 2], scores.last().unwrap());
    for k in [11u8, 12, 13, 14, 15] {
        let hit = results.iter().filter(|r| r.max_rank >= k).count();
        println!("reached {:>5}: {:>5.1}%", 1u32 << k, 100.0 * hit as f64 / n);
    }
    let won = results.iter().filter(|r| r.won_65536).count();
    println!("reached 65536: {:>5.1}%", 100.0 * won as f64 / n);
}

/// Net-backed AI from --net / --depth / --endgame-depth / --cprob.
fn net_ai(args: &[String], default_depth: u32) -> ai::Ai {
    let path: String = flag(args, "--net").unwrap_or_else(|| panic!("{USAGE}"));
    let net = NTuple::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}"));
    ai::Ai::with_net(Arc::new(net), flag(args, "--depth").unwrap_or(default_depth))
        .with_endgame_depth(flag(args, "--endgame-depth"))
        .with_cprob(flag(args, "--cprob"))
}

/// Runs `job(i)` for i in 0..n across all cores, returning results in index order.
fn parallel<T: Send + 'static>(n: u64, job: impl Fn(u64) -> T + Send + Sync + 'static) -> Vec<T> {
    let job = Arc::new(job);
    let next = Arc::new(AtomicU64::new(0));
    let handles: Vec<_> = (0..threads(n))
        .map(|_| {
            let (job, next) = (job.clone(), next.clone());
            std::thread::spawn(move || {
                let mut out = vec![];
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break out;
                    }
                    out.push((i, job(i)));
                }
            })
        })
        .collect();
    let mut all: Vec<(u64, T)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, t)| t).collect()
}

/// Plays fresh games and keeps the board right after 16384 first appears.
fn positions(args: &[String]) {
    let pos = positional(args);
    let out = pos.first().unwrap_or_else(|| panic!("{USAGE}")).to_string();
    let count: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let ai = Arc::new(net_ai(args, 2));
    let found = parallel(count * 2, move |i| {
        let mut rng = Rng((5000 + i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut b = spawn(spawn(0, &mut rng), &mut rng);
        while let Some(d) = ai.best_move(b) {
            b = spawn(ai.tables().apply(b, d).0, &mut rng);
            if max_rank(b) >= 14 {
                return Some(b);
            }
        }
        None
    });
    let boards: Vec<Board> = found.into_iter().flatten().take(count as usize).collect();
    let text: String = boards.iter().map(|b| format!("{b:016x}\n")).collect();
    std::fs::write(&out, text).expect("writing positions");
    println!("saved {} boards to {out}", boards.len());
}

/// Plays each saved board to the end and reports how often 32768 / 65536 follow.
fn endgame(args: &[String]) {
    let pos = positional(args);
    let file = pos.first().unwrap_or_else(|| panic!("{USAGE}"));
    let boards: Vec<Board> = std::fs::read_to_string(file)
        .expect("reading positions")
        .lines()
        .filter_map(|l| u64::from_str_radix(l.trim(), 16).ok())
        .collect();
    let ai = Arc::new(net_ai(args, 2));
    let start = Instant::now();
    let n = boards.len() as u64;
    let boards = Arc::new(boards);
    let results = parallel(n, move |i| {
        let mut rng = Rng((9000 + i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut b = boards[i as usize];
        let (mut moves, mut won) = (0u64, false);
        while let Some(d) = ai.best_move(b) {
            let nb = ai.tables().apply(b, d).0;
            moves += 1;
            if made_65536(b, nb) {
                won = true;
                break;
            }
            b = spawn(nb, &mut rng);
        }
        (max_rank(b), won, moves)
    });
    let pct = |f: &dyn Fn(&(u8, bool, u64)) -> bool| 100.0 * results.iter().filter(|r| f(r)).count() as f64 / n as f64;
    let moves: u64 = results.iter().map(|r| r.2).sum();
    println!(
        "{} boards  32768 {:>5.1}%  65536 {:>5.1}%  avg moves {:.0}  {:.0}s ({:.0} moves/s)",
        n,
        pct(&|r| r.0 >= 15),
        pct(&|r| r.1),
        moves as f64 / n as f64,
        start.elapsed().as_secs_f64(),
        moves as f64 / start.elapsed().as_secs_f64()
    );
}

fn bench(args: &[String]) {
    let pos = positional(args);
    let games: u64 = pos.first().and_then(|s| s.parse().ok()).unwrap_or(16);
    let seed0: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ai = Arc::new(match flag::<String>(args, "--net") {
        Some(path) => {
            let net = NTuple::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}"));
            ai::Ai::with_net(Arc::new(net), flag(args, "--depth").unwrap_or(2)).with_endgame_depth(flag(args, "--endgame-depth"))
        }
        None => ai::Ai::new(),
    });
    let nt = threads(games);
    let next = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let handles: Vec<_> = (0..nt)
        .map(|_| {
            let (ai, next) = (ai.clone(), next.clone());
            std::thread::spawn(move || {
                let mut out = vec![];
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break out;
                    }
                    let r = play(&ai, seed0 + g);
                    let tile = if r.won_65536 { 65536 } else { 1u32 << r.max_rank };
                    eprintln!("game {:>3}: score {:>7}  max tile {:>5}  moves {}", g, r.score, tile, r.moves);
                    out.push(r);
                }
            })
        })
        .collect();
    let results: Vec<GameResult> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    report(&results, start.elapsed().as_secs_f64(), nt);
}

fn train(args: &[String]) {
    let pos = positional(args);
    let out = pos.first().unwrap_or_else(|| panic!("{USAGE}")).to_string();
    let games: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let alpha_flag: Option<f32> = flag(args, "--alpha");
    let seed: u64 = flag(args, "--seed").unwrap_or(42);
    let mut net = match flag::<String>(args, "--resume") {
        Some(p) => NTuple::load(&p).unwrap_or_else(|e| panic!("loading {p}: {e}")),
        None => NTuple::new(0.0, 1, if flag::<u8>(args, "--tuples") == Some(8) { &ntuple::TUPLES_8 } else { &ntuple::TUPLES_4 }),
    };
    net.expand_stages(flag(args, "--stages").unwrap_or(1));
    let restart_p: f32 = flag(args, "--restart").unwrap_or(0.0);
    let pool = Arc::new(ntuple::RestartPool::new(net.stages(), 100_000));
    if flag::<u8>(args, "--tc") == Some(1) {
        net.enable_tc();
    }
    // Per-weight step: 0.1 spread over the weights each board touches (8 per tuple).
    let alpha = alpha_flag.unwrap_or(0.1 / (8 * net.tuple_count()) as f32);
    let net = Arc::new(net);
    let tables = Arc::new(Tables::new());
    let nt = threads(games);
    let next = Arc::new(AtomicU64::new(0));
    const WINDOW: u64 = 10_000;
    static WINDOWS_DONE: AtomicU64 = AtomicU64::new(0);
    // Per-window tallies: [score sum, games, >=2048, >=4096, >=8192, >=16384]
    let stats: Arc<Vec<AtomicU64>> = Arc::new((0..6).map(|_| AtomicU64::new(0)).collect());
    let start = Instant::now();
    let handles: Vec<_> = (0..nt as u64)
        .map(|tid| {
            let (net, tables, next, stats, out, pool) =
                (net.clone(), tables.clone(), next.clone(), stats.clone(), out.clone(), pool.clone());
            std::thread::spawn(move || {
                let mut rng = Rng((seed ^ (tid + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let e = ntuple::train_episode(&net, &tables, &mut rng, alpha, &pool, restart_p);
                    // Only fresh games say how strong the net is; restarts begin mid-game.
                    if !e.fresh {
                        continue;
                    }
                    stats[0].fetch_add(e.score, Ordering::Relaxed);
                    for (i, k) in [11u8, 12, 13, 14].iter().enumerate() {
                        if e.max_rank >= *k {
                            stats[2 + i].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if stats[1].fetch_add(1, Ordering::Relaxed) + 1 == WINDOW {
                        let v: Vec<u64> = stats.iter().map(|s| s.swap(0, Ordering::Relaxed)).collect();
                        let pct = |x: u64| 100.0 * x as f64 / v[1] as f64;
                        println!(
                            "{:>9} games {:>6.0}s  mean {:>7.0}  2048 {:>5.1}%  4096 {:>5.1}%  8192 {:>5.1}%  16384 {:>4.1}%  pool {:?}",
                            g + 1, start.elapsed().as_secs_f64(), v[0] as f64 / v[1] as f64, pct(v[2]), pct(v[3]), pct(v[4]), pct(v[5]), pool.sizes()
                        );
                        if WINDOWS_DONE.fetch_add(1, Ordering::Relaxed) % 10 == 9 {
                            net.save(&out).expect("saving weights");
                        }
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    net.save(&out).expect("saving weights");
    println!("saved {out}");
}

/// Board from 16 hex digits, one tile rank per cell, row-major from the top-left.
fn parse_board(hex: &str) -> Option<Board> {
    if hex.len() != 16 {
        return None;
    }
    hex.chars().enumerate().try_fold(0u64, |b, (i, ch)| Some(b | (ch.to_digit(16)? as u64) << (4 * i)))
}

/// Tiny HTTP server for the browser bot: GET /move?b=<16 hex ranks> -> "up" | "down" | "left" | "right" | "none".
fn serve(args: &[String]) {
    use std::io::{BufRead, BufReader, Write};
    let path: String = flag(args, "--net").unwrap_or_else(|| panic!("{USAGE}"));
    let net = NTuple::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}"));
    let ai = ai::Ai::with_net(Arc::new(net), flag(args, "--depth").unwrap_or(3)).with_endgame_depth(flag(args, "--endgame-depth"));
    let port: u16 = flag(args, "--port").unwrap_or(20480);
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).expect("binding port");
    println!("serving on http://127.0.0.1:{port}");
    for stream in listener.incoming().flatten() {
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            continue;
        }
        // Drain headers so the browser sees a clean response.
        let mut h = String::new();
        while reader.read_line(&mut h).map_or(false, |n| n > 2) {
            h.clear();
        }
        let target = line.split_whitespace().nth(1).unwrap_or("");
        let body = match line.split_whitespace().next() {
            Some("OPTIONS") => String::new(),
            _ => match target.strip_prefix("/move?b=").and_then(parse_board) {
                Some(b) => match ai.best_move(b) {
                    Some(Dir::Up) => "up",
                    Some(Dir::Down) => "down",
                    Some(Dir::Left) => "left",
                    Some(Dir::Right) => "right",
                    None => "none",
                }
                .to_string(),
                None => "bad request".to_string(),
            },
        };
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Private-Network: true\r\nAccess-Control-Allow-Methods: GET, OPTIONS\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = (&stream).write_all(resp.as_bytes());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]),
        Some("train") => train(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("positions") => positions(&args[1..]),
        Some("endgame") => endgame(&args[1..]),
        _ => eprintln!("{USAGE}"),
    }
}
