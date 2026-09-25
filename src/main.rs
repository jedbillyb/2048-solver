mod ai;
mod board;
mod ntuple;

use board::*;
use ntuple::NTuple;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage:
  g2048 bench [games=16] [seed=1] [--net FILE --depth N]
  g2048 train OUT_FILE [games=1000000] [--resume FILE] [--alpha A] [--seed S]";

struct GameResult {
    score: u64,
    max_rank: u8,
    moves: u64,
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
    let (mut score, mut moves) = (0u64, 0u64);
    while let Some(d) = ai.best_move(b) {
        let (nb, s) = ai.tables().apply(b, d);
        score += s as u64;
        moves += 1;
        b = spawn(nb, &mut rng);
    }
    GameResult { score, max_rank: max_rank(b), moves }
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
}

fn bench(args: &[String]) {
    let pos = positional(args);
    let games: u64 = pos.first().and_then(|s| s.parse().ok()).unwrap_or(16);
    let seed0: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ai = Arc::new(match flag::<String>(args, "--net") {
        Some(path) => {
            let net = NTuple::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}"));
            ai::Ai::with_net(Arc::new(net), flag(args, "--depth").unwrap_or(2))
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
                    eprintln!("game {:>3}: score {:>7}  max tile {:>5}  moves {}", g, r.score, 1u32 << r.max_rank, r.moves);
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
    // Per-weight step: 0.1 spread over the 32 weights each board touches.
    let alpha: f32 = flag(args, "--alpha").unwrap_or(0.1 / 32.0);
    let seed: u64 = flag(args, "--seed").unwrap_or(42);
    let net = Arc::new(match flag::<String>(args, "--resume") {
        Some(p) => NTuple::load(&p).unwrap_or_else(|e| panic!("loading {p}: {e}")),
        None => NTuple::new(0.0),
    });
    let tables = Arc::new(Tables::new());
    let nt = threads(games);
    let next = Arc::new(AtomicU64::new(0));
    const WINDOW: u64 = 10_000;
    // Per-window tallies: [score sum, games, >=2048, >=4096, >=8192, >=16384]
    let stats: Arc<Vec<AtomicU64>> = Arc::new((0..6).map(|_| AtomicU64::new(0)).collect());
    let start = Instant::now();
    let handles: Vec<_> = (0..nt as u64)
        .map(|tid| {
            let (net, tables, next, stats, out) = (net.clone(), tables.clone(), next.clone(), stats.clone(), out.clone());
            std::thread::spawn(move || {
                let mut rng = Rng((seed ^ (tid + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let e = ntuple::train_episode(&net, &tables, &mut rng, alpha);
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
                            "{:>9} games {:>6.0}s  mean {:>7.0}  2048 {:>5.1}%  4096 {:>5.1}%  8192 {:>5.1}%  16384 {:>4.1}%",
                            g + 1, start.elapsed().as_secs_f64(), v[0] as f64 / v[1] as f64, pct(v[2]), pct(v[3]), pct(v[4]), pct(v[5])
                        );
                        if (g + 1) % (WINDOW * 10) < WINDOW {
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]),
        Some("train") => train(&args[1..]),
        _ => eprintln!("{USAGE}"),
    }
}
