mod ai;
mod board;
mod dist;
mod endgame;
mod ntuple;

use board::*;
use ntuple::NTuple;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage:
  g2048 bench [games=16] [seed=1] [--net FILE] [--depth N] [--endgame-depth N] [--top-bias C] [--eval rows|mx] [LOOKUP]
  g2048 positions OUT_FILE [count=64] --net FILE [--depth 2] [--rank 14] [--from-file SEEDS]   (boards where the tile first appears; --from-file starts games from seed boards)
  g2048 endgame POS_FILE --net FILE [--depth N] [--cprob P] [LOOKUP]       (play saved boards to the end)
  g2048 formation POS_FILE [--layouts L]                        (which endgame layouts the boards fit)
    LOOKUP: --tables DIR [--layouts block10,five7] [--lookup-rank 15] [--lookup-min P]
            exact endgame tables (macroxue style), filled on first use and saved in DIR;
            five8 needs about 9 GB of RAM, five9 (macroxue's 2022 table) about 33 GB
            --endgame-eval mx [--endgame-rank 15] [--endgame-depth 3]: from a board holding
            that rank, play like macroxue (its evaluation and depth) instead of the net
  g2048 serve --net FILE [--depth N] [--port 20480]
  g2048 coord --net MASTER --token-file F [--port 20490]      (hands out training work)
  g2048 worker --url URL --token-file F [--name N] [--threads N] [--cache DIR]
  g2048 boundary GAMES --net FILE --vs FILE                    (both nets' view of the 32768 merge)
  g2048 stages FILE [N]                                        (show, or grow to N stages, a saved net)
  g2048 train OUT_FILE [games=1000000] [--resume FILE] [--alpha A] [--seed S] [--tc 1] [--stages 3] [--restart 0.5] [--restart-stage 1] [--restart-file FILE] [--freeze N] [--tuples 4|8] [--init 320000]";

struct GameResult {
    score: u64,
    max_rank: u8,
    moves: u64,
    /// A 65536 was made at some point in the game.
    won_65536: bool,
    /// A 131072 was made: the win, and the game stops there.
    won_131072: bool,
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
    let (mut score, mut moves, mut won_65536, mut won_131072) = (0u64, 0u64, false, false);
    while let Some(d) = ai.best_move(b) {
        let (nb, s) = ai.tables().apply(b, d);
        score += s as u64;
        moves += 1;
        won_65536 |= made_65536(b, nb);
        if made_131072(b, nb) {
            // 131072 is the largest tile a 4x4 board can hold: the win, so stop here.
            won_131072 = true;
            b = nb;
            break;
        }
        b = spawn(nb, &mut rng);
    }
    if std::env::var_os("G2048_SHOW_END").is_some() {
        eprintln!("final board (score {score}):\n{}", board::print(b));
    }
    GameResult { score, max_rank: max_rank(b), moves, won_65536, won_131072 }
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
    let won = results.iter().filter(|r| r.won_65536 || r.max_rank >= 16).count();
    println!("reached 65536: {:>5.1}%", 100.0 * won as f64 / n);
    let won_big = results.iter().filter(|r| r.won_131072 || r.max_rank >= 17).count();
    println!("reached 131072: {:>4.2}%", 100.0 * won_big as f64 / n);
}

/// Endgame tables from --tables DIR and --layouts, or None without --tables.
fn lookup(args: &[String]) -> Option<Arc<endgame::Lookup>> {
    let dir: String = flag(args, "--tables")?;
    let names: String = flag(args, "--layouts").unwrap_or_else(|| "block10,five7".into());
    let layouts: Vec<_> = names.split(',').map(|n| endgame::Layout::parse(n.trim()).unwrap_or_else(|| panic!("unknown layout {n}; use e.g. block10 or snake8"))).collect();
    Some(Arc::new(endgame::Lookup::open(std::path::Path::new(&dir), &layouts).unwrap_or_else(|e| panic!("opening endgame tables in {dir}: {e}"))))
}

/// Net-backed AI from --net / --depth / --endgame-depth / --cprob / --top-bias and the
/// endgame tables, or the hand-tuned heuristic (adaptive depth unless --depth) when --net
/// is absent.
fn net_ai(args: &[String], default_depth: u32) -> ai::Ai {
    let ai = match flag::<String>(args, "--net") {
        None => {
            let h = match flag::<String>(args, "--eval").as_deref() {
                None | Some("rows") => ai::Heuristic::Rows,
                Some("mx") => ai::Heuristic::Macroxue,
                Some(other) => panic!("unknown --eval {other}; use rows or mx"),
            };
            ai::Ai::new().with_heuristic(h).with_depth(flag(args, "--depth"))
        }
        Some(path) => {
            let net = NTuple::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}"));
            ai::Ai::with_net(Arc::new(net), flag(args, "--depth").unwrap_or(default_depth)).with_endgame_depth(flag(args, "--endgame-depth")).with_top_bias(flag(args, "--top-bias"))
        }
    };
    // --endgame-eval mx [--endgame-rank R]: from a board holding rank R (default 15),
    // play like macroxue (its evaluation, depth 3 unless --endgame-depth, its tables).
    let endgame = match flag::<String>(args, "--endgame-eval").as_deref() {
        None => None,
        Some("mx") => Some(ai::Heuristic::Macroxue),
        Some(other) => panic!("unknown --endgame-eval {other}; use mx"),
    };
    let endgame_rank: Option<u8> = flag(args, "--endgame-rank");
    let lookup_rank = flag(args, "--lookup-rank").or(if endgame.is_some() { Some(endgame_rank.unwrap_or(15)) } else { None });
    ai.with_cprob(flag(args, "--cprob")).with_pass_score(flag(args, "--pass-score")).with_endgame_eval(endgame, endgame_rank).with_lookup(lookup(args), lookup_rank, flag(args, "--lookup-min"))
}

/// Saves the endgame tables and reports how often they answered.
fn finish_lookup(ai: &ai::Ai, moves: u64) {
    let Some(lk) = ai.lookup() else { return };
    let hits = lk.hits.load(std::sync::atomic::Ordering::Relaxed);
    let asked = lk.asked.load(std::sync::atomic::Ordering::Relaxed);
    let tables: Vec<String> = lk.tables().iter().map(|t| format!("{} {}", t.layout.name(), t.computed())).collect();
    println!("endgame tables: {} of {} moves from a table ({} boards asked); positions known: {}", hits, moves, asked, tables.join(", "));
    if let Err(e) = lk.save() {
        eprintln!("saving endgame tables: {e}");
    }
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
    let rank: u8 = flag(args, "--rank").unwrap_or(14);
    let ai = Arc::new(net_ai(args, 2));
    // --from-file: start each game from a random board in this file (e.g. pos32k)
    // instead of a fresh spawn, so a rank-16 target is reached from one tile away.
    // Self-play almost never reaches rank 16 from scratch, so this is the only
    // tractable way to harvest 65536 boards on a small box.
    let seeds = flag::<String>(args, "--from-file").map(|f| Arc::new(read_boards(&f)));
    // Rarer tiles need more games per board kept.
    let tries = count * if rank >= 15 { 4 } else { 2 };
    let found = parallel(tries, move |i| {
        let mut rng = Rng((5000 + i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut b = match &seeds {
            Some(s) if !s.is_empty() => s[(i as usize) % s.len()],
            _ => spawn(spawn(0, &mut rng), &mut rng),
        };
        while let Some(d) = ai.best_move(b) {
            b = spawn(ai.tables().apply(b, d).0, &mut rng);
            if max_rank(b) >= rank {
                return Some(b);
            }
        }
        None
    });
    let boards: Vec<Board> = found.into_iter().flatten().take(count as usize).collect();
    let text: String = boards.iter().map(|b| format!("{}\n", board_to_str(*b))).collect();
    std::fs::write(&out, text).expect("writing positions");
    println!("saved {} boards to {out}", boards.len());
}

/// A board on one line of a positions file. Boards with no tile above 32768 use the old
/// 16-hex-digit u64 form, so files interchange with the u64 build and with pos32k.txt.
/// Boards holding a 65536 or 131072 use one radix-32 char per cell (cell 0 first), the
/// only form that can carry ranks above 15.
fn board_to_str(b: Board) -> String {
    if max_rank(b) <= 15 {
        let mut n = 0u64;
        for i in 0..16 {
            n |= (((b >> (5 * i)) & 0x1F) as u64) << (4 * i);
        }
        format!("{n:016x}")
    } else {
        (0..16).map(|i| std::char::from_digit(((b >> (5 * i)) & 0x1F) as u32, 32).unwrap()).collect()
    }
}

fn parse_file_board(tok: &str) -> Option<Board> {
    if tok.len() != 16 {
        return None;
    }
    if tok.bytes().all(|c| c.is_ascii_hexdigit()) {
        let n = u64::from_str_radix(tok, 16).ok()?;
        let mut b = 0u128;
        for i in 0..16 {
            b |= (((n >> (4 * i)) & 0xF) as u128) << (5 * i);
        }
        Some(b)
    } else {
        let mut b = 0u128;
        for (i, ch) in tok.chars().enumerate() {
            b |= (ch.to_digit(32)? as u128) << (5 * i);
        }
        Some(b)
    }
}

/// Boards from a positions file: the first word of each line (see `board_to_str`).
fn read_boards(file: &str) -> Vec<Board> {
    std::fs::read_to_string(file)
        .unwrap_or_else(|e| panic!("reading {file}: {e}"))
        .lines()
        .filter_map(|l| parse_file_board(l.split_whitespace().next()?))
        .collect()
}

/// Plays each saved board to the end and reports how often 32768 / 65536 / 131072 follow.
/// Plays the whole game out (does not stop at the first 65536) so the 65536 -> 131072
/// band, the one that actually gates the 131072 goal, is measured directly.
fn endgame(args: &[String]) {
    let pos = positional(args);
    let file = pos.first().unwrap_or_else(|| panic!("{USAGE}"));
    let boards = read_boards(file);
    let ai = Arc::new(net_ai(args, 2));
    let ai2 = ai.clone();
    let start = Instant::now();
    let n = boards.len() as u64;
    let boards = Arc::new(boards);
    // status.sh reads this to show progress.
    let progress = format!("/tmp/g2048-progress-{}", std::process::id());
    let done = Arc::new(AtomicU64::new(0));
    let (progress2, done2) = (progress.clone(), done.clone());
    let results = parallel(n, move |i| {
        let ai = &ai2;
        let mut rng = Rng((9000 + i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut b = boards[i as usize];
        let (mut moves, mut w65, mut w131) = (0u64, max_rank(b) >= 16, max_rank(b) >= 17);
        while let Some(d) = ai.best_move(b) {
            let nb = ai.tables().apply(b, d).0;
            moves += 1;
            w65 |= made_65536(b, nb);
            w131 |= made_131072(b, nb);
            b = spawn(nb, &mut rng);
        }
        let k = done2.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = std::fs::write(&progress2, format!("{k}/{n} boards done"));
        (max_rank(b), w65, w131, moves)
    });
    let _ = std::fs::remove_file(&progress);
    let pct = |f: &dyn Fn(&(u8, bool, bool, u64)) -> bool| 100.0 * results.iter().filter(|r| f(r)).count() as f64 / n as f64;
    let moves: u64 = results.iter().map(|r| r.3).sum();
    finish_lookup(&ai, moves);
    println!(
        "{} boards  32768 {:>5.1}%  65536 {:>5.1}%  131072 {:>5.2}%  avg moves {:.0}  {:.0}s ({:.0} moves/s)",
        n,
        pct(&|r| r.0 >= 15 || r.1),
        pct(&|r| r.1),
        pct(&|r| r.2),
        moves as f64 / n as f64,
        start.elapsed().as_secs_f64(),
        moves as f64 / start.elapsed().as_secs_f64()
    );
}

/// Plays games with `--net` and, wherever a move can make 32768, scores every move as
/// reward + V(afterstate) under both nets; on boards holding 32768 it also compares each
/// net's V with the score the game actually went on to make.
fn boundary(args: &[String]) {
    let games: u64 = positional(args).first().and_then(|s| s.parse().ok()).unwrap_or(64);
    let load = |f: &str| Arc::new(NTuple::load(f).unwrap_or_else(|e| panic!("loading {f}: {e}")));
    let a = load(&flag::<String>(args, "--net").unwrap_or_else(|| panic!("{USAGE}")));
    let b = load(&flag::<String>(args, "--vs").unwrap_or_else(|| panic!("{USAGE}")));
    let ai = Arc::new(ai::Ai::with_net(a.clone(), 2));
    // Per game: (merge chances as [gap under a, gap under b]), (V a, V b, actual return) on 32768 boards.
    let per = parallel(games, move |g| {
        let t = ai.tables();
        let mut rng = Rng((1 + g).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut bd = spawn(spawn(0, &mut rng), &mut rng);
        let (mut score, mut chances, mut vals) = (0u64, vec![], vec![]);
        while let Some(d) = ai.best_move(bd) {
            if max_rank(bd) < 15 {
                let q = |n: &NTuple| -> (Option<f32>, f32) {
                    let (mut merge, mut other) = (None, f32::MIN);
                    for m in DIRS {
                        let (nb, r) = t.apply(bd, m);
                        if nb == bd { continue; }
                        let v = r as f32 + n.value(nb);
                        if max_rank(nb) >= 15 { merge = Some(merge.map_or(v, |x: f32| x.max(v))); } else { other = other.max(v); }
                    }
                    (merge, other)
                };
                if let ((Some(ma), oa), (Some(mb), ob)) = (q(&a), q(&b)) {
                    if oa > f32::MIN { chances.push([ma - oa, mb - ob]); }
                }
            } else {
                vals.push((a.value(bd), b.value(bd), score));
            }
            let (nb, r) = t.apply(bd, d);
            score += r as u64;
            bd = spawn(nb, &mut rng);
        }
        let vals: Vec<[f64; 3]> = vals.into_iter().map(|(va, vb, s)| [va as f64, vb as f64, (score - s) as f64]).collect();
        (chances, vals)
    });
    let chances: Vec<[f32; 2]> = per.iter().flat_map(|p| p.0.clone()).collect();
    let vals: Vec<[f64; 3]> = per.iter().flat_map(|p| p.1.clone()).collect();
    let n = chances.len().max(1) as f64;
    let mean = |i: usize| chances.iter().map(|c| c[i] as f64).sum::<f64>() / n;
    let best = |i: usize| 100.0 * chances.iter().filter(|c| c[i] > 0.0).count() as f64 / n;
    println!("{} positions where a move makes 32768 ({} games)", chances.len(), games);
    println!("  merge minus best other move:  --net {:>9.0} (merge best {:>5.1}%)   --vs {:>9.0} (merge best {:>5.1}%)", mean(0), best(0), mean(1), best(1));
    let m = vals.len().max(1) as f64;
    let avg = |i: usize| vals.iter().map(|v| v[i]).sum::<f64>() / m;
    println!("{} boards holding 32768: mean V  --net {:.0}  --vs {:.0}  actual score still to come {:.0}", vals.len(), avg(0), avg(1), avg(2));
}

/// `g2048 probe POS_FILE --tables DIR [--layouts L]`: for each saved board, the table's
/// move and goal probability (or "-"), to compare against another implementation.
fn probe(args: &[String]) {
    let file = positional(args).first().unwrap_or_else(|| panic!("{USAGE}")).to_string();
    let boards = read_boards(&file);
    let lk = lookup(args).unwrap_or_else(|| panic!("probe needs --tables DIR"));
    for b in boards {
        match lk.suggest(b, Some(-1.0)) {
            Some((d, p)) => println!("{b:016x} {} {p:.4}", format!("{d:?}").to_lowercase()),
            None => println!("{b:016x} - -"),
        }
    }
    let _ = lk.save();
}

fn bench(args: &[String]) {
    let pos = positional(args);
    let games: u64 = pos.first().and_then(|s| s.parse().ok()).unwrap_or(16);
    let seed0: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ai = Arc::new(net_ai(args, 2));
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
    finish_lookup(&ai, results.iter().map(|r| r.moves).sum());
}

/// `g2048 formation POS_FILE [--layouts L]`: how many saved boards some endgame layout
/// applies to, in any orientation, so a net's corner shape can be checked against the
/// tables before playing with them.
fn formation(args: &[String]) {
    let file = positional(args).first().unwrap_or_else(|| panic!("{USAGE}")).to_string();
    let boards = read_boards(&file);
    let names: String = flag(args, "--layouts").unwrap_or_else(|| "block10,five7,snake8".into());
    let mut counts: Vec<(String, usize)> = Vec::new();
    for name in names.split(',') {
        let layout = endgame::Layout::parse(name.trim()).unwrap_or_else(|| panic!("unknown layout {name}"));
        let lk = endgame::Lookup::open(&std::env::temp_dir().join("g2048-formation"), &[layout]).expect("tables");
        let n = boards.iter().filter(|&&b| lk.applies(b).is_some()).count();
        counts.push((name.trim().to_string(), n));
    }
    println!("{} boards", boards.len());
    for (name, n) in counts {
        println!("  {name:<10} applies to {n:>6} ({:.1}%)", 100.0 * n as f64 / boards.len().max(1) as f64);
    }
}

fn train(args: &[String]) {
    let pos = positional(args);
    let out = pos.first().unwrap_or_else(|| panic!("{USAGE}")).to_string();
    let games: u64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let alpha_flag: Option<f32> = flag(args, "--alpha");
    let seed: u64 = flag(args, "--seed").unwrap_or(42);
    let mut net = match flag::<String>(args, "--resume") {
        Some(p) => NTuple::load(&p).unwrap_or_else(|e| panic!("loading {p}: {e}")),
        None => {
            let tuples: &[[usize; 6]] = if flag::<u8>(args, "--tuples") == Some(8) { &ntuple::TUPLES_8 } else { &ntuple::TUPLES_4 };
            // Optimistic initialization: --init is the starting value of every board, spread
            // evenly over the 8 symmetric lookups of each tuple (320000 in Guei et al.).
            let init: f32 = flag(args, "--init").unwrap_or(0.0);
            NTuple::new(init / (8 * tuples.len()) as f32, 1, tuples)
        }
    };
    net.expand_stages(flag(args, "--stages").unwrap_or(1));
    let restart_p: f32 = flag(args, "--restart").unwrap_or(0.0);
    let restart_stage: usize = flag(args, "--restart-stage").unwrap_or(1);
    // Keep the lower stages fixed while training a higher one, so sharpening the
    // top stage (the 65536 regime) can't un-learn the early game (catastrophic
    // forgetting). --freeze N stops `update` touching stages below N.
    if let Some(f) = flag::<usize>(args, "--freeze") {
        net.set_frozen(f);
    }
    let pool = ntuple::Pool::new(100_000);
    // Seed the restart pool from a file of boards (e.g. generated 65536 boards),
    // so `--restart-stage N` training gets states self-play almost never reaches.
    // pool_key maps a 65536 board to a stage-3 key, so --restart-stage 3 can then
    // restart from real 65536 positions and train the 65536 -> 131072 conversion.
    if let Some(f) = flag::<String>(args, "--restart-file") {
        let mut rng = Rng(seed ^ 0x5eed_1234);
        let boards = read_boards(&f);
        let seeded = boards.iter().filter(|&&b| ntuple::pool_key(b).map(|k| pool.add(k, b, &mut rng)).is_some()).count();
        println!("seeded restart pool with {seeded}/{} boards from {f}", boards.len());
    }
    if flag::<u8>(args, "--tc") == Some(1) {
        net.enable_tc();
    }
    // Per-weight step: 0.1 spread over the weights each board touches (8 per tuple).
    let alpha = alpha_flag.unwrap_or(0.1 / (8 * net.tuple_count()) as f32);
    const WINDOW: u64 = 10_000;
    let windows_done = AtomicU64::new(0);
    // Per-window tallies: [score sum, games, >=2048, >=4096, >=8192, >=16384, >=32768, restarts, restarts that progressed]
    let stats: Vec<AtomicU64> = (0..9).map(|_| AtomicU64::new(0)).collect();
    let start = Instant::now();
    ntuple::train_parallel(&net, &pool, alpha, restart_p, restart_stage, seed, threads(games), games, None, &|g, e| {
        // Only fresh games say how strong the net is; restarts begin mid-game.
        if !e.fresh {
            stats[7].fetch_add(1, Ordering::Relaxed);
            stats[8].fetch_add(e.progressed() as u64, Ordering::Relaxed);
            return;
        }
        stats[0].fetch_add(e.score, Ordering::Relaxed);
        for (i, k) in [11u8, 12, 13, 14, 15].iter().enumerate() {
            if e.max_rank >= *k {
                stats[2 + i].fetch_add(1, Ordering::Relaxed);
            }
        }
        if stats[1].fetch_add(1, Ordering::Relaxed) + 1 == WINDOW {
            let v: Vec<u64> = stats.iter().map(|s| s.swap(0, Ordering::Relaxed)).collect();
            let pct = |x: u64| 100.0 * x as f64 / v[1] as f64;
            println!(
                "{:>9} games {:>6.0}s  mean {:>7.0}  2048 {:>5.1}%  4096 {:>5.1}%  8192 {:>5.1}%  16384 {:>4.1}%  32768 {:>4.2}%  restarts {} ({} progressed)  pool {}",
                g + 1, start.elapsed().as_secs_f64(), v[0] as f64 / v[1] as f64, pct(v[2]), pct(v[3]), pct(v[4]), pct(v[5]), pct(v[6]), v[7], v[8], pool.total()
            );
            if windows_done.fetch_add(1, Ordering::Relaxed) % 10 == 9 {
                net.save(&out).expect("saving weights");
            }
        }
    });
    net.save(&out).expect("saving weights");
    println!("saved {out}");
}

/// `g2048 stages FILE [N]`: reports a saved net's shape, and with N grows it to N stages,
/// each new stage a copy of the last (run on the server, coordinator stopped).
fn stages(args: &[String]) {
    let pos = positional(args);
    let file = pos.first().unwrap_or_else(|| panic!("{USAGE}")).to_string();
    let mut net = NTuple::load(&file).unwrap_or_else(|e| panic!("loading {file}: {e}"));
    println!("{file}: {} stages, {} tuples, {} MB", net.stages(), net.tuple_count(), net.stages() * net.stage_size() * 4 >> 20);
    if let Some(n) = pos.get(1).and_then(|s| s.parse::<usize>().ok()) {
        if n <= net.stages() {
            println!("already has {} stages, nothing to do", net.stages());
            return;
        }
        let from = net.stages() - 1;
        net.expand_stages(n);
        net.save(&file).expect("saving weights");
        println!("grown to {n} stages (the new ones copied from stage {from}), saved");
    }
}

/// Board from 16 hex digits, one tile rank per cell, row-major from the top-left.
fn parse_board(hex: &str) -> Option<Board> {
    if hex.len() != 16 {
        return None;
    }
    hex.chars().enumerate().try_fold(0u128, |b, (i, ch)| Some(b | (ch.to_digit(16)? as u128) << (5 * i)))
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

/// Shared secret from --token-file (or --token), so it stays out of shell history and ps.
fn token(args: &[String]) -> String {
    match flag::<String>(args, "--token-file") {
        Some(f) => std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("reading {f}: {e}")).trim().to_string(),
        None => flag(args, "--token").unwrap_or_else(|| panic!("{USAGE}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => println!("{}", dist::BUILD),
        Some("bench") => bench(&args[1..]),
        Some("train") => train(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("positions") => positions(&args[1..]),
        Some("endgame") => endgame(&args[1..]),
        Some("formation") => formation(&args[1..]),
        Some("probe") => probe(&args[1..]),
        Some("stages") => stages(&args[1..]),
        Some("boundary") => boundary(&args[1..]),
        Some("coord") => {
            let a = &args[1..];
            dist::coord(flag(a, "--net").unwrap_or_else(|| panic!("{USAGE}")), token(a), flag(a, "--port").unwrap_or(20490))
        }
        Some("worker") => {
            let a = &args[1..];
            let url: String = flag(a, "--url").unwrap_or_else(|| panic!("{USAGE}"));
            let cache = flag::<String>(a, "--cache").unwrap_or_else(|| "g2048-cache".into());
            if std::env::var_os(dist::CHILD_ENV).is_none() {
                return dist::supervise(&url);
            }
            dist::worker(url, token(a), flag(a, "--name"), flag(a, "--threads").unwrap_or_else(|| threads(u64::MAX)), cache.into())
        }
        _ => eprintln!("{USAGE}"),
    }
}
