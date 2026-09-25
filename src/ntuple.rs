//! N-tuple network value function over afterstates (Szubert & Jaskowski style),
//! 4 or 8 six-tuples x 8 board symmetries, trained by TD(0). Weights are shared across
//! training threads Hogwild-style through relaxed atomics.
//!
//! The network can be split into stages by the largest tile on the board
//! (below 16384 / 16384 / 32768), each with its own weights, because endgame
//! positions want a very different evaluation from the opening.

use crate::board::*;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::Mutex;

/// Cells are row-major, 0 = top-left. TUPLES_4 is Yeh's 4x6-tuple set; TUPLES_8 is
/// Matsuzaki's 8x6-tuple set, the one behind the best published results (Guei et al. 2021).
/// Saved files record their own tuples, so older nets with other shapes still load.
pub const TUPLES_4: [[usize; 6]; 4] = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10],
];
pub const TUPLES_8: [[usize; 6]; 8] = [
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6, 9],
    [0, 1, 2, 5, 9, 10],
    [3, 4, 5, 6, 7, 8],
    [1, 3, 4, 5, 6, 7],
    [0, 1, 4, 8, 9, 10],
];
const MAX_TUPLES: usize = 8;
const TUPLE_SIZE: usize = 1 << 24;
/// Rank of the tile that opens stage 1 (16384); each rank above opens the next stage.
const FIRST_STAGE_RANK: u8 = 14;
const MAGIC_V1: &[u8; 8] = b"2048NT01";
const MAGIC_V2: &[u8; 8] = b"2048NT02";
const MAGIC_V3: &[u8; 8] = b"2048NT03";

pub struct NTuple {
    w: Vec<AtomicU32>,
    stages: usize,
    tuples: Vec<[usize; 6]>,
    /// Temporal coherence accumulators (sum of errors, sum of |errors|) per weight;
    /// empty unless TC learning is enabled.
    tc: Vec<(AtomicU32, AtomicU32)>,
    /// [tuple][symmetry] -> 6 cell indices
    cells: Vec<[[usize; 6]; 8]>,
}

fn symmetries(c: usize) -> [usize; 8] {
    let (r, k) = (c / 4, c % 4);
    let pts = [(r, k), (k, 3 - r), (3 - r, 3 - k), (3 - k, r), (r, 3 - k), (3 - k, 3 - r), (3 - r, k), (k, r)];
    pts.map(|(a, b)| a * 4 + b)
}

impl NTuple {
    pub fn new(init: f32, stages: usize, tuples: &[[usize; 6]]) -> Self {
        assert!(!tuples.is_empty() && tuples.len() <= MAX_TUPLES);
        let cells = tuples
            .iter()
            .map(|t| std::array::from_fn(|s| t.map(|c| symmetries(c)[s])))
            .collect();
        let bits = init.to_bits();
        let stages = stages.max(1);
        let n = stages * tuples.len() * TUPLE_SIZE;
        NTuple { w: (0..n).map(|_| AtomicU32::new(bits)).collect(), stages, tuples: tuples.to_vec(), tc: vec![], cells }
    }

    #[inline]
    fn stage_size(&self) -> usize {
        self.tuples.len() * TUPLE_SIZE
    }

    pub fn tuple_count(&self) -> usize {
        self.tuples.len()
    }

    pub fn stages(&self) -> usize {
        self.stages
    }

    #[inline]
    pub fn stage(&self, b: Board) -> usize {
        (max_rank(b).saturating_sub(FIRST_STAGE_RANK - 1) as usize).min(self.stages - 1)
    }

    /// Grow to `n` stages, seeding each new stage with a copy of the last existing one.
    pub fn expand_stages(&mut self, n: usize) {
        if n <= self.stages {
            return;
        }
        let size = self.stage_size();
        let last = (self.stages - 1) * size;
        for _ in self.stages..n {
            for i in 0..size {
                self.w.push(AtomicU32::new(self.w[last + i].load(Relaxed)));
            }
        }
        self.stages = n;
    }

    pub fn tc_enabled(&self) -> bool {
        !self.tc.is_empty()
    }

    /// Switch on temporal coherence learning: each weight's step is scaled by
    /// |sum of its errors| / sum of |errors|, so noisy weights slow down on their own.
    pub fn enable_tc(&mut self) {
        self.tc = (0..self.w.len()).map(|_| (AtomicU32::new(0), AtomicU32::new(0))).collect();
    }

    #[inline]
    fn indices(&self, b: Board, out: &mut [usize; 8 * MAX_TUPLES]) -> usize {
        let b = downgrade(b);
        let base = self.stage(b) * self.stage_size();
        let mut n = 0;
        for (t, syms) in self.cells.iter().enumerate() {
            for s in syms {
                let mut idx = 0usize;
                for &c in s {
                    idx = (idx << 4) | ((b >> (4 * c)) & 0xF) as usize;
                }
                out[n] = base + t * TUPLE_SIZE + idx;
                n += 1;
            }
        }
        n
    }

    #[inline]
    pub fn value(&self, b: Board) -> f32 {
        let mut ix = [0; 8 * MAX_TUPLES];
        let n = self.indices(b, &mut ix);
        ix[..n].iter().map(|&i| f32::from_bits(self.w[i].load(Relaxed))).sum()
    }

    /// Moves every weight the board touches by `alpha * err` (scaled per weight under TC).
    pub fn update(&self, b: Board, alpha: f32, err: f32) {
        let mut ix = [0; 8 * MAX_TUPLES];
        let n = self.indices(b, &mut ix);
        let load = |a: &AtomicU32| f32::from_bits(a.load(Relaxed));
        for &i in &ix[..n] {
            let mut step = alpha * err;
            if let Some((e, a)) = self.tc.get(i) {
                let (ev, av) = (load(e), load(a));
                if av > 0.0 {
                    step *= ev.abs() / av;
                }
                e.store((ev + err).to_bits(), Relaxed);
                a.store((av + err.abs()).to_bits(), Relaxed);
            }
            let w = &self.w[i];
            w.store((load(w) + step).to_bits(), Relaxed);
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let tmp = format!("{path}.tmp");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(MAGIC_V3)?;
        f.write_all(&(self.stages as u32).to_le_bytes())?;
        f.write_all(&(self.tuples.len() as u32).to_le_bytes())?;
        for t in &self.tuples {
            f.write_all(&t.map(|c| c as u8))?;
        }
        for w in &self.w {
            f.write_all(&w.load(Relaxed).to_le_bytes())?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_from(std::io::BufReader::new(std::fs::File::open(path)?))
    }

    pub fn load_from(mut f: impl Read) -> std::io::Result<Self> {
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        let mut buf = [0u8; 4];
        let mut read_u32 = |f: &mut dyn Read| -> std::io::Result<usize> {
            f.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf) as usize)
        };
        let (stages, tuples) = match &magic {
            m if m == MAGIC_V1 => (1, TUPLES_4.to_vec()),
            m if m == MAGIC_V2 => (read_u32(&mut f)?, TUPLES_4.to_vec()),
            m if m == MAGIC_V3 => {
                let stages = read_u32(&mut f)?;
                let n = read_u32(&mut f)?;
                let mut tuples = vec![];
                for _ in 0..n {
                    let mut t = [0u8; 6];
                    f.read_exact(&mut t)?;
                    tuples.push(t.map(|c| c as usize));
                }
                (stages, tuples)
            }
            _ => return Err(std::io::Error::other("not an n-tuple weights file")),
        };
        let net = NTuple::new(0.0, stages, &tuples);
        let mut buf = [0u8; 4];
        for w in &net.w {
            f.read_exact(&mut buf)?;
            w.store(u32::from_le_bytes(buf), Relaxed);
        }
        Ok(net)
    }

    /// Current weights, to diff against after some training.
    pub fn snapshot(&self) -> Vec<f32> {
        self.w.iter().map(|w| f32::from_bits(w.load(Relaxed))).collect()
    }

    /// (index, current - base) for every weight that differs from `base`.
    pub fn diff(&self, base: &[f32]) -> Vec<(u32, f32)> {
        self.w
            .iter()
            .zip(base)
            .enumerate()
            .filter_map(|(i, (w, &old))| {
                let d = f32::from_bits(w.load(Relaxed)) - old;
                (d != 0.0).then_some((i as u32, d))
            })
            .collect()
    }

    /// Adds a delta from `diff` (another machine's training) into these weights.
    pub fn apply(&self, delta: &[(u32, f32)]) {
        for &(i, d) in delta {
            if let Some(w) = self.w.get(i as usize) {
                w.store((f32::from_bits(w.load(Relaxed)) + d).to_bits(), Relaxed);
            }
        }
    }
}

/// Training never reaches 32768, so weights for that tile stay near 0 and the net would
/// refuse the merge that makes it. Boards holding 32768 are therefore valued as the same
/// board one step down: every tile above the largest missing rank is halved, which turns
/// the fresh 32768 (whose 16384 just merged away) into a familiar 16384 position.
#[inline]
pub fn downgrade(b: Board) -> Board {
    if max_rank(b) < 15 {
        return b;
    }
    let mut present = 0u32;
    for i in 0..16 {
        present |= 1 << ((b >> (4 * i)) & 0xF);
    }
    let missing = match (1..15u32).rev().find(|r| present & (1 << r) == 0) {
        Some(m) => m,
        None => return b,
    };
    let mut out = 0u64;
    for i in 0..16 {
        let r = (b >> (4 * i)) & 0xF;
        let r = if r as u32 > missing { r - 1 } else { r };
        out |= r << (4 * i);
    }
    out
}

/// Boards seen the first time a game entered each stage >= 1, so training games
/// can restart there and practise endgames far more often than fresh games reach them.
pub struct RestartPool {
    boards: Vec<Mutex<Vec<Board>>>,
    cap: usize,
}

impl RestartPool {
    pub fn new(stages: usize, cap: usize) -> Self {
        RestartPool { boards: (0..stages).map(|_| Mutex::new(vec![])).collect(), cap }
    }

    fn add(&self, stage: usize, b: Board, rng: &mut Rng) {
        let mut v = self.boards[stage].lock().unwrap();
        if v.len() < self.cap {
            v.push(b);
        } else {
            let i = rng.below(v.len() as u32) as usize;
            v[i] = b;
        }
    }

    /// A random saved board from a random non-empty stage, if any.
    fn pick(&self, rng: &mut Rng) -> Option<Board> {
        let filled: Vec<_> = self.boards.iter().filter(|m| !m.lock().unwrap().is_empty()).collect();
        if filled.is_empty() {
            return None;
        }
        let v = filled[rng.below(filled.len() as u32) as usize].lock().unwrap();
        Some(v[rng.below(v.len() as u32) as usize])
    }

    pub fn sizes(&self) -> Vec<usize> {
        self.boards.iter().map(|m| m.lock().unwrap().len()).collect()
    }
}

pub struct EpisodeStats {
    pub score: u64,
    pub max_rank: u8,
    /// False when the episode started from a restart-pool board.
    pub fresh: bool,
}

/// Plays one greedy (1-ply) game, learning from it with TD(0) on afterstates.
/// With probability `restart_p` it starts from a saved endgame board instead of a new game.
pub fn train_episode(net: &NTuple, t: &Tables, rng: &mut Rng, alpha: f32, pool: &RestartPool, restart_p: f32) -> EpisodeStats {
    let restart = if restart_p > 0.0 && (rng.below(1_000_000) as f32) < restart_p * 1e6 { pool.pick(rng) } else { None };
    let fresh = restart.is_none();
    let mut b = restart.unwrap_or_else(|| spawn(spawn(0, rng), rng));
    let mut stage = net.stage(b);
    let mut prev_after: Option<Board> = None;
    let mut score = 0u64;
    loop {
        let mut best: Option<(Board, u32, f32)> = None;
        for d in DIRS {
            let (nb, r) = t.apply(b, d);
            if nb == b {
                continue;
            }
            let v = r as f32 + net.value(nb);
            if best.map_or(true, |(_, _, bv)| v > bv) {
                best = Some((nb, r, v));
            }
        }
        match best {
            None => {
                if let Some(p) = prev_after {
                    net.update(p, alpha, 0.0 - net.value(p));
                }
                return EpisodeStats { score, max_rank: max_rank(b), fresh };
            }
            Some((after, r, v)) => {
                if let Some(p) = prev_after {
                    net.update(p, alpha, v - net.value(p));
                }
                score += r as u64;
                prev_after = Some(after);
                b = spawn(after, rng);
                let s = net.stage(b);
                if s > stage {
                    stage = s;
                    pool.add(s, b, rng);
                }
            }
        }
    }
}

/// Runs training episodes on `threads` threads until `games` have been played or
/// `deadline` passes, handing each finished episode to `each` with its game number.
pub fn train_parallel(
    net: &NTuple,
    pool: &RestartPool,
    alpha: f32,
    restart_p: f32,
    seed: u64,
    threads: usize,
    games: u64,
    deadline: Option<std::time::Instant>,
    each: &(dyn Fn(u64, &EpisodeStats) + Sync),
) {
    let t = Tables::new();
    let next = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|s| {
        for tid in 0..threads as u64 {
            let (t, next) = (&t, &next);
            s.spawn(move || {
                let mut rng = Rng((seed ^ (tid + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
                loop {
                    let g = next.fetch_add(1, Relaxed);
                    if g >= games || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                        break;
                    }
                    each(g, &train_episode(net, t, &mut rng, alpha, pool, restart_p));
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_boards_share_value() {
        let net = NTuple::new(0.0, 1, &TUPLES_4);
        let t = Tables::new();
        let mut rng = Rng(7);
        let pool = RestartPool::new(1, 10);
        for _ in 0..50 {
            train_episode(&net, &t, &mut rng, 0.01, &pool, 0.0);
        }
        let b: Board = 0x0000_0012_0341_1235;
        let v = net.value(b);
        assert!((net.value(transpose(b)) - v).abs() < 1e-2 * v.abs().max(1.0));
    }

    #[test]
    fn diff_then_apply_reproduces_training() {
        let a = NTuple::new(0.0, 1, &TUPLES_4);
        let b = NTuple::new(0.0, 1, &TUPLES_4);
        let snap = a.snapshot();
        let pool = RestartPool::new(1, 10);
        train_parallel(&a, &pool, 0.01, 0.0, 3, 1, 20, None, &|_, _| {});
        let delta = a.diff(&snap);
        assert!(!delta.is_empty());
        b.apply(&delta);
        assert_eq!(a.snapshot(), b.snapshot());
        assert!(a.diff(&b.snapshot()).is_empty());
    }

    #[test]
    fn downgrade_turns_fresh_32768_into_16384() {
        // 32768, 8192, 4096 and a 2: no 16384 left, so 32768 -> 16384 and nothing else moves.
        let b: Board = 0x0000_0000_1000_CDF0;
        assert_eq!(downgrade(b), 0x0000_0000_1000_CDE0);
        // Boards without 32768 are untouched.
        assert_eq!(downgrade(0x0000_0000_1000_CDE0), 0x0000_0000_1000_CDE0);
    }

    #[test]
    fn stages_split_on_big_tiles() {
        let mut net = NTuple::new(0.0, 1, &TUPLES_4);
        net.expand_stages(3);
        assert_eq!(net.stage(0x0000_0000_0000_00D1), 0); // 8192
        assert_eq!(net.stage(0x0000_0000_0000_00E1), 1); // 16384
        assert_eq!(net.stage(0x0000_0000_0000_00F1), 2); // 32768
        let one = NTuple::new(0.0, 1, &TUPLES_4);
        assert_eq!(one.stage(0x0000_0000_0000_00F1), 0);
    }
}
