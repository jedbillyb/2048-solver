//! N-tuple network value function over afterstates (Szubert & Jaskowski style),
//! 4 six-tuples x 8 board symmetries, trained by TD(0). Weights are shared across
//! training threads Hogwild-style through relaxed atomics.
//!
//! The network can be split into stages by the largest tile on the board
//! (below 16384 / 16384 / 32768), each with its own weights, because endgame
//! positions want a very different evaluation from the opening.

use crate::board::*;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::Mutex;

const BASE_TUPLES: [[usize; 6]; 4] = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10],
];
const TUPLE_SIZE: usize = 1 << 24;
const STAGE_SIZE: usize = BASE_TUPLES.len() * TUPLE_SIZE;
/// Rank of the tile that opens stage 1 (16384); each rank above opens the next stage.
const FIRST_STAGE_RANK: u8 = 14;
const MAGIC_V1: &[u8; 8] = b"2048NT01";
const MAGIC_V2: &[u8; 8] = b"2048NT02";

pub struct NTuple {
    w: Vec<AtomicU32>,
    stages: usize,
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
    pub fn new(init: f32, stages: usize) -> Self {
        let cells = BASE_TUPLES
            .iter()
            .map(|t| std::array::from_fn(|s| t.map(|c| symmetries(c)[s])))
            .collect();
        let bits = init.to_bits();
        let stages = stages.max(1);
        NTuple { w: (0..stages * STAGE_SIZE).map(|_| AtomicU32::new(bits)).collect(), stages, tc: vec![], cells }
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
        let last = (self.stages - 1) * STAGE_SIZE;
        for _ in self.stages..n {
            for i in 0..STAGE_SIZE {
                self.w.push(AtomicU32::new(self.w[last + i].load(Relaxed)));
            }
        }
        self.stages = n;
    }

    /// Switch on temporal coherence learning: each weight's step is scaled by
    /// |sum of its errors| / sum of |errors|, so noisy weights slow down on their own.
    pub fn enable_tc(&mut self) {
        self.tc = (0..self.w.len()).map(|_| (AtomicU32::new(0), AtomicU32::new(0))).collect();
    }

    #[inline]
    fn indices(&self, b: Board, out: &mut [usize; 32]) {
        let base = self.stage(b) * STAGE_SIZE;
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
    }

    #[inline]
    pub fn value(&self, b: Board) -> f32 {
        let mut ix = [0; 32];
        self.indices(b, &mut ix);
        ix.iter().map(|&i| f32::from_bits(self.w[i].load(Relaxed))).sum()
    }

    /// Moves every weight the board touches by `alpha * err` (scaled per weight under TC).
    pub fn update(&self, b: Board, alpha: f32, err: f32) {
        let mut ix = [0; 32];
        self.indices(b, &mut ix);
        let load = |a: &AtomicU32| f32::from_bits(a.load(Relaxed));
        for &i in &ix {
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
        f.write_all(MAGIC_V2)?;
        f.write_all(&(self.stages as u32).to_le_bytes())?;
        for w in &self.w {
            f.write_all(&w.load(Relaxed).to_le_bytes())?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        let mut buf = [0u8; 4];
        let stages = match &magic {
            m if m == MAGIC_V1 => 1,
            m if m == MAGIC_V2 => {
                f.read_exact(&mut buf)?;
                u32::from_le_bytes(buf) as usize
            }
            _ => return Err(std::io::Error::other("not an n-tuple weights file")),
        };
        let net = NTuple::new(0.0, stages);
        for w in &net.w {
            f.read_exact(&mut buf)?;
            w.store(u32::from_le_bytes(buf), Relaxed);
        }
        Ok(net)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_boards_share_value() {
        let net = NTuple::new(0.0, 1);
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
    fn stages_split_on_big_tiles() {
        let mut net = NTuple::new(0.0, 1);
        net.expand_stages(3);
        assert_eq!(net.stage(0x0000_0000_0000_00D1), 0); // 8192
        assert_eq!(net.stage(0x0000_0000_0000_00E1), 1); // 16384
        assert_eq!(net.stage(0x0000_0000_0000_00F1), 2); // 32768
        let one = NTuple::new(0.0, 1);
        assert_eq!(one.stage(0x0000_0000_0000_00F1), 0);
    }
}
