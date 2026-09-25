//! N-tuple network value function over afterstates (Szubert & Jaskowski style),
//! 4 six-tuples x 8 board symmetries, trained by TD(0). Weights are shared across
//! training threads Hogwild-style through relaxed atomics.

use crate::board::*;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

const BASE_TUPLES: [[usize; 6]; 4] = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10],
];
const TUPLE_SIZE: usize = 1 << 24;
const MAGIC: &[u8; 8] = b"2048NT01";

pub struct NTuple {
    w: Vec<AtomicU32>,
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
    pub fn new(init: f32) -> Self {
        let cells = BASE_TUPLES
            .iter()
            .map(|t| std::array::from_fn(|s| t.map(|c| symmetries(c)[s])))
            .collect();
        let bits = init.to_bits();
        NTuple { w: (0..BASE_TUPLES.len() * TUPLE_SIZE).map(|_| AtomicU32::new(bits)).collect(), tc: vec![], cells }
    }

    /// Switch on temporal coherence learning: each weight's step is scaled by
    /// |sum of its errors| / sum of |errors|, so noisy weights slow down on their own.
    pub fn enable_tc(&mut self) {
        self.tc = (0..self.w.len()).map(|_| (AtomicU32::new(0), AtomicU32::new(0))).collect();
    }

    #[inline]
    fn indices(&self, b: Board, out: &mut [usize; 32]) {
        let mut n = 0;
        for (t, syms) in self.cells.iter().enumerate() {
            for s in syms {
                let mut idx = 0usize;
                for &c in s {
                    idx = (idx << 4) | ((b >> (4 * c)) & 0xF) as usize;
                }
                out[n] = t * TUPLE_SIZE + idx;
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
        f.write_all(MAGIC)?;
        for w in &self.w {
            f.write_all(&w.load(Relaxed).to_le_bytes())?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let net = NTuple::new(0.0);
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::other("not an n-tuple weights file"));
        }
        let mut buf = [0u8; 4];
        for w in &net.w {
            f.read_exact(&mut buf)?;
            w.store(u32::from_le_bytes(buf), Relaxed);
        }
        Ok(net)
    }
}

pub struct EpisodeStats {
    pub score: u64,
    pub max_rank: u8,
}

/// Plays one greedy (1-ply) game, learning from it with TD(0) on afterstates.
pub fn train_episode(net: &NTuple, t: &Tables, rng: &mut Rng, alpha: f32) -> EpisodeStats {
    let mut b = spawn(spawn(0, rng), rng);
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
                return EpisodeStats { score, max_rank: max_rank(b) };
            }
            Some((after, r, v)) => {
                if let Some(p) = prev_after {
                    net.update(p, alpha, v - net.value(p));
                }
                score += r as u64;
                prev_after = Some(after);
                b = spawn(after, rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_boards_share_value() {
        let net = NTuple::new(0.0);
        let t = Tables::new();
        let mut rng = Rng(7);
        for _ in 0..50 {
            train_episode(&net, &t, &mut rng, 0.01);
        }
        let b: Board = 0x0000_0012_0341_1235;
        let v = net.value(b);
        assert!((net.value(transpose(b)) - v).abs() < 1e-2 * v.abs().max(1.0));
    }
}
