//! N-tuple network value function over afterstates (Szubert & Jaskowski style),
//! 4 or 8 six-tuples x 8 board symmetries, trained by TD(0). Weights are shared across
//! training threads Hogwild-style through relaxed atomics.
//!
//! The network can be split into stages by the largest tile on the board
//! (below 16384 / 16384 / 32768 / 65536), each with its own weights, because endgame
//! positions want a very different evaluation from the opening. The stage is taken from
//! the board as it is; the tile pattern inside the stage goes through `downgrade` so a
//! board with a 32768 is looked up as the familiar 16384 board one step down.

use crate::board::*;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};
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
    /// Stages below this one are left as they are by `update` (later stages train on their own).
    frozen: AtomicUsize,
    /// [tuple][symmetry] -> 6 cell indices
    cells: Vec<[[usize; 6]; 8]>,
}

/// A vec of `n` items from `f`, backed by 2 MB pages on Linux where possible. The weights
/// are read at random, and with 4 KB pages most reads first miss the TLB (~4% faster).
fn huge_vec<T>(n: usize, f: impl FnMut(usize) -> T) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
        }
        const MADV_HUGEPAGE: i32 = 14;
        let (start, end) = (v.as_mut_ptr() as usize, v.as_mut_ptr() as usize + n * std::mem::size_of::<T>());
        let start = (start + 4095) & !4095;
        if end > start {
            // Advice only: if the kernel refuses, the vec just stays on normal pages.
            unsafe { madvise(start as *mut u8, end - start, MADV_HUGEPAGE) };
        }
    }
    v.extend((0..n).map(f));
    v
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
        NTuple { w: huge_vec(n, |_| AtomicU32::new(bits)), stages, tuples: tuples.to_vec(), tc: vec![], frozen: AtomicUsize::new(0), cells }
    }

    /// Weights per stage; stage `s` owns indices `s * stage_size() ..`.
    #[inline]
    pub fn stage_size(&self) -> usize {
        self.tuples.len() * TUPLE_SIZE
    }

    /// First weight index that `update` may touch under the current freeze.
    pub fn frozen_end(&self) -> usize {
        self.frozen.load(Relaxed).min(self.stages) * self.stage_size()
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
        self.w.reserve((n - self.stages) * size);
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

    /// Stop `update` from touching stages below `s`.
    pub fn set_frozen(&self, s: usize) {
        self.frozen.store(s, Relaxed);
    }

    /// Switch on temporal coherence learning: each weight's step is scaled by
    /// |sum of its errors| / sum of |errors|, so noisy weights slow down on their own.
    pub fn enable_tc(&mut self) {
        self.tc = huge_vec(self.w.len(), |_| (AtomicU32::new(0), AtomicU32::new(0)));
    }

    #[inline]
    fn indices(&self, b: Board, out: &mut [usize; 8 * MAX_TUPLES]) -> usize {
        let base = self.stage(b) * self.stage_size();
        let b = downgrade(b);
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
        if ix[0] / self.stage_size() < self.frozen.load(Relaxed) {
            return;
        }
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
    #[cfg(test)]
    pub fn snapshot(&self) -> Vec<f32> {
        self.snapshot_from(0)
    }

    /// The weights from index `start` on. Frozen stages never change, so a worker keeps
    /// its copy of the master only from the first trainable weight.
    pub fn snapshot_from(&self, start: usize) -> Vec<f32> {
        self.w[start.min(self.w.len())..].iter().map(|w| f32::from_bits(w.load(Relaxed))).collect()
    }

    /// (index, current - base) for every weight that differs from `base`.
    #[cfg(test)]
    pub fn diff(&self, base: &[f32]) -> Vec<(u32, f32)> {
        self.diff_from(base, 0)
    }

    /// Like `diff`, with `base` holding the weights from index `start` on.
    pub fn diff_from(&self, base: &[f32], start: usize) -> Vec<(u32, f32)> {
        self.w[start.min(self.w.len())..]
            .iter()
            .zip(base)
            .enumerate()
            .filter_map(|(i, (w, &old))| {
                let d = f32::from_bits(w.load(Relaxed)) - old;
                (d != 0.0).then_some(((start + i) as u32, d))
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

/// The chain state of a board, the unit the restart pool is keyed by. `max` is the rank
/// of the largest tile, `floor` the smallest rank of the chain max, max-1, ... where each
/// rank is present exactly once, and `top_free` the largest rank below the floor (0 for
/// none). Within a stage the key walks the build cycle: the free tiles grow towards the
/// floor, merge into it, the floor rises and the free tiles start small again. Boards
/// below 16384 have no key; fresh play covers them.
pub fn pool_key(b: Board) -> Option<u16> {
    let mut count = [0u8; 16];
    for i in 0..16 {
        count[((b >> (4 * i)) & 0xF) as usize] += 1;
    }
    let max = (1..16).rev().find(|&r| count[r] > 0)?;
    if max < 14 {
        return None;
    }
    let mut floor = max;
    while floor > 1 && count[floor - 1] == 1 {
        floor -= 1;
    }
    let top_free = (1..floor).rev().find(|&r| count[r] > 0).unwrap_or(0);
    Some(((max as u16) << 10) | ((floor as u16) << 5) | top_free as u16)
}

/// (max, floor, top_free) ranks of a key.
pub fn key_parts(k: u16) -> (u8, u8, u8) {
    (((k >> 10) & 31) as u8, ((k >> 5) & 31) as u8, (k & 31) as u8)
}

/// The training stage a key's boards belong to (16384 -> 1, 32768 -> 2, 65536 -> 3).
pub fn key_stage(k: u16) -> usize {
    (key_parts(k).0 as usize).saturating_sub(13)
}

/// Boards seen so far for one key, kept as a uniform sample of everything offered.
#[derive(Default)]
struct Reservoir {
    boards: Vec<Board>,
    seen: u64,
}

impl Reservoir {
    fn add(&mut self, b: Board, cap: usize, rng: &mut Rng) {
        self.seen += 1;
        if self.boards.len() < cap {
            self.boards.push(b);
        } else {
            let j = (rng.next() % self.seen) as usize;
            if j < self.boards.len() {
                self.boards[j] = b;
            }
        }
    }
}

/// One key's boards as they travel: (key, boards offered so far, sample).
pub type PoolEntry = (u16, u64, Vec<Board>);

/// Boards to restart training games from, keyed by chain state, so endgames are
/// practised far more often than fresh games reach them. `add` also files each board in
/// an outbox, which a worker drains to send its harvest to the coordinator.
pub struct Pool {
    buckets: Mutex<HashMap<u16, Reservoir>>,
    outbox: Mutex<HashMap<u16, Reservoir>>,
    cap: AtomicUsize,
    dirty: AtomicUsize,
}

/// Boards per key a worker sends per chunk; the coordinator samples across chunks.
const OUTBOX_CAP: usize = 64;
const POOL_MAGIC: &[u8; 4] = b"POL1";

impl Pool {
    pub fn new(cap: usize) -> Self {
        Pool { buckets: Mutex::new(HashMap::new()), outbox: Mutex::new(HashMap::new()), cap: AtomicUsize::new(cap.max(1)), dirty: AtomicUsize::new(0) }
    }

    pub fn set_cap(&self, cap: usize) {
        self.cap.store(cap.max(1), Relaxed);
    }

    pub fn add(&self, key: u16, b: Board, rng: &mut Rng) {
        let cap = self.cap.load(Relaxed);
        self.buckets.lock().unwrap().entry(key).or_default().add(b, cap, rng);
        self.outbox.lock().unwrap().entry(key).or_default().add(b, OUTBOX_CAP, rng);
        self.dirty.fetch_add(1, Relaxed);
    }

    /// Files boards from another pool (the coordinator's sample, a worker's upload).
    pub fn merge(&self, entries: &[PoolEntry], rng: &mut Rng) {
        let cap = self.cap.load(Relaxed);
        let mut buckets = self.buckets.lock().unwrap();
        for (key, _, boards) in entries {
            let r = buckets.entry(*key).or_default();
            for &b in boards {
                r.add(b, cap, rng);
            }
        }
        self.dirty.fetch_add(entries.len(), Relaxed);
    }

    /// A board from a key at stage `min_stage` or above: the key is drawn uniformly
    /// among the non-empty ones, so a state fresh play hits once an hour gets the same
    /// share of restarts as one it hits every second.
    pub fn pick(&self, rng: &mut Rng, min_stage: usize) -> Option<Board> {
        let buckets = self.buckets.lock().unwrap();
        let keys: Vec<&Reservoir> = buckets.iter().filter(|(k, r)| key_stage(**k) >= min_stage && !r.boards.is_empty()).map(|(_, r)| r).collect();
        if keys.is_empty() {
            return None;
        }
        let r = keys[rng.below(keys.len() as u32) as usize];
        Some(r.boards[rng.below(r.boards.len() as u32) as usize])
    }

    /// `n` boards drawn like `pick`, grouped by key.
    pub fn sample(&self, n: usize, min_stage: usize, rng: &mut Rng) -> Vec<PoolEntry> {
        let buckets = self.buckets.lock().unwrap();
        let keys: Vec<(&u16, &Reservoir)> = buckets.iter().filter(|(k, r)| key_stage(**k) >= min_stage && !r.boards.is_empty()).collect();
        if keys.is_empty() {
            return vec![];
        }
        let mut out: HashMap<u16, Vec<Board>> = HashMap::new();
        for _ in 0..n {
            let (k, r) = keys[rng.below(keys.len() as u32) as usize];
            out.entry(*k).or_default().push(r.boards[rng.below(r.boards.len() as u32) as usize]);
        }
        let mut v: Vec<PoolEntry> = out.into_iter().map(|(k, b)| (k, 0, b)).collect();
        v.sort_by_key(|e| e.0);
        v
    }

    /// Everything filed since the last call.
    pub fn take_outbox(&self) -> Vec<PoolEntry> {
        let mut v: Vec<PoolEntry> = self.outbox.lock().unwrap().drain().map(|(k, r)| (k, r.seen, r.boards)).collect();
        v.sort_by_key(|e| e.0);
        v
    }

    fn export(&self) -> Vec<PoolEntry> {
        let mut v: Vec<PoolEntry> = self.buckets.lock().unwrap().iter().map(|(k, r)| (*k, r.seen, r.boards.clone())).collect();
        v.sort_by_key(|e| e.0);
        v
    }

    /// (key, boards held, boards offered) per key, largest keys first.
    pub fn summary(&self) -> Vec<(u16, usize, u64)> {
        let mut v: Vec<_> = self.buckets.lock().unwrap().iter().map(|(k, r)| (*k, r.boards.len(), r.seen)).collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v
    }

    pub fn total(&self) -> usize {
        self.buckets.lock().unwrap().values().map(|r| r.boards.len()).sum()
    }

    /// Adds since the last `save`.
    pub fn dirty(&self) -> bool {
        self.dirty.load(Relaxed) > 0
    }

    /// Wire and file format: magic, u8 bytes per board (8), u32 keys, then per key
    /// u16 key, u64 seen, u32 count, the boards.
    pub fn encode(entries: &[PoolEntry]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(POOL_MAGIC);
        out.push(8);
        out.extend((entries.len() as u32).to_le_bytes());
        for (k, seen, boards) in entries {
            out.extend(k.to_le_bytes());
            out.extend(seen.to_le_bytes());
            out.extend((boards.len() as u32).to_le_bytes());
            for b in boards {
                out.extend(b.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Vec<PoolEntry>> {
        let mut at = 0;
        let mut take = |n: usize| {
            let s = bytes.get(at..at + n)?;
            at += n;
            Some(s)
        };
        if take(4)? != POOL_MAGIC || take(1)? != [8] {
            return None;
        }
        // Counts come off the wire: never reserve more than the bytes could hold (a key
        // entry is at least 14 bytes, a board 8), so a corrupt body cannot abort the process.
        let nkeys = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let mut out = Vec::with_capacity(nkeys.min(bytes.len() / 14));
        for _ in 0..nkeys {
            let key = u16::from_le_bytes(take(2)?.try_into().ok()?);
            let seen = u64::from_le_bytes(take(8)?.try_into().ok()?);
            let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
            let mut boards = Vec::with_capacity(n.min(bytes.len() / 8));
            for _ in 0..n {
                boards.push(u64::from_le_bytes(take(8)?.try_into().ok()?));
            }
            out.push((key, seen, boards));
        }
        Some(out)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, Pool::encode(&self.export()))?;
        std::fs::rename(tmp, path)?;
        self.dirty.store(0, Relaxed);
        Ok(())
    }

    /// The pool saved at `path`, or an empty one if there is none or it is unreadable.
    pub fn load(path: &str, cap: usize) -> Self {
        let pool = Pool::new(cap);
        if let Some(entries) = std::fs::read(path).ok().and_then(|b| Pool::decode(&b)) {
            let mut buckets = pool.buckets.lock().unwrap();
            for (k, seen, boards) in entries {
                let mut r = Reservoir { boards, seen: 0 };
                r.boards.truncate(cap);
                r.seen = seen.max(r.boards.len() as u64);
                buckets.insert(k, r);
            }
        }
        pool
    }
}

pub struct EpisodeStats {
    pub score: u64,
    /// 16 when two 32768s merged: the board cannot hold a 65536, so the game ends there.
    pub max_rank: u8,
    /// False when the episode started from a restart-pool board.
    pub fresh: bool,
    /// Largest tile rank of the board the episode started from; 0 for a fresh game.
    pub start_rank: u8,
}

impl EpisodeStats {
    /// A restart episode that built a bigger tile than it started with.
    pub fn progressed(&self) -> bool {
        !self.fresh && self.max_rank > self.start_rank
    }
}

/// Plays one greedy (1-ply) game, learning from it with TD(0) on afterstates.
/// With probability `restart_p` it starts from a saved endgame board at stage
/// `restart_stage` or above instead of a new game. Boards holding 16384 or more are
/// filed in the pool the first time each chain state appears in the game.
pub fn train_episode(net: &NTuple, t: &Tables, rng: &mut Rng, alpha: f32, pool: &Pool, restart_p: f32, restart_stage: usize) -> EpisodeStats {
    let restart = if restart_p > 0.0 && (rng.below(1_000_000) as f32) < restart_p * 1e6 { pool.pick(rng, restart_stage) } else { None };
    let fresh = restart.is_none();
    let mut b = restart.unwrap_or_else(|| spawn(spawn(0, rng), rng));
    let start_rank = if fresh { 0 } else { max_rank(b) };
    let mut top = max_rank(b);
    let mut seen_keys: Vec<u16> = Vec::new();
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
                return EpisodeStats { score, max_rank: top, fresh, start_rank };
            }
            Some((after, r, v)) => {
                score += r as u64;
                if top == 15 && made_65536(b, after) {
                    // The 65536 does not fit a nibble, so the game ends here. The last
                    // transition is not learned: the afterstate before it keeps its
                    // bootstrapped value instead of one from a board the engine cannot hold.
                    return EpisodeStats { score, max_rank: 16, fresh, start_rank };
                }
                if let Some(p) = prev_after {
                    net.update(p, alpha, v - net.value(p));
                }
                prev_after = Some(after);
                b = spawn(after, rng);
                top = max_rank(b);
                if top >= 14 {
                    if let Some(k) = pool_key(b) {
                        if !seen_keys.contains(&k) {
                            seen_keys.push(k);
                            pool.add(k, b, rng);
                        }
                    }
                }
            }
        }
    }
}

/// Runs training episodes on `threads` threads until `games` have been played or
/// `deadline` passes, handing each finished episode to `each` with its game number.
pub fn train_parallel(
    net: &NTuple,
    pool: &Pool,
    alpha: f32,
    restart_p: f32,
    restart_stage: usize,
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
                    each(g, &train_episode(net, t, &mut rng, alpha, pool, restart_p, restart_stage));
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
        let pool = Pool::new(10);
        for _ in 0..50 {
            train_episode(&net, &t, &mut rng, 0.01, &pool, 0.0, 1);
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
        let pool = Pool::new(10);
        train_parallel(&a, &pool, 0.01, 0.0, 1, 3, 1, 20, None, &|_, _| {});
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

    fn grid(g: [[u8; 4]; 4]) -> Board {
        let mut b = 0;
        for r in 0..4 {
            for c in 0..4 {
                b |= (g[r][c] as u64) << (4 * (4 * r + c));
            }
        }
        b
    }

    #[test]
    fn stage_comes_from_the_raw_board() {
        // 32768, 16384, 8192, 2048 (4096 missing): downgraded it looks like a 16384 board,
        // but it is a 32768 board and uses stage 2's weights.
        let mut net = NTuple::new(0.0, 1, &TUPLES_4);
        net.expand_stages(3);
        let b = grid([[15, 14, 13, 11], [1, 2, 0, 0], [0; 4], [0; 4]]);
        let mut ix = [0; 8 * MAX_TUPLES];
        let n = net.indices(b, &mut ix);
        assert!(ix[..n].iter().all(|&i| i / net.stage_size() == 2));
        // ... with the downgraded pattern inside that stage.
        let mut ix2 = [0; 8 * MAX_TUPLES];
        net.indices(downgrade(b), &mut ix2);
        assert!(ix[..n].iter().zip(&ix2[..n]).all(|(a, b)| a - 2 * net.stage_size() == b - net.stage_size()));
    }

    #[test]
    fn frozen_stages_stay_out_of_the_diff() {
        let mut net = NTuple::new(0.0, 1, &TUPLES_4);
        net.expand_stages(2);
        net.set_frozen(1);
        let start = net.frozen_end();
        assert_eq!(start, net.stage_size());
        let mirror = net.snapshot_from(start);
        assert_eq!(mirror.len(), net.stage_size());
        // Updates on a 16384 board land in stage 1 and show up relative to `start`.
        let b = grid([[14, 13, 12, 1], [1, 2, 0, 0], [0; 4], [0; 4]]);
        net.update(b, 0.1, 1.0);
        let d = net.diff_from(&mirror, start);
        assert!(!d.is_empty() && d.iter().all(|&(i, _)| i as usize >= start));
        // Updates on a small board are refused entirely.
        let snap = net.snapshot();
        net.update(grid([[5, 4, 3, 1], [0; 4], [0; 4], [0; 4]]), 0.1, 1.0);
        assert!(net.diff(&snap).is_empty());
    }

    #[test]
    fn pool_keys_follow_the_chain() {
        // 32768, 16384, 8192 chain, free tiles up to 2048.
        let b = grid([[15, 14, 13, 11], [1, 2, 3, 0], [0; 4], [0; 4]]);
        assert_eq!(key_parts(pool_key(b).unwrap()), (15, 13, 11));
        assert_eq!(key_stage(pool_key(b).unwrap()), 2);
        // A duplicate ends the chain: two 8192s are free tiles.
        let b = grid([[15, 14, 13, 13], [1, 2, 3, 0], [0; 4], [0; 4]]);
        assert_eq!(key_parts(pool_key(b).unwrap()), (15, 14, 13));
        // Fresh 16384 with nothing else big.
        let b = grid([[14, 2, 1, 0], [1, 0, 0, 0], [0; 4], [0; 4]]);
        assert_eq!(key_parts(pool_key(b).unwrap()), (14, 14, 2));
        assert_eq!(key_stage(pool_key(b).unwrap()), 1);
        assert_eq!(pool_key(grid([[13, 12, 1, 0], [0; 4], [0; 4], [0; 4]])), None);
    }

    #[test]
    fn pool_caps_samples_and_filters_by_stage() {
        let pool = Pool::new(5);
        let mut rng = Rng(11);
        let k1 = pool_key(grid([[14, 1, 0, 0], [0; 4], [0; 4], [0; 4]])).unwrap();
        let k2 = pool_key(grid([[15, 1, 0, 0], [0; 4], [0; 4], [0; 4]])).unwrap();
        for i in 0..20u64 {
            pool.add(k1, 0xE000_0000_0000_0000 | i, &mut rng);
        }
        pool.add(k2, 0xF000_0000_0000_0001, &mut rng);
        assert_eq!(pool.summary().iter().find(|e| e.0 == k1).map(|e| (e.1, e.2)), Some((5, 20)));
        assert_eq!(pool.total(), 6);
        for _ in 0..50 {
            assert_eq!(pool.pick(&mut rng, 2), Some(0xF000_0000_0000_0001));
        }
        assert!(pool.pick(&mut rng, 3).is_none());
        let sample = pool.sample(100, 1, &mut rng);
        assert_eq!(sample.iter().map(|e| e.2.len()).sum::<usize>(), 100);
        assert!(sample.iter().any(|e| e.0 == k1) && sample.iter().any(|e| e.0 == k2));
        // The outbox holds everything offered (under its own cap) and empties when taken.
        let out = pool.take_outbox();
        assert_eq!(out.iter().map(|e| e.2.len()).sum::<usize>(), 21);
        assert!(pool.take_outbox().is_empty());
        // Wire round trip and file round trip.
        assert_eq!(Pool::decode(&Pool::encode(&out)).unwrap(), out);
        assert_eq!(Pool::decode(b"nope"), None);
        // A header claiming billions of keys or boards must fail, not reserve memory.
        let mut huge = b"POL1\x08".to_vec();
        huge.extend(u32::MAX.to_le_bytes());
        assert_eq!(Pool::decode(&huge), None);
        let mut huge = b"POL1\x08\x01\0\0\0".to_vec();
        huge.extend([0u8; 10]);
        huge.extend(u32::MAX.to_le_bytes());
        assert_eq!(Pool::decode(&huge), None);
        let path = std::env::temp_dir().join(format!("g2048-pool-test-{}.bin", std::process::id()));
        pool.save(path.to_str().unwrap()).unwrap();
        let loaded = Pool::load(path.to_str().unwrap(), 5);
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.summary(), pool.summary());
        let other = Pool::new(3);
        other.merge(&out, &mut rng);
        assert_eq!(other.summary().iter().find(|e| e.0 == k1).map(|e| e.1), Some(3));
    }

    #[test]
    fn episode_ends_when_65536_is_made() {
        // Only left and right are legal, and both merge the two 32768s.
        let net = NTuple::new(0.0, 1, &TUPLES_4);
        let t = Tables::new();
        let pool = Pool::new(10);
        let mut rng = Rng(3);
        let b = grid([[15, 15, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]);
        let k = pool_key(b).unwrap();
        pool.add(k, b, &mut rng);
        let e = train_episode(&net, &t, &mut rng, 0.01, &pool, 1.0, 2);
        assert!(!e.fresh);
        assert_eq!((e.max_rank, e.start_rank, e.score), (16, 15, 65536));
        assert!(e.progressed());
        // The restart stage filter keeps a stage-1 pool from feeding stage-2 restarts.
        let e = train_episode(&net, &t, &mut rng, 0.01, &pool, 1.0, 3);
        assert!(e.fresh);
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
