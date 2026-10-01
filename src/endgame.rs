//! Exact endgame lookups, after the "tuple moves" of macroxue/2048-ai.
//!
//! Late in a game the largest tiles sit still in one corner and the game is really played
//! on the few cells left. For a fixed shape of that corner (a `Layout`), the sub-game on
//! the free cells is small enough to solve outright: expectimax with memoisation over
//! every reachable position until the next chain tile stands where the chain continues
//! (the goal) or the position is lost, which depth-limited search can never see. The
//! anchors' real values never enter the table: they are replaced by a canonical descending
//! staircase, so one table serves a 32768 corner and a 2048 corner alike. Each entry holds
//! the best move and its probability of reaching the goal; tables fill lazily from the
//! positions that come up and are saved to disk between runs.

use crate::board::*;
use crate::ntuple::symmetries;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering::Relaxed};

/// A board as 16 ranks, row-major from the top-left.
type Grid = [u8; 16];

fn to_grid(b: Board) -> Grid {
    std::array::from_fn(|i| ((b >> (4 * i)) & 0xF) as u8)
}

fn from_grid(g: &Grid) -> Board {
    g.iter().enumerate().fold(0, |b, (i, &r)| b | (r as Board) << (4 * i))
}

/// The board under symmetry `s` (0 = identity, 1..3 rotations, 4..7 their mirrors).
pub fn transform(b: Board, s: usize) -> Board {
    let mut out = 0;
    for c in 0..16 {
        out |= ((b >> (4 * c)) & 0xF) << (4 * symmetries(c)[s]);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// The six largest tiles fill the top-left 3x2 block; the next chain tile is built
    /// below it (column 0) or beside it (column 3). macroxue's Tuple10.
    Block,
    /// The top row and the first three cells of the second row hold the chain; the next
    /// tile is built at the end of the second row. macroxue's Snake9.
    Snake,
    /// Five anchors: the top-left 3x2 block without its bottom-right cell, which is where
    /// the next tile is built. macroxue's Tuple11 (the small one).
    Five,
}

/// Which cells are anchors, how many values each free cell can take, and what counts as
/// a regular position and as the goal.
#[derive(Clone, Debug)]
pub struct Layout {
    kind: Kind,
    /// The largest anchor rank the table is exact for (macroxue's kMaxAnchorRank).
    top: u8,
    /// Per cell: 1 for an anchor, else the number of ranks the cell may hold.
    caps: [u8; 16],
}

impl Layout {
    pub fn block(top: u8) -> Layout {
        let mut caps = [top - 2; 16];
        for c in [0, 1, 2, 4, 5, 6] {
            caps[c] = 1;
        }
        caps[3] = top;
        caps[7] = top - 1;
        caps[8] = top;
        caps[9] = top - 1;
        Layout { kind: Kind::Block, top, caps }
    }

    pub fn five(top: u8) -> Layout {
        let mut caps = [top; 16];
        for c in [0, 1, 2, 4, 5] {
            caps[c] = 1;
        }
        caps[6] = top + 1;
        for c in 12..16 {
            caps[c] = top - 1;
        }
        Layout { kind: Kind::Five, top, caps }
    }

    pub fn snake(top: u8) -> Layout {
        let mut caps = [top; 16];
        for c in 0..7 {
            caps[c] = 1;
        }
        caps[7] = top + 1;
        for c in 12..16 {
            caps[c] = top - 1;
        }
        Layout { kind: Kind::Snake, top, caps }
    }

    /// `block10`, `five7`, `snake8`: the kind and its largest exact anchor rank. The lower
    /// bounds keep every spawned 4 inside the rules; the upper ones keep the canonical
    /// anchors inside a nibble.
    pub fn parse(name: &str) -> Option<Layout> {
        let (kind, top) = name.split_at(name.find(|c: char| c.is_ascii_digit())?);
        let top: u8 = top.parse().ok()?;
        match kind {
            "block" if (5..=10).contains(&top) => Some(Layout::block(top)),
            "five" if (4..=7).contains(&top) => Some(Layout::five(top)),
            "snake" if (4..=8).contains(&top) => Some(Layout::snake(top)),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        let kind = match self.kind {
            Kind::Block => "block",
            Kind::Five => "five",
            Kind::Snake => "snake",
        };
        format!("{kind}{}", self.top)
    }

    /// macroxue follows the five-anchor table only when it is nearly sure (0.9); the
    /// others whenever they see any way to the goal.
    pub fn threshold(&self) -> f32 {
        if self.kind == Kind::Five { 0.9 } else { 0.0 }
    }

    pub fn entries(&self) -> usize {
        self.caps.iter().map(|&c| c as usize).product()
    }

    fn index(&self, g: &Grid) -> usize {
        let mut v = 0;
        for c in 0..16 {
            if self.caps[c] > 1 {
                v = v * self.caps[c] as usize + g[c] as usize;
            }
        }
        v
    }

    /// Anchors as the table knows them: a descending staircase whose smallest step makes
    /// the anchor rank `top`.
    fn prefill(&self, g: &mut Grid) {
        let mut r = self.top
            + match self.kind {
                Kind::Block => 5,
                Kind::Snake => 7,
                Kind::Five => 8,
            };
        for c in 0..16 {
            if self.caps[c] == 1 {
                g[c] = r;
                r -= 1;
            }
        }
    }

    fn same_anchors(&self, a: &Grid, b: &Grid) -> bool {
        (0..16).all(|c| self.caps[c] > 1 || a[c] == b[c])
    }

    /// The anchor rank the free cells are measured against.
    fn anchor(&self, g: &Grid) -> i32 {
        let a = |r: usize, c: usize| g[4 * r + c] as i32;
        match self.kind {
            Kind::Block => a(1, 0).min(a(1, 1)).min(a(1, 2)),
            Kind::Snake => (self.top as i32).min(a(1, 2)).min(a(0, 3)),
            Kind::Five => (self.top as i32).min(a(1, 1)).min(a(0, 2)),
        }
    }

    /// Anchors in the layout's order and distinct wherever they touch, every free cell
    /// below the bound the layout gives it. Only regular positions are in the table, and
    /// only on regular boards is its advice sound. The rules are macroxue's exactly: the
    /// anchors that border free cells are all at least the anchor rank, so no free tile
    /// can ever merge into one; the top-left anchors touch nothing but anchors.
    pub fn regular(&self, g: &Grid) -> bool {
        let a = |r: usize, c: usize| g[4 * r + c] as i32;
        let k = self.anchor(g);
        match self.kind {
            Kind::Block => {
                if a(0, 0) == a(0, 1) || a(0, 1) == a(0, 2) || a(1, 0) == a(1, 1) || a(1, 1) == a(1, 2) {
                    return false;
                }
                if a(0, 0) == a(1, 0) || a(0, 1) == a(1, 1) || a(0, 2) <= a(1, 2) {
                    return false;
                }
                if k > self.top as i32 {
                    return false;
                }
                a(0, 3) <= k - 1
                    && a(1, 3) <= k - 2
                    && a(2, 0) <= k - 1
                    && a(2, 1) <= k - 2
                    && [a(2, 2), a(2, 3), a(3, 0), a(3, 1), a(3, 2), a(3, 3)].iter().all(|&x| x <= k - 3)
            }
            Kind::Five => {
                if a(0, 0) == a(0, 1) || a(0, 1) == a(0, 2) || a(1, 0) <= a(1, 1) || a(1, 1) <= a(1, 2) {
                    return false;
                }
                if a(0, 0) == a(1, 0) || a(0, 1) == a(1, 1) || a(0, 2) <= a(1, 2) || a(1, 1) == a(0, 2) {
                    return false;
                }
                if a(1, 2) > self.top as i32 {
                    return false;
                }
                a(0, 3) <= k - 1 && a(1, 3) <= k - 1 && (8..12).all(|c| g[c] as i32 <= k - 1) && (12..16).all(|c| g[c] as i32 <= k - 2)
            }
            Kind::Snake => {
                if a(0, 0) == a(0, 1) || a(0, 1) == a(0, 2) || a(0, 2) == a(0, 3) {
                    return false;
                }
                if a(1, 0) <= a(1, 1) || a(1, 1) <= a(1, 2) || a(1, 2) <= a(1, 3) {
                    return false;
                }
                if a(0, 0) == a(1, 0) || a(0, 1) == a(1, 1) || a(0, 2) == a(1, 2) || a(0, 3) <= a(1, 3) || a(0, 3) <= a(1, 0) {
                    return false;
                }
                (8..12).all(|c| g[c] as i32 <= k - 1) && (12..16).all(|c| g[c] as i32 <= k - 2)
            }
        }
    }

    /// The next chain tile stands where the chain continues, with the staircase under it
    /// that the following merges need.
    pub fn goal(&self, g: &Grid) -> bool {
        let a = |r: usize, c: usize| g[4 * r + c] as i32;
        match self.kind {
            Kind::Block => {
                let k = self.anchor(g);
                (a(2, 0) == k - 1
                    && a(2, 1) == k - 2
                    && ((a(2, 2) == k - 3 && (a(3, 2) == k - 3 || a(2, 3) == k - 3)) || (a(3, 1) == k - 3 && (a(3, 0) == k - 3 || a(3, 2) == k - 3))))
                    || (a(0, 3) == k - 1 && a(1, 3) == k - 2 && a(2, 3) == k - 3 && (a(2, 2) == k - 3 || a(3, 3) == k - 3))
            }
            Kind::Snake => {
                let k = (self.top as i32).min(a(1, 2).min(a(0, 3)) - 1);
                a(1, 3) == k && a(2, 3) == k - 1 && (a(2, 2) == k - 1 || (a(3, 3) == k - 2 && a(3, 2) == k - 2))
            }
            Kind::Five => {
                let k = (self.top as i32).min(a(1, 1).min(a(0, 2)) - 1);
                a(1, 2) == k
                    && ((a(1, 3) == k - 1 && (a(0, 3) == k - 1 || a(2, 3) == k - 1))
                        || (a(0, 3) == k - 1 && a(1, 3) == k - 2 && a(2, 3) == k - 2)
                        || (a(2, 2) == k - 1 && (a(2, 1) == k - 1 || a(3, 2) == k - 1 || a(2, 3) == k - 1)))
            }
        }
    }
}

/// Probabilities travel as 14 bits; 0 marks an entry not computed yet.
const PROB_ONE: u16 = 16000;
const MAGIC: &[u8; 8] = b"2048EG01";
/// Entries per saved block; only blocks with something in them go to disk.
const CHUNK: usize = 4096;

/// One layout's table of best move and goal probability per free-cell configuration.
pub struct Table {
    pub layout: Layout,
    data: Box<[AtomicU16]>,
    computed: AtomicU64,
}

impl Table {
    pub fn new(layout: Layout) -> Table {
        let n = layout.entries();
        // Zeroed pages are not touched until written, so an unused table costs nothing
        // but address space.
        let data: Box<[AtomicU16]> = unsafe {
            let bytes = std::alloc::alloc_zeroed(std::alloc::Layout::array::<AtomicU16>(n).expect("table size"));
            assert!(!bytes.is_null(), "out of memory for the {} table", layout.name());
            Box::from_raw(std::slice::from_raw_parts_mut(bytes as *mut AtomicU16, n))
        };
        Table { layout, data, computed: AtomicU64::new(0) }
    }

    pub fn computed(&self) -> u64 {
        self.computed.load(Relaxed)
    }

    fn get(&self, v: usize) -> Option<(usize, f32)> {
        let e = self.data[v].load(Relaxed);
        (e != 0).then(|| ((e >> 14) as usize, ((e & 0x3FFF) - 1) as f32 / PROB_ONE as f32))
    }

    fn set(&self, v: usize, mv: usize, p: f32) {
        let q = (p * PROB_ONE as f32).round() as u16;
        if self.data[v].swap(((mv as u16) << 14) | (q.min(PROB_ONE) + 1), Relaxed) == 0 {
            self.computed.fetch_add(1, Relaxed);
        }
    }

    /// Best move from a regular, non-goal position with canonical anchors, solving every
    /// position reachable from it on the way.
    fn try_moves(&self, t: &Tables, b: Board) -> f32 {
        let g = to_grid(b);
        // A spawn can only break the rules in a layout whose bounds are below 4; such a
        // position is lost and must not be indexed.
        if !self.layout.regular(&g) {
            return 0.0;
        }
        let v = self.layout.index(&g);
        if let Some((_, p)) = self.get(v) {
            return p;
        }
        let (mut best_move, mut best) = (0, 0.0f32);
        for (m, d) in DIRS.iter().enumerate() {
            let (nb, _) = t.apply(b, *d);
            if nb == b {
                continue;
            }
            let ng = to_grid(nb);
            if !self.layout.regular(&ng) || !self.layout.same_anchors(&g, &ng) {
                continue;
            }
            let p = if self.layout.goal(&ng) { 1.0 } else { self.try_tiles(t, nb) };
            if p > best {
                (best_move, best) = (m, p);
            }
        }
        self.set(v, best_move, best);
        best
    }

    fn try_tiles(&self, t: &Tables, b: Board) -> f32 {
        let (mut p, mut n) = (0.0, 0);
        for i in 0..16 {
            if (b >> (4 * i)) & 0xF == 0 {
                n += 1;
                p += 0.9 * self.try_moves(t, b | 1 << (4 * i)) + 0.1 * self.try_moves(t, b | 2 << (4 * i));
            }
        }
        if n == 0 { 0.0 } else { p / n as f32 }
    }

    /// The table's advice for a board already in the layout's orientation: (move index
    /// into DIRS, probability), or None when the board is not regular or is at the goal.
    fn advice(&self, t: &Tables, b: Board) -> Option<(usize, f32)> {
        let g = to_grid(b);
        if !self.layout.regular(&g) || self.layout.goal(&g) {
            return None;
        }
        let v = self.layout.index(&g);
        if let Some(e) = self.get(v) {
            return Some(e);
        }
        let mut canon = g;
        self.layout.prefill(&mut canon);
        debug_assert!(self.layout.regular(&canon) && !self.layout.goal(&canon));
        let start = std::time::Instant::now();
        let before = self.computed();
        self.try_moves(t, from_grid(&canon));
        let secs = start.elapsed().as_secs_f64();
        if secs > 2.0 {
            eprintln!("{}: solved {} positions in {secs:.0}s", self.layout.name(), self.computed() - before);
        }
        self.get(v)
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let tmp = path.with_extension("tmp");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        let name = self.layout.name();
        f.write_all(&[name.len() as u8])?;
        f.write_all(name.as_bytes())?;
        f.write_all(&(self.data.len() as u64).to_le_bytes())?;
        let chunks = self.data.chunks(CHUNK);
        let mut bitmap = vec![0u8; chunks.len().div_ceil(8)];
        for (i, c) in self.data.chunks(CHUNK).enumerate() {
            if c.iter().any(|e| e.load(Relaxed) != 0) {
                bitmap[i / 8] |= 1 << (i % 8);
            }
        }
        f.write_all(&bitmap)?;
        for (i, c) in self.data.chunks(CHUNK).enumerate() {
            if bitmap[i / 8] & (1 << (i % 8)) != 0 {
                let bytes: Vec<u8> = c.iter().flat_map(|e| e.load(Relaxed).to_le_bytes()).collect();
                f.write_all(&bytes)?;
            }
        }
        f.flush()?;
        drop(f);
        std::fs::rename(tmp, path)
    }

    /// Fills the table from a file written by `save`; a missing file is an empty table.
    pub fn load(&self, path: &std::path::Path) -> std::io::Result<()> {
        let Ok(file) = std::fs::File::open(path) else { return Ok(()) };
        let mut f = std::io::BufReader::new(file);
        let bad = |what: &str| std::io::Error::other(format!("{}: {what}", path.display()));
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not an endgame table"));
        }
        let mut len = [0u8; 1];
        f.read_exact(&mut len)?;
        let mut name = vec![0u8; len[0] as usize];
        f.read_exact(&mut name)?;
        let mut n = [0u8; 8];
        f.read_exact(&mut n)?;
        if name != self.layout.name().as_bytes() || u64::from_le_bytes(n) as usize != self.data.len() {
            return Err(bad("table for another layout"));
        }
        let nchunks = self.data.len().div_ceil(CHUNK);
        let mut bitmap = vec![0u8; nchunks.div_ceil(8)];
        f.read_exact(&mut bitmap)?;
        let mut buf = vec![0u8; 2 * CHUNK];
        for (i, c) in self.data.chunks(CHUNK).enumerate() {
            if bitmap[i / 8] & (1 << (i % 8)) == 0 {
                continue;
            }
            f.read_exact(&mut buf[..2 * c.len()])?;
            for (e, b) in c.iter().zip(buf.chunks(2)) {
                let v = u16::from_le_bytes([b[0], b[1]]);
                if v != 0 && e.swap(v, Relaxed) == 0 {
                    self.computed.fetch_add(1, Relaxed);
                }
            }
        }
        Ok(())
    }
}

/// The tables in use, consulted in every orientation of the board.
pub struct Lookup {
    tables: Vec<Table>,
    t: Tables,
    dir: std::path::PathBuf,
    /// Moves answered from a table, and boards asked about.
    pub hits: AtomicU64,
    pub asked: AtomicU64,
}

impl Lookup {
    /// Tables for `layouts`, loaded from `dir` where a file exists.
    pub fn open(dir: &std::path::Path, layouts: &[Layout]) -> std::io::Result<Lookup> {
        std::fs::create_dir_all(dir)?;
        let mut tables = Vec::new();
        for l in layouts {
            let table = Table::new(l.clone());
            table.load(&dir.join(format!("{}.bin", l.name())))?;
            eprintln!("endgame table {}: {} of {} positions known", l.name(), table.computed(), l.entries());
            tables.push(table);
        }
        Ok(Lookup { tables, t: Tables::new(), dir: dir.to_path_buf(), hits: AtomicU64::new(0), asked: AtomicU64::new(0) })
    }

    /// Writes every table that learnt something back to its file.
    pub fn save(&self) -> std::io::Result<()> {
        for table in &self.tables {
            if table.computed() > 0 {
                table.save(&self.dir.join(format!("{}.bin", table.layout.name())))?;
            }
        }
        Ok(())
    }

    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    /// Whether some table applies to `b` in some orientation (no solving).
    pub fn applies(&self, b: Board) -> Option<String> {
        for table in &self.tables {
            for s in 0..8 {
                let g = to_grid(transform(b, s));
                if table.layout.regular(&g) && !table.layout.goal(&g) {
                    return Some(table.layout.name());
                }
            }
        }
        None
    }

    /// The move with the best goal probability over all tables and orientations, if any
    /// table applies and the probability is above `min`.
    pub fn suggest(&self, b: Board, min: f32) -> Option<(Dir, f32)> {
        self.asked.fetch_add(1, Relaxed);
        let mut best: Option<(Dir, f32)> = None;
        for table in &self.tables {
            for s in 0..8 {
                let tb = transform(b, s);
                let Some((m, p)) = table.advice(&self.t, tb) else { continue };
                if p <= min.max(table.layout.threshold()) || best.is_some_and(|(_, bp)| p <= bp) {
                    continue;
                }
                // Back to the board's own orientation: the direction whose result is the
                // advised move's result seen through the same symmetry.
                let want = self.t.apply(tb, DIRS[m]).0;
                if let Some(d) = DIRS.iter().copied().find(|&d| transform(self.t.apply(b, d).0, s) == want) {
                    best = Some((d, p));
                }
            }
        }
        if best.is_some() {
            self.hits.fetch_add(1, Relaxed);
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(g: [[u8; 4]; 4]) -> Board {
        from_grid(&std::array::from_fn(|i| g[i / 4][i % 4]))
    }

    #[test]
    fn layouts_match_macroxue_sizes() {
        assert_eq!(Layout::block(10).entries(), 2_123_366_400); // Tuple10
        assert_eq!(Layout::snake(8).entries(), 9 * 8 * 8 * 8 * 8 * 7 * 7 * 7 * 7); // Snake9
        assert_eq!(Layout::five(7).entries(), 7 * 8 * 7 * 7 * 7 * 7 * 7 * 6 * 6 * 6 * 6); // Tuple11 (small)
        assert_eq!(Layout::parse("five7").map(|l| l.caps[6]), Some(8));
        assert!(Layout::parse("block11").is_none() && Layout::parse("five8").is_none());
        assert_eq!(Layout::parse("block10").map(|l| l.name()), Some("block10".into()));
        assert_eq!(Layout::parse("snake8").map(|l| l.caps[7]), Some(9));
        assert!(Layout::parse("tower5").is_none());
    }

    #[test]
    fn canonical_anchors_are_regular_and_in_caps() {
        for l in [Layout::block(10), Layout::block(6), Layout::snake(8), Layout::snake(5), Layout::five(7), Layout::five(4)] {
            let mut g = [0u8; 16];
            l.prefill(&mut g);
            assert!(l.regular(&g) && !l.goal(&g), "{}", l.name());
            // The largest free values the rules allow fit the caps.
            let k = l.anchor(&g);
            assert_eq!(k, l.top as i32);
            for c in 0..16 {
                if l.caps[c] > 1 {
                    g[c] = l.caps[c] - 1;
                }
            }
            assert!(l.index(&g) < l.entries());
        }
    }

    #[test]
    fn block_regular_and_goal_follow_the_rules() {
        let l = Layout::block(10);
        // 32768 16384 8192 / 4096 2048 1024, small tiles below: regular, anchor 1024.
        let b = to_grid(grid([[15, 14, 13, 2], [12, 11, 10, 1], [5, 4, 3, 1], [2, 1, 0, 0]]));
        assert!(l.regular(&b) && !l.goal(&b));
        // 512 under the block with 256, 128, 128: the goal.
        let g = to_grid(grid([[15, 14, 13, 2], [12, 11, 10, 1], [9, 8, 7, 1], [2, 1, 7, 0]]));
        assert!(l.regular(&g) && l.goal(&g));
        // A free tile as large as the anchor allows breaks regularity.
        let bad = to_grid(grid([[15, 14, 13, 2], [12, 11, 10, 1], [10, 4, 3, 1], [2, 1, 0, 0]]));
        assert!(!l.regular(&bad));
        // Anchors at 2048 (anchor rank 11) are beyond what the table is exact for.
        let big = to_grid(grid([[15, 14, 13, 2], [12, 11, 11, 1], [5, 4, 3, 1], [2, 1, 0, 0]]));
        assert!(!l.regular(&big));
        let f = Layout::five(7);
        // 32768 16384 8192 / 4096 2048 and small tiles: the sixth block tile is wanted.
        let fv = to_grid(grid([[15, 14, 13, 2], [12, 11, 5, 1], [4, 3, 2, 1], [1, 2, 0, 0]]));
        assert!(f.regular(&fv) && !f.goal(&fv));
        // 128 built at the block's corner with 64 and 32,32 beside it: the goal.
        let fg = to_grid(grid([[15, 14, 13, 2], [12, 11, 7, 6], [4, 3, 2, 6], [1, 2, 0, 5]]));
        assert!(f.regular(&fg) && f.goal(&fg));
        let s = Layout::snake(8);
        let sn = to_grid(grid([[15, 14, 13, 12], [11, 10, 9, 3], [5, 4, 3, 1], [2, 1, 0, 0]]));
        assert!(s.regular(&sn) && !s.goal(&sn));
        let sg = to_grid(grid([[15, 14, 13, 12], [11, 10, 9, 8], [5, 4, 7, 7], [2, 1, 0, 0]]));
        assert!(s.regular(&sg) && s.goal(&sg));
    }

    #[test]
    fn tiny_table_solves_and_advises_in_every_orientation() {
        // block5: anchors 10..5, most free cells hold at most a 4: 291,600 positions.
        let dir = std::env::temp_dir().join(format!("g2048-eg-test-{}", std::process::id()));
        let lk = Lookup::open(&dir, &[Layout::block(5)]).unwrap();
        // A board whose only free tile cannot move without dragging the anchors is lost:
        // the table says so with probability 0, and the search keeps the move.
        let stuck = grid([[10, 9, 8, 0], [7, 6, 5, 0], [1, 0, 0, 0], [0, 0, 0, 0]]);
        assert_eq!(lk.suggest(stuck, 0.0), None);
        let b = grid([[10, 9, 8, 1], [7, 6, 5, 0], [1, 0, 2, 0], [0, 1, 0, 0]]);
        let (d, p) = lk.suggest(b, 0.0).expect("advice");
        assert!(p > 0.0 && p <= 1.0, "p = {p}");
        let table = &lk.tables()[0];
        assert!(table.computed() > 100);
        // Every orientation of the board gets the same probability and the matching move.
        for s in 1..8 {
            let tb = transform(b, s);
            let (td, tp) = lk.suggest(tb, 0.0).expect("advice");
            assert!((tp - p).abs() < 1e-4);
            assert_eq!(transform(lk.t.apply(b, d).0, s), lk.t.apply(tb, td).0);
        }
        assert_eq!(lk.applies(b), Some("block5".into()));
        assert_eq!(lk.applies(grid([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])), None);
        // Save, reload, same answer without solving again.
        lk.save().unwrap();
        let again = Lookup::open(&dir, &[Layout::block(5)]).unwrap();
        assert_eq!(again.tables()[0].computed(), table.computed());
        assert_eq!(again.suggest(b, 0.0), Some((d, p)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transform_is_a_symmetry() {
        let b = grid([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]);
        assert_eq!(transform(b, 0), b);
        assert_eq!(transform(b, 7), transpose(b));
        for s in 0..8 {
            assert_eq!(count_empty(transform(b, s)), 1);
        }
    }
}
