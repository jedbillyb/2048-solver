//! 4x4 2048 board packed into a u128: one 5-bit rank per cell (0 = empty, k = 2^k).
//! Cell (row r, col c) lives at bit 5 * (4*r + c); row r occupies bits 20*r .. 20*r+20.
//! Five bits hold ranks 0..31, so 65536 (rank 16) and 131072 (rank 17) are ordinary
//! tiles and the game runs past them. Rules match play2048.co (see RULES.md).

pub type Board = u128;

/// Bit at the low end of every cell (positions 0, 5, 10, ... 75). Used for empty counts.
const fn cells_low() -> u128 {
    let (mut m, mut k) = (0u128, 0);
    while k < 16 {
        m |= 1u128 << (5 * k);
        k += 1;
    }
    m
}
const CELLS_LOW: u128 = cells_low();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

pub const DIRS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

pub struct Tables {
    left: Vec<u32>,
    right: Vec<u32>,
    score: Vec<u32>, // points gained sliding this row (same for left and right after reversal)
    score_rev: Vec<u32>,
    empties: Vec<u8>, // zero cells in a row, summed over the four rows for count_empty
}

/// A 20-bit row holds four 5-bit cells. Reverse their order (cell 0 <-> 3, 1 <-> 2).
fn reverse_row(r: u32) -> u32 {
    ((r & 0x1F) << 15) | ((r & 0x3E0) << 5) | ((r & 0x7C00) >> 5) | ((r & 0xF_8000) >> 15)
}

/// Slide one 20-bit row towards cell 0, merging each tile at most once.
pub fn slide_row(row: u32) -> (u32, u32) {
    let mut line = [0u8; 4];
    let mut n = 0;
    for i in 0..4 {
        let v = ((row >> (5 * i)) & 0x1F) as u8;
        if v != 0 {
            line[n] = v;
            n += 1;
        }
    }
    let mut out = [0u8; 4];
    let (mut i, mut o, mut score) = (0, 0, 0u32);
    while i < n {
        if i + 1 < n && line[i] == line[i + 1] {
            // Two equal tiles merge into one of the next rank. 65536 and 131072 fit a
            // 5-bit cell, so unlike the old u64 board there is nothing special here.
            out[o] = line[i] + 1;
            score += 1 << (line[i] + 1);
            i += 2;
        } else {
            out[o] = line[i];
            i += 1;
        }
        o += 1;
    }
    let r = out.iter().enumerate().fold(0u32, |acc, (k, &v)| acc | (v as u32) << (5 * k));
    (r, score)
}

fn row_empties(row: u32) -> u8 {
    (0..4).filter(|i| (row >> (5 * i)) & 0x1F == 0).count() as u8
}

impl Tables {
    pub fn new() -> Self {
        let n = 1usize << 20;
        let mut t = Tables {
            left: vec![0; n],
            right: vec![0; n],
            score: vec![0; n],
            score_rev: vec![0; n],
            empties: vec![0; n],
        };
        for row in 0..n as u32 {
            let (l, s) = slide_row(row);
            t.left[row as usize] = l;
            t.score[row as usize] = s;
            let rev = reverse_row(row);
            let (rl, rs) = slide_row(rev);
            t.right[row as usize] = reverse_row(rl);
            t.score_rev[row as usize] = rs;
            t.empties[row as usize] = row_empties(row);
        }
        t
    }

    /// Returns (new board, points gained). Board is unchanged if the move is illegal.
    #[inline]
    pub fn apply(&self, b: Board, d: Dir) -> (Board, u32) {
        let (src, horizontal) = match d {
            Dir::Left | Dir::Right => (b, true),
            Dir::Up | Dir::Down => (transpose(b), false),
        };
        let (tbl, stbl) = match d {
            Dir::Left | Dir::Up => (&self.left, &self.score),
            Dir::Right | Dir::Down => (&self.right, &self.score_rev),
        };
        let mut out = 0u128;
        let mut score = 0u32;
        for r in 0..4 {
            let row = ((src >> (20 * r)) & 0xF_FFFF) as usize;
            out |= (tbl[row] as u128) << (20 * r);
            score += stbl[row];
        }
        (if horizontal { out } else { transpose(out) }, score)
    }
}

/// Swap the two 5-bit cells at bit `lo` and bit `lo + d`.
#[inline]
fn swap5(x: u128, lo: u32, d: u32) -> u128 {
    let t = ((x >> lo) ^ (x >> (lo + d))) & 0x1F;
    x ^ ((t << lo) | (t << (lo + d)))
}

/// Transpose the 4x4 board: swap each off-diagonal cell (r,c) with (c,r). The old u64
/// nibble transpose used magic constants that do not generalise to 5-bit cells, so this
/// is six pairwise field swaps instead (the diagonal cells 0, 5, 10, 15 stay put).
#[inline]
pub fn transpose(x: u128) -> u128 {
    let x = swap5(x, 5, 15); // (0,1) <-> (1,0)
    let x = swap5(x, 10, 30); // (0,2) <-> (2,0)
    let x = swap5(x, 15, 45); // (0,3) <-> (3,0)
    let x = swap5(x, 30, 15); // (1,2) <-> (2,1)
    let x = swap5(x, 35, 30); // (1,3) <-> (3,1)
    swap5(x, 55, 15) // (2,3) <-> (3,2)
}

#[inline]
pub fn count_empty(b: Board) -> u32 {
    // Each cell's low bit becomes the OR of its five bits, then count the nonzero cells.
    let nz = (b | (b >> 1) | (b >> 2) | (b >> 3) | (b >> 4)) & CELLS_LOW;
    16 - nz.count_ones()
}

pub fn max_rank(b: Board) -> u8 {
    (0..16).map(|i| ((b >> (5 * i)) & 0x1F) as u8).max().unwrap()
}

pub fn count_rank(b: Board, rank: u32) -> u32 {
    (0..16).filter(|i| ((b >> (5 * i)) & 0x1F) as u32 == rank).count() as u32
}

/// True if the move from `before` to `after` merged two 32768s, i.e. made a 65536.
pub fn made_65536(before: Board, after: Board) -> bool {
    count_rank(after, 15) < count_rank(before, 15)
}

/// True if the move made a 131072 (rank 17), the win: two 65536s merged.
pub fn made_131072(before: Board, after: Board) -> bool {
    count_rank(after, 16) < count_rank(before, 16)
}

pub fn distinct_tiles(b: Board) -> u32 {
    let mut seen = 0u32;
    for i in 0..16 {
        seen |= 1 << ((b >> (5 * i)) & 0x1F);
    }
    (seen >> 1).count_ones()
}

/// xorshift64* - fast, good enough for self-play.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next() >> 32) * n as u64 >> 32) as u32
    }
}

/// Site rule: uniform empty cell, 90% a 2, 10% a 4.
pub fn spawn(b: Board, rng: &mut Rng) -> Board {
    let empty = count_empty(b);
    if empty == 0 {
        return b;
    }
    let mut pick = rng.below(empty);
    let rank: u128 = if rng.below(10) == 0 { 2 } else { 1 };
    for i in 0..16 {
        if (b >> (5 * i)) & 0x1F == 0 {
            if pick == 0 {
                return b | (rank << (5 * i));
            }
            pick -= 1;
        }
    }
    unreachable!()
}

pub fn print(b: Board) -> String {
    let mut s = String::new();
    for r in 0..4 {
        for c in 0..4 {
            let k = (b >> (5 * (4 * r + c))) & 0x1F;
            let v = if k == 0 { 0 } else { 1u32 << k };
            s += &format!("{:>7}", if v == 0 { ".".into() } else { v.to_string() });
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cells: [u8; 4]) -> u32 {
        cells.iter().enumerate().fold(0u32, |acc, (k, &v)| acc | (v as u32) << (5 * k))
    }

    fn from_grid(g: [[u8; 4]; 4]) -> Board {
        let mut b = 0u128;
        for r in 0..4 {
            for c in 0..4 {
                b |= (g[r][c] as u128) << (5 * (4 * r + c));
            }
        }
        b
    }

    #[test]
    fn row_merges_once() {
        // [2,2,2,2] -> [4,4,.,.], score 8
        assert_eq!(slide_row(row([1, 1, 1, 1])), (row([2, 2, 0, 0]), 8));
        // [4,.,2,2] -> [4,4,.,.] (merged 4 must not merge again)
        assert_eq!(slide_row(row([2, 0, 1, 1])), (row([2, 2, 0, 0]), 4));
        // [2,2,4,.] -> [4,4]
        assert_eq!(slide_row(row([1, 1, 2, 0])), (row([2, 2, 0, 0]), 4));
        // [32768,32768,.,.] -> one 65536 (rank 16), worth 65536 points, and it stays
        let (r, sc) = slide_row(row([15, 15, 0, 0]));
        assert_eq!((r, sc), (row([16, 0, 0, 0]), 65536));
        // [65536,65536,.,.] -> one 131072 (rank 17), worth 131072 points
        let (r2, sc2) = slide_row(row([16, 16, 0, 0]));
        assert_eq!((r2, sc2), (row([17, 0, 0, 0]), 131072));
    }

    #[test]
    fn directions() {
        let t = Tables::new();
        let b = from_grid([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]);
        assert_eq!(t.apply(b, Dir::Left), (from_grid([[2, 0, 0, 0], [0; 4], [0; 4], [1, 0, 0, 0]]), 4));
        assert_eq!(t.apply(b, Dir::Right), (from_grid([[0, 0, 0, 2], [0; 4], [0; 4], [0, 0, 0, 1]]), 4));
        assert_eq!(t.apply(b, Dir::Up), (from_grid([[2, 0, 0, 1], [0; 4], [0; 4], [0; 4]]), 4));
        assert_eq!(t.apply(b, Dir::Down), (from_grid([[0; 4], [0; 4], [0; 4], [2, 0, 0, 1]]), 4));
    }

    #[test]
    fn made_tiles() {
        let before = from_grid([[15, 15, 0, 0], [0; 4], [0; 4], [0; 4]]);
        let after = from_grid([[16, 0, 0, 0], [0; 4], [0; 4], [0; 4]]);
        assert!(made_65536(before, after));
        assert!(!made_131072(before, after));
        let b2 = from_grid([[16, 16, 0, 0], [0; 4], [0; 4], [0; 4]]);
        let a2 = from_grid([[17, 0, 0, 0], [0; 4], [0; 4], [0; 4]]);
        assert!(made_131072(b2, a2));
    }

    #[test]
    fn empty_count() {
        assert_eq!(count_empty(0), 16);
        let b = from_grid([[1, 2, 0, 16], [0; 4], [17, 0, 0, 0], [0, 0, 0, 3]]);
        assert_eq!(count_empty(b), 11);
    }

    #[test]
    fn transpose_roundtrip() {
        let b = from_grid([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
        assert_eq!(transpose(transpose(b)), b);
        // cell (0,1) after transpose equals cell (1,0) before
        assert_eq!((transpose(b) >> 5) & 0x1F, (b >> 20) & 0x1F);
    }

    #[test]
    fn transpose_matches_reference() {
        // Against a plain per-cell transpose, over a few pseudo-random boards.
        let mut s = 0x1234_5678_9ABC_DEF0u128;
        for _ in 0..200 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = s & ((1u128 << 80) - 1);
            let mut want = 0u128;
            for r in 0..4u32 {
                for c in 0..4u32 {
                    let cell = (b >> (5 * (4 * r + c))) & 0x1F;
                    want |= cell << (5 * (4 * c + r));
                }
            }
            assert_eq!(transpose(b), want, "board {b:#x}");
        }
    }
}
