//! 4x4 2048 board packed into a u64: one 4-bit rank per cell (0 = empty, k = 2^k).
//! Cell (row r, col c) lives at nibble 4*r + c. Rules match play2048.co (see RULES.md).

pub type Board = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

pub const DIRS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

pub struct Tables {
    left: Vec<u16>,
    right: Vec<u16>,
    score: Vec<u32>, // points gained sliding this row (same for left and right after reversal)
    score_rev: Vec<u32>,
}

fn reverse_row(r: u16) -> u16 {
    (r >> 12) | ((r >> 4) & 0x00F0) | ((r << 4) & 0x0F00) | (r << 12)
}

/// Slide one row towards cell 0, merging each tile at most once.
pub fn slide_row(row: u16) -> (u16, u32) {
    let mut line = [0u8; 4];
    let mut n = 0;
    for i in 0..4 {
        let v = ((row >> (4 * i)) & 0xF) as u8;
        if v != 0 {
            line[n] = v;
            n += 1;
        }
    }
    let mut out = [0u8; 4];
    let (mut i, mut o, mut score) = (0, 0, 0u32);
    while i < n {
        if i + 1 < n && line[i] == line[i + 1] {
            // 65536 doesn't fit a nibble, so two 32768s merge into a 32768 worth 65536
            // points. Callers spot it as the count of 32768s dropping (see count_rank).
            out[o] = (line[i] + 1).min(15);
            score += 1 << (line[i] + 1);
            i += 2;
        } else {
            out[o] = line[i];
            i += 1;
        }
        o += 1;
    }
    let r = out.iter().enumerate().fold(0u16, |acc, (k, &v)| acc | (v as u16) << (4 * k));
    (r, score)
}

impl Tables {
    pub fn new() -> Self {
        let mut t = Tables {
            left: vec![0; 65536],
            right: vec![0; 65536],
            score: vec![0; 65536],
            score_rev: vec![0; 65536],
        };
        for row in 0..=u16::MAX {
            let (l, s) = slide_row(row);
            t.left[row as usize] = l;
            t.score[row as usize] = s;
            let rev = reverse_row(row);
            let (rl, rs) = slide_row(rev);
            t.right[row as usize] = reverse_row(rl);
            t.score_rev[row as usize] = rs;
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
        let mut out = 0u64;
        let mut score = 0u32;
        for r in 0..4 {
            let row = ((src >> (16 * r)) & 0xFFFF) as usize;
            out |= (tbl[row] as u64) << (16 * r);
            score += stbl[row];
        }
        (if horizontal { out } else { transpose(out) }, score)
    }
}

#[inline]
pub fn transpose(x: u64) -> u64 {
    let a1 = x & 0xF0F0_0F0F_F0F0_0F0F;
    let a2 = x & 0x0000_F0F0_0000_F0F0;
    let a3 = x & 0x0F0F_0000_0F0F_0000;
    let a = a1 | (a2 << 12) | (a3 >> 12);
    let b1 = a & 0xFF00_FF00_00FF_00FF;
    let b2 = a & 0x00FF_00FF_0000_0000;
    let b3 = a & 0x0000_0000_FF00_FF00;
    b1 | (b2 >> 24) | (b3 << 24)
}

#[inline]
pub fn count_empty(b: Board) -> u32 {
    let mut x = b;
    x |= (x >> 2) & 0x3333_3333_3333_3333;
    x |= x >> 1;
    x = !x & 0x1111_1111_1111_1111;
    x.count_ones()
}

pub fn max_rank(b: Board) -> u8 {
    (0..16).map(|i| ((b >> (4 * i)) & 0xF) as u8).max().unwrap()
}

pub fn count_rank(b: Board, rank: u64) -> u32 {
    (0..16).filter(|i| (b >> (4 * i)) & 0xF == rank).count() as u32
}

/// True if the move from `before` to `after` merged two 32768s, i.e. made 65536.
pub fn made_65536(before: Board, after: Board) -> bool {
    count_rank(after, 15) < count_rank(before, 15)
}

pub fn distinct_tiles(b: Board) -> u32 {
    let mut seen = 0u16;
    for i in 0..16 {
        seen |= 1 << ((b >> (4 * i)) & 0xF);
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
    let rank: u64 = if rng.below(10) == 0 { 2 } else { 1 };
    for i in 0..16 {
        if (b >> (4 * i)) & 0xF == 0 {
            if pick == 0 {
                return b | (rank << (4 * i));
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
            let k = (b >> (4 * (4 * r + c))) & 0xF;
            let v = if k == 0 { 0 } else { 1u32 << k };
            s += &format!("{:>6}", if v == 0 { ".".into() } else { v.to_string() });
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_grid(g: [[u8; 4]; 4]) -> Board {
        let mut b = 0;
        for r in 0..4 {
            for c in 0..4 {
                b |= (g[r][c] as u64) << (4 * (4 * r + c));
            }
        }
        b
    }

    #[test]
    fn row_merges_once() {
        // [2,2,2,2] -> [4,4,.,.], score 8
        assert_eq!(slide_row(0x1111), (0x0022, 8));
        // [4,.,2,2] -> [4,4,.,.] (merged 4 must not merge again)
        assert_eq!(slide_row(0x1102), (0x0022, 4));
        // [2,2,4,.] -> [4,4]
        assert_eq!(slide_row(0x0211), (0x0022, 4));
        // [32768,32768,.,.] -> one tile worth 65536 points
        let (r, sc) = slide_row(0x00FF);
        assert_eq!((r, sc), (0x000F, 65536));
        assert!(made_65536(0x00FF, r as u64));
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
    fn transpose_roundtrip() {
        let b = 0x0123_4567_89AB_CDEF;
        assert_eq!(transpose(transpose(b)), b);
        assert_eq!((transpose(b) >> 4) & 0xF, (b >> 16) & 0xF);
    }
}
