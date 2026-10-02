//! Expectimax search. Leaves are scored either by a hand-tuned row heuristic or by a
//! trained n-tuple network (which values afterstates in points, so rewards are added).

use crate::board::*;
use crate::endgame::Lookup;
use crate::ntuple::NTuple;
use std::collections::HashMap;
use std::sync::Arc;

const CPROB_THRESH: f32 = 0.0001;
const CACHE_DEPTH_LIMIT: u32 = 15;

const LOST_PENALTY: f32 = 200000.0;
const MONO_POW: f32 = 4.0;
const MONO_W: f32 = 47.0;
const SUM_POW: f32 = 3.5;
const SUM_W: f32 = 11.0;
const MERGES_W: f32 = 700.0;
const EMPTY_W: f32 = 270.0;

/// macroxue/2048-ai's line score: T(r) = r * 2^r per tile, a bonus for each pair that
/// descends left to right and a steep penalty for each that ascends, so big tiles are
/// pushed to the top-left corner. Rows and columns are summed in that one orientation.
fn mx_line(row: u32) -> f32 {
    let ts = |c: usize| {
        let r = ((row >> (5 * c)) & 0x1F) as i64;
        (r << r) as f32
    };
    let mut score = ts(0);
    for c in 0..3 {
        let (a, b) = (ts(c), ts(c + 1));
        score += if a >= b { a + b } else { (a - b) * 12.0 };
        if a == b {
            score += a;
        }
    }
    score
}

/// How leaves are scored when no net is loaded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Heuristic {
    /// The row heuristic below (monotonicity, merges, empties), symmetric in orientation.
    Rows,
    /// macroxue's corner-seeking line score, with a large negative value for a lost board.
    Macroxue,
}

pub struct Ai {
    t: Tables,
    heur: Vec<f32>,
    heuristic: Heuristic,
    net: Option<Arc<NTuple>>,
    /// Fixed search depth in chance layers; None = heuristic's adaptive depth.
    depth: Option<u32>,
    /// Deeper search once a 16384 is on the board, where the final merge chain is won or lost.
    endgame_depth: Option<u32>,
    /// Branches less likely than this are cut off and scored by the evaluator directly.
    cprob_thresh: f32,
    /// Added to the net's value on boards scored by its last stage (an experiment in
    /// lining up a retrained top stage with the frozen stage below it).
    top_bias: f32,
    /// Exact endgame tables, asked first on boards whose largest tile is at least
    /// `lookup_rank`; their move is taken when its goal probability is above `lookup_min`.
    lookup: Option<Arc<Lookup>>,
    lookup_rank: u8,
    lookup_min: Option<f32>,
    /// Hand the endgame to the heuristic: from boards whose largest tile is at least
    /// `endgame_rank`, leaves are scored by `heuristic` instead of the net, moves carry
    /// no reward, a loss is macroxue's negative score, the depth is `endgame_depth`
    /// (3 unless given) and spawns below 1 / 2^(depth + 4) are cut, as macroxue plays.
    endgame_eval: bool,
    endgame_rank: u8,
    /// macroxue's `pass_score` cut after the handover: chance nodes scoring below twice
    /// the largest tile's cost are not expanded. `big` is its BIG_TUPLES variant.
    pass_score: Option<PassScore>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PassScore {
    Small,
    Big,
}

impl std::str::FromStr for PassScore {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "small" => Ok(PassScore::Small),
            "big" => Ok(PassScore::Big),
            _ => Err(()),
        }
    }
}

/// What this search uses at its leaves.
#[derive(Clone, Copy)]
struct Mode {
    /// Score with the heuristic table even though a net is loaded.
    heuristic: bool,
    cprob_thresh: f32,
    /// Chance nodes whose static value is below this are scored without expanding them.
    pass: f32,
    /// Overrides the value of a lost board (macroxue's retry without the cut).
    dead: Option<f32>,
}

struct Search<'a> {
    ai: &'a Ai,
    depth_limit: u32,
    mode: Mode,
    tt: HashMap<Board, (u32, f32)>,
}

fn row_heur(row: u32) -> f32 {
    let line: [u32; 4] = std::array::from_fn(|i| (row >> (5 * i)) & 0x1F);
    let (mut sum, mut empty, mut merges) = (0.0, 0.0, 0.0);
    let (mut prev, mut counter) = (0, 0);
    for &r in &line {
        sum += (r as f32).powf(SUM_POW);
        if r == 0 {
            empty += 1.0;
        } else {
            if prev == r {
                counter += 1;
            } else if counter > 0 {
                merges += 1.0 + counter as f32;
                counter = 0;
            }
            prev = r;
        }
    }
    if counter > 0 {
        merges += 1.0 + counter as f32;
    }
    let (mut mono_l, mut mono_r) = (0.0, 0.0);
    for i in 1..4 {
        let (a, b) = ((line[i - 1] as f32).powf(MONO_POW), (line[i] as f32).powf(MONO_POW));
        if line[i - 1] > line[i] {
            mono_l += a - b;
        } else {
            mono_r += b - a;
        }
    }
    LOST_PENALTY + EMPTY_W * empty + MERGES_W * merges - MONO_W * f32::min(mono_l, mono_r) - SUM_W * sum
}

impl Ai {
    pub fn new() -> Self {
        Ai { t: Tables::new(), heur: (0..(1u32 << 20)).map(row_heur).collect(), heuristic: Heuristic::Rows, net: None, depth: None, endgame_depth: None, cprob_thresh: CPROB_THRESH, top_bias: 0.0, lookup: None, lookup_rank: 15, lookup_min: None, endgame_eval: false, endgame_rank: 15, pass_score: None }
    }

    /// From boards holding a tile of `rank`, play like the heuristic player `h` (with the
    /// tables if any) instead of the net.
    pub fn with_endgame_eval(mut self, h: Option<Heuristic>, rank: Option<u8>) -> Self {
        if let Some(h) = h {
            self = self.with_heuristic(h);
            self.endgame_eval = true;
            self.endgame_rank = rank.unwrap_or(15);
        }
        self
    }

    /// The heuristic player with macroxue's evaluation (no effect once a net is loaded).
    pub fn with_pass_score(mut self, p: Option<PassScore>) -> Self {
        self.pass_score = p;
        self
    }

    pub fn with_heuristic(mut self, h: Heuristic) -> Self {
        if h != self.heuristic {
            self.heuristic = h;
            self.heur = (0..(1u32 << 20)).map(if h == Heuristic::Macroxue { mx_line } else { row_heur }).collect();
        }
        self
    }

    /// A fixed search depth for the heuristic player too (None keeps its adaptive depth).
    pub fn with_depth(mut self, d: Option<u32>) -> Self {
        if let Some(d) = d {
            self.depth = Some(d.max(1));
        }
        self
    }

    pub fn with_lookup(mut self, lookup: Option<Arc<Lookup>>, rank: Option<u8>, min: Option<f32>) -> Self {
        self.lookup = lookup;
        self.lookup_rank = rank.unwrap_or(15);
        self.lookup_min = min;
        self
    }

    pub fn lookup(&self) -> Option<&Arc<Lookup>> {
        self.lookup.as_ref()
    }

    pub fn with_net(net: Arc<NTuple>, depth: u32) -> Self {
        Ai { net: Some(net), depth: Some(depth.max(1)), ..Ai::new() }
    }

    pub fn with_cprob(mut self, c: Option<f32>) -> Self {
        if let Some(c) = c {
            self.cprob_thresh = c;
        }
        self
    }

    pub fn with_top_bias(mut self, c: Option<f32>) -> Self {
        self.top_bias = c.unwrap_or(0.0);
        self
    }

    pub fn with_endgame_depth(mut self, d: Option<u32>) -> Self {
        self.endgame_depth = d;
        self
    }

    /// Whether `b` is played by the heuristic: no net, or the endgame handover reached.
    fn heuristic_board(&self, b: Board) -> bool {
        self.net.is_none() || (self.endgame_eval && max_rank(b) >= self.endgame_rank)
    }

    /// What a move is worth on top of its afterstate value.
    #[inline]
    fn reward(&self, r: u32, mode: Mode) -> f32 {
        if mode.heuristic { 0.0 } else { r as f32 }
    }

    /// The value of a board with no legal move. The net values future points, so none;
    /// macroxue's scores can go negative, so a loss must sit below every live board.
    fn dead(&self, b: Board, mode: Mode) -> f32 {
        if let Some(v) = mode.dead {
            v
        } else if mode.heuristic && self.heuristic == Heuristic::Macroxue {
            let r = max_rank(b) as i64;
            -((1i64 << 17).max(2 * (r << r)) as f32)
        } else {
            0.0
        }
    }

    pub fn tables(&self) -> &Tables {
        &self.t
    }

    fn eval(&self, b: Board, mode: Mode) -> f32 {
        if let (Some(net), false) = (&self.net, mode.heuristic) {
            let top = net.stages() > 1 && net.stage(b) == net.stages() - 1;
            return net.value(b) + if top { self.top_bias } else { 0.0 };
        }
        let rows = |x: Board| (0..4).map(|r| self.heur[((x >> (20 * r)) & 0xF_FFFF) as usize]).sum::<f32>();
        rows(b) + rows(transpose(b))
    }

    /// Best legal move, or None if the game is over.
    pub fn best_move(&self, b: Board) -> Option<Dir> {
        if let Some(lk) = &self.lookup {
            if max_rank(b) >= self.lookup_rank {
                if let Some((d, _)) = lk.suggest(b, self.lookup_min) {
                    return Some(d);
                }
            }
        }
        self.search_move(b)
    }

    /// Best legal move by expectimax alone.
    pub fn search_move(&self, b: Board) -> Option<Dir> {
        let handed_over = self.net.is_some() && self.heuristic_board(b);
        let depth_limit = if handed_over {
            self.endgame_depth.unwrap_or(3)
        } else {
            match (self.endgame_depth, self.depth) {
                (Some(e), _) if !self.endgame_eval && max_rank(b) >= 14 => e,
                (_, Some(d)) => d,
                _ => distinct_tiles(b).saturating_sub(2).max(3),
            }
        };
        let mut mode = Mode {
            heuristic: self.heuristic_board(b),
            // macroxue's spawn cut once its evaluation plays, unless --cprob was given.
            cprob_thresh: if handed_over && self.cprob_thresh == CPROB_THRESH { 1.0 / (1u64 << (depth_limit + 4)) as f32 } else { self.cprob_thresh },
            pass: f32::NEG_INFINITY,
            dead: None,
        };
        let pass_score = self.pass_score.filter(|_| mode.heuristic && self.heuristic == Heuristic::Macroxue);
        if let Some(p) = pass_score {
            // As macroxue's Node::Search: the cut is off when the board already scores badly.
            let r = max_rank(b) as i64;
            let twice_max = (2 * (r << r)) as f32;
            let score = self.eval(b, mode);
            let floor = if p == PassScore::Big { twice_max } else { 0.0 };
            if score >= floor {
                mode.pass = twice_max;
            }
        }
        let (best, value) = self.root(b, depth_limit, mode);
        // BIG_TUPLES: a losing-looking root is searched again with no cut and a harsher loss.
        if pass_score == Some(PassScore::Big) && value < 0.0 {
            mode.pass = f32::NEG_INFINITY;
            mode.dead = Some(-((1i64 << 22) as f32));
            return self.root(b, depth_limit, mode).0;
        }
        best
    }

    fn root(&self, b: Board, depth_limit: u32, mode: Mode) -> (Option<Dir>, f32) {
        let mut s = Search { ai: self, depth_limit, mode, tt: HashMap::new() };
        let mut best: Option<(Dir, f32)> = None;
        for d in DIRS {
            let (nb, r) = self.t.apply(b, d);
            if nb == b {
                continue;
            }
            let v = self.reward(r, mode) + s.chance(nb, 1.0, 0) + 1e-6;
            if best.map_or(true, |(_, bv)| v > bv) {
                best = Some((d, v));
            }
        }
        (best.map(|(d, _)| d), best.map_or(f32::NEG_INFINITY, |(_, v)| v))
    }
}

impl Search<'_> {
    fn chance(&mut self, b: Board, cprob: f32, depth: u32) -> f32 {
        if cprob < self.mode.cprob_thresh || depth >= self.depth_limit {
            return self.ai.eval(b, self.mode);
        }
        if self.mode.pass > f32::NEG_INFINITY {
            let v = self.ai.eval(b, self.mode);
            if v < self.mode.pass {
                return v;
            }
        }
        if depth < CACHE_DEPTH_LIMIT {
            if let Some(&(d, v)) = self.tt.get(&b) {
                if d <= depth {
                    return v;
                }
            }
        }
        let open = count_empty(b);
        let cp = cprob / open as f32;
        let mut res = 0.0;
        for i in 0..16 {
            if (b >> (5 * i)) & 0x1F == 0 {
                res += self.max(b | (1 << (5 * i)), cp * 0.9, depth) * 0.9;
                res += self.max(b | (2 << (5 * i)), cp * 0.1, depth) * 0.1;
            }
        }
        res /= open as f32;
        if depth < CACHE_DEPTH_LIMIT {
            self.tt.insert(b, (depth, res));
        }
        res
    }

    fn max(&mut self, b: Board, cprob: f32, depth: u32) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for d in DIRS {
            let (nb, r) = self.ai.t.apply(b, d);
            if nb != b {
                best = best.max(self.ai.reward(r, self.mode) + self.chance(nb, cprob, depth + 1));
            }
        }
        if best == f32::NEG_INFINITY { self.ai.dead(b, self.mode) } else { best }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntuple::{NTuple, TUPLES_4};

    fn grid(g: [[u8; 4]; 4]) -> Board {
        let mut b = 0u128;
        for r in 0..4 {
            for c in 0..4 {
                b |= (g[r][c] as u128) << (5 * (4 * r + c));
            }
        }
        b
    }

    #[test]
    fn endgame_handover_plays_like_the_heuristic_from_the_given_rank() {
        let net = Arc::new(NTuple::new(5000.0, 1, &TUPLES_4));
        let with_net = Ai::with_net(net.clone(), 2);
        let handover = Ai::with_net(net, 2).with_endgame_eval(Some(Heuristic::Macroxue), Some(14)).with_endgame_depth(Some(2));
        let mx = Ai::new().with_heuristic(Heuristic::Macroxue).with_depth(Some(2));
        // Below the handover rank the net decides; from it, macroxue's evaluation does.
        let small = grid([[11, 10, 9, 1], [2, 3, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]);
        assert_eq!(handover.best_move(small), with_net.best_move(small));
        let big = grid([[14, 13, 12, 1], [2, 5, 1, 0], [1, 0, 0, 3], [0, 0, 0, 0]]);
        assert_eq!(handover.best_move(big), mx.best_move(big));
        // The loss value only applies while macroxue's evaluation is in charge.
        let mode = |h| Mode { heuristic: h, cprob_thresh: CPROB_THRESH, pass: f32::NEG_INFINITY, dead: None };
        assert_eq!(handover.dead(big, mode(false)), 0.0);
        assert!(handover.dead(big, mode(true)) < 0.0);
    }
}
