//! Expectimax search. Leaves are scored either by a hand-tuned row heuristic or by a
//! trained n-tuple network (which values afterstates in points, so rewards are added).

use crate::board::*;
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

pub struct Ai {
    t: Tables,
    heur: Vec<f32>,
    net: Option<Arc<NTuple>>,
    /// Fixed search depth in chance layers; None = heuristic's adaptive depth.
    depth: Option<u32>,
    /// Deeper search once a 16384 is on the board, where the final merge chain is won or lost.
    endgame_depth: Option<u32>,
    /// Branches less likely than this are cut off and scored by the evaluator directly.
    cprob_thresh: f32,
}

struct Search<'a> {
    ai: &'a Ai,
    depth_limit: u32,
    tt: HashMap<Board, (u32, f32)>,
}

fn row_heur(row: u16) -> f32 {
    let line: [u32; 4] = std::array::from_fn(|i| ((row >> (4 * i)) & 0xF) as u32);
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
        Ai { t: Tables::new(), heur: (0..=u16::MAX).map(row_heur).collect(), net: None, depth: None, endgame_depth: None, cprob_thresh: CPROB_THRESH }
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

    pub fn with_endgame_depth(mut self, d: Option<u32>) -> Self {
        self.endgame_depth = d;
        self
    }

    /// What a move is worth on top of its afterstate value.
    #[inline]
    fn reward(&self, r: u32) -> f32 {
        if self.net.is_some() { r as f32 } else { 0.0 }
    }

    pub fn tables(&self) -> &Tables {
        &self.t
    }

    fn eval(&self, b: Board) -> f32 {
        if let Some(net) = &self.net {
            return net.value(b);
        }
        let rows = |x: Board| (0..4).map(|r| self.heur[((x >> (16 * r)) & 0xFFFF) as usize]).sum::<f32>();
        rows(b) + rows(transpose(b))
    }

    /// Best legal move, or None if the game is over.
    pub fn best_move(&self, b: Board) -> Option<Dir> {
        let mut s = Search { ai: self, depth_limit: match (self.endgame_depth, self.depth) {
                (Some(e), _) if max_rank(b) >= 14 => e,
                (_, Some(d)) => d,
                _ => distinct_tiles(b).saturating_sub(2).max(3),
            }, tt: HashMap::new() };
        let mut best: Option<(Dir, f32)> = None;
        for d in DIRS {
            let (nb, r) = self.t.apply(b, d);
            if nb == b {
                continue;
            }
            let v = self.reward(r) + s.chance(nb, 1.0, 0) + 1e-6;
            if best.map_or(true, |(_, bv)| v > bv) {
                best = Some((d, v));
            }
        }
        best.map(|(d, _)| d)
    }
}

impl Search<'_> {
    fn chance(&mut self, b: Board, cprob: f32, depth: u32) -> f32 {
        if cprob < self.ai.cprob_thresh || depth >= self.depth_limit {
            return self.ai.eval(b);
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
            if (b >> (4 * i)) & 0xF == 0 {
                res += self.max(b | (1 << (4 * i)), cp * 0.9, depth) * 0.9;
                res += self.max(b | (2 << (4 * i)), cp * 0.1, depth) * 0.1;
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
                best = best.max(self.ai.reward(r) + self.chance(nb, cprob, depth + 1));
            }
        }
        if best == f32::NEG_INFINITY { 0.0 } else { best }
    }
}
