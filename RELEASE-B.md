# Release B: the u128 board (BUILD 9, before stage 4)

Goal: let a tile exceed 32768 so the board can hold a 65536 (and a 131072), which the
u64 nibble board cannot. This is the one hard blocker for the 131072 end goal. Today the
game ends the instant two 32768s merge (`main.rs` play loop, `made_65536`), because the
result does not fit a 4-bit cell. Release B removes that wall.

DESIGN.md (branch stage3) section 3 and the release plan own this. Stage 4 training waits
on it. This file is the code-level plan; it changes the game engine, so correctness is
everything and the test suite is the gate.

## Gate (from the review)

`cargo test --release` passes, and games/s on a tiny run (one coordinator, one worker,
tiny net) must not drop more than 15% end to end. The two perf risks are the transpose
and the move tables growing from 1<<16 to 1<<20 entries (512 KB to ~16 MB, a worse cache
footprint). macroxue uses the same 20-bit line maps, so it is known feasible, but our
training runs far more moves/s than his depth-8 search, so measure before merging.

## Representation

`pub type Board = u128`. Five bits per cell, 0 = empty, k = 2^k. Cell (row r, col c) at
bit `5 * (4*r + c)`; row r occupies bits `20*r .. 20*r+20`. The board uses 80 of 128
bits. Rank 17 (131072) is the win; ranks go to 31, so no tile needs special handling in
the engine any more.

## File-by-file

### src/board.rs (the core, all of it touched)
- `type Board = u64` -> `u128`. Update the module doc (4-bit -> 5-bit, nibble -> cell).
- `Tables`: `left`/`right` become `Vec<u32>` (a 20-bit row does not fit u16), sized
  `1 << 20`; `score`/`score_rev` stay u32, sized `1 << 20`. Add `empties: Vec<u8>` sized
  `1 << 20` (per-row empty count) to keep `count_empty` a table sum, see below.
- `slide_row(row: u32) -> (u32, u32)`: same algorithm on 5-bit cells (shift by 5, mask
  0x1F). Drop the `.min(15)` cap and the 65536 comment: a merge of two rank-16 tiles now
  makes a real rank-17 tile. No cap needed below 31.
- `reverse_row`: rewrite for four 5-bit fields in a 20-bit word (reverse cell order).
- `Tables::new`: loop `for row in 0..(1u32 << 20)`; fill left/right/score/score_rev/empties.
- `apply`: rows are 20 bits: `((src >> (20*r)) & 0xF_FFFF) as usize`, write back shifted
  by `20*r`. Otherwise unchanged.
- `transpose` (THE hard part): the current nibble magic constants do not generalise.
  Implement a branch-free 5-bit transpose as six pairwise 5-bit field swaps for the
  off-diagonal cell pairs (0,1)-(1,0), (0,2)-(2,0), (0,3)-(3,0), (1,2)-(2,1), (1,3)-(3,1),
  (2,3)-(3,2). Each swap: mask the two fields, XOR-swap by the fixed bit distance between
  them. Verify against a reference loop transpose over a random board sample in a test.
- `count_empty`: the nibble SWAR trick does not align to 5-bit cells. Replace with a sum
  of `empties[row]` over the four rows, or a 16-cell loop. Prefer the table (it is hot in
  spawn and search).
- `max_rank`, `count_rank`, `distinct` (the `seen |=` loop), `spawn`, `from_grid`/
  `to_grid`, `made_65536`: shift by 5, mask 0x1F, place 5-bit ranks.
- `made_65536` -> `made_131072` (two rank-16 merge), used only for the win flag now.

### src/ntuple.rs
- `downgrade` -> keep, and add `abstract_board` (DESIGN "abstract()"): downgrade to a max
  rank of 14, applied once for a 32768 board and twice for a 65536 board (iterate until
  `max_rank <= 14`). The net sees boards through this, so the copied stage weights are a
  warm start and every existing weight stays valid. `downgrade`'s `missing`-rank scan and
  `present` bitmask move to 5-bit (shift 5, mask 0x1F; scan ranks 1..=16).
- Stage index (`max_rank.saturating_sub(FIRST_STAGE_RANK-1)`): unchanged in form; 65536
  boards now land in stage 3 as intended.
- Episode end: today it returns `max_rank: 16` when a 65536 is made and stops. Change to
  continue play; stop at rank 17 (131072) or a dead board. The 65536 merge reward is a
  normal TD step now (build 10 already fixed the reward path on u64; on u128 the tile
  simply stays).
- `net.indices` is fed abstracted boards, so index ranges do not change.

### src/main.rs
- `GameResult`: add `won_131072`; keep `won_65536` as a milestone flag.
- `play`: stop on rank 17, not on `made_65536`.
- Position IO (`positions`, the readers at lines ~198-201 and ~503): hex u64 cannot hold
  5-bit cells past rank 15. Switch to one-char-per-cell (0-9a-v for ranks 0-31), 16 chars
  per board, as DESIGN says. Keep a reader for old hex files behind a length check so
  `pos32k.txt` still loads.
- `report`: add a 131072 row; the `1u32 << r.max_rank` tile label already generalises.

### src/ai.rs
- Rank comparisons (`>= 14`, `>= 15`, `endgame_rank`, `lookup_rank`) stay valid; ranks
  just extend. Scores up to `1 << 17` fit the existing f32/i64. The endgame handover path
  calls into endgame.rs (below). No structural change expected; audit the `max_rank`
  guards once board.rs changes compile.

### src/endgame.rs (macroxue tables)
- Audit for 4-bit assumptions in how a board is indexed into a `Table` (cell extraction,
  anchor ranks). macroxue's own tables run on 5-bit cells, so the anchors (ranks up to 17)
  are the point; make cell extraction shift by 5. This only affects the 65536 handover
  experiments, not core training, so it can land in a second commit if needed.

## What stays for the laptop / is NOT in scope here
- The farm rollout (coordinator freeze, job keys, `/positions` endpoints, pool pipeline)
  is Release A and already shipped. Release B is the engine only.
- Do not start stage 4 training. Release B is the prerequisite; stage 4 is a separate,
  laptop-driven step.
- `BUILD` bump 8 -> 9 and any coordinator-visible format change go out only with the
  laptop in the loop, so a half-rolled-out engine never trains the farm on a bad board.

## Test plan
1. `cargo test --release` green (17 existing tests, ported to u128 literals, plus a new
   transpose-vs-reference test and a round-trip test for the one-char position format).
2. A 65536-then-continue test: a board one merge from 131072 reaches rank 17 and the game
   does not end early.
3. Perf microbench: `bench 200 1` games/s and a tiny `train` rate before (u64, this
   branch) and after (u128), on the same box, must be within 15%.

## Suggested path
Do board.rs + the test port as commit 1 on a `release-b` branch (self-contained once the
type cascade is fixed, gated by tests). Then ntuple.rs abstract/episode, then main.rs IO
and reports, then the endgame.rs audit. Measure the gate after board.rs and again at the
end. Keep it off main and off the active training lineage until the gate passes with the
laptop watching.
