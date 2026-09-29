# Training stages 3+ : design

Target: the 131072 tile. This document covers the training system for the stages after
stage 2 (16384 states, stage-0 frozen): the 32768 stage, the 65536 stage, the pipeline
that feeds them rare late positions, the big-tile abstraction for the endgame net, and
optional TD with search. Nothing here is implemented yet; the plan is to review this
first, then build it on branch `stage3` behind a `BUILD` bump so the farm updates itself.

Stage numbering used below (the index into the net's weight blocks, as `NTuple::stage`
computes it today):

| index | boards holding | name in README |
|---|---|---|
| 0 | at most 8192 | stage 1 |
| 1 | 16384 | stage 2 (running) |
| 2 | 32768 | stage 3 |
| 3 | 65536 | stage 4 |

## Decisions at a glance

1. `Board` becomes `u128`, 5 bits per cell. 65536 and 131072 are ordinary ranks; the
   win is rank 17. The net file format does not change.
2. Stage index from the raw board: 32768 boards use stage 2's weights, 65536 boards
   stage 3's. Stages are frozen with the existing `freeze=N`; a new stage is a copy of
   the previous one.
3. The net sees boards through `abstract()`: iterated tile downgrading to a max rank of
   14 (today's `downgrade`, applied twice for 65536 boards). That makes the copied
   weights a real warm start and keeps every existing weight valid.
4. The restart pool becomes a farm-wide pipeline: workers harvest boards keyed by chain
   state `(max, floor, top_free)`, upload them with each chunk, fetch a sample back, and
   the coordinator keeps and persists the global pool. Restarts pick uniformly over
   keys, which is the rarity weighting.
5. Search enters training only as an option for late-stage restart episodes (1-ply
   expectimax move choice, plain TD target) and as an offline harvester that fills the
   65536 buckets.
6. No TC. Coordinator-side shared TC is designed (4.4) but not planned.
7. `BUILD` 7 -> 8, two new endpoints (`/positions` GET and POST), eight new job keys, the
   coordinator enforces `freeze` on incoming deltas.

## 1. Where we are and how far 131072 is

Current farm result: mean 387k, 32768 in about 26% of fresh games, 1-ply greedy.
macroxue's expectimax AI at depth 8: mean 711,769, 32768 in 80.5%, 65536 in 3.5%,
record 1,704,908 (65536 + 32768 + 16384), and it does not attempt 131072.

A rough decomposition of 131072, taking macroxue's numbers as what a strong player does:

- build 32768 with 16 free cells: about 80%
- build a second 32768 with one huge tile on the board (15 cells): 3.5% / 80%, about 4%
  conditional. That gives 65536.
- build 32768 again with 65536 on the board (15 cells): about 4% again
- build one more 32768 with 65536 and 32768 on the board (14 cells): far harder, guess
  0.1% to 0.5%. That gives the second 65536, and 131072 follows.

So the whole game hinges on the last two builds, states that a fresh game reaches once in
tens of thousands of games at best. Fresh self-play spends almost none of its moves
there. Everything in this design serves one purpose: play and learn those states millions
of times instead of thousands, with an evaluation that generalises across them.

## 2. What macroxue does, and what we take from it

Read from the source (`board.h`, `node.h`, `tuple.h`, `plan.h`, `2048.cc`), not the README.

**Board.** 5 bits per cell (`int board[4][4]`, lines packed as 20-bit integers).
`move_map[1 << 20]` slides a line, `score_map[1 << 20]` scores it. Ranks up to 31, so
65536 and 131072 need no special handling.

**Evaluation.** Very simple and hand-made. For a line a,b,c,d with tile "cost"
T(r) = r * 2^r: sum over adjacent pairs of `a + b` if a >= b, else `(a - b) * 12` (a
penalty), plus `a` again when a == b. Rows and columns are all scored in the orientation
that puts big tiles top-left, so the search is what keeps the chain in that corner.

**Search.** Expectimax, depth 3 to 8, spawn probability cut at `1 / 2^(depth + 4)`, a 4M
entry transposition cache, and a `pass_score` cut: a node whose static value is below
twice the cost of the largest tile is not expanded. Depth 8 runs at 17 moves/s.

**The lookup tables, which is the part that matters for us.** `Tuple<T0..T15>` fixes some
cells as anchors (parameter 1) and treats the rest as a small-tile sub-board with a rank
cap per cell. `Tuple10` fixes the top-left 3x2 block (six anchors) and indexes the other
ten cells with caps 10,9 / 10,9,8,8 / 8,8,8,8: 10*9*10*9*8^6 = 2.1G entries of 2 bytes
(the 4.2 GB "small" table). `Tuple11` (the 2022 build) fixes five anchors and indexes
eleven cells: about 22G entries, 27 GB of RAM. Each entry holds the best move and the
probability (14 bits) of reaching the goal.

The value is computed once by exhaustive expectimax with memoisation over the sub-board
index (`TryMoves` / `TryTiles`), where:

- `Prefill` replaces the anchors by the canonical ranks 15, 14, 13, ... The anchors' true
  values never enter the table. Only the small tiles are indexed. This is the big-tile
  abstraction: a 65536-32768-16384 chain and a 16384-8192-4096 chain with the same small
  tiles share one entry.
- `IsRegular` keeps the sub-board inside the abstraction: anchors distinct and ordered,
  every free cell below `anchor_rank - offset` where `anchor_rank` is the smallest anchor
  (so the free tiles can never merge into the anchors during the lookup, except at the
  goal).
- `HasSameTops` rejects moves that shift an anchor.
- `IsGoal` is "the next chain tile stands where it continues the staircase" (for Tuple10:
  `anchor_rank - 1` at (0,2), `anchor_rank - 2` at (1,2) and a specific tail). Success
  probability 1 there; the normal engine then merges it in.
- At play time all eight symmetries are tried; Tuple11 is used when its probability is
  above 0.1, else Tuple10 when above 0, else search.

**What carries over.**

1. The problem decomposes into "build the next chain tile with the chain intact". That is
   the unit our restart pool should harvest and our metrics should report, not "reached
   tile X".
2. Anchor canonicalisation is what lets one table serve every stage. For a learned value
   function the equivalent is tile downgrading (section 4.3), which we already use; the
   design keeps it and makes it exact for 65536 boards.
3. Their strength in the constrained endgame comes from search, not from the evaluation.
   Greedy 1-ply is too weak to even generate the trajectories stage 4 needs, which is
   why search enters training as a harvesting and an optional learning tool (section 6).
4. Exact DP over a 10-cell sub-board is a play-time tool (4 to 27 GB tables), not a
   training substitute. A Rust port of a Tuple10-like table could sit on top of our
   search for the final attempt; it is out of scope here (section 11).
5. Their board is 5 bits per cell. Ours is 4, and that is the first blocker.

## 3. The blocker: 4-bit ranks cannot hold a 65536 board

`Board = u64`, one nibble per cell: 0 = empty, 1..15 = 2..32768. A live 65536 board needs
the symbols empty, 2, 4, ..., 65536: 17 symbols, and a nibble has 16. Today two 32768s
merge into a nibble 15 worth 65536 points and the game is declared won there
(`made_65536`). To play for 131072 the engine has to keep going with a real 65536 on the
board, build a second one, and merge them.

Every encoding trick that keeps u64 fails or degenerates into the same table size:

- Removing the largest missing rank (what `downgrade` does for the evaluator) is lossless
  for live boards but not playable: spawns are absolute (2 and 4), so the shift is wrong
  whenever the missing rank is small.
- A side mask marking which nibble-15 cells are 65536s needs the slide table indexed by
  (16-bit row, 4-bit mask): 2^20 entries, the same as 5 bits per cell, with a far uglier
  data model and mask transport on every slide.

**Decision: `Board = u128`, 5 bits per cell, 80 bits used.** Lines are 20 bits, the four
slide tables grow to 2^20 entries (16 MB total, from 768 KB). Ranks go to 31; 17 is the
131072 tile and ends the game as the win. Everything else in this document assumes it.

Cost: engine ops roughly 2x slower (bigger tables, u128 shifts). Training time is
dominated by the 64 random weight reads per `value()`, so the end-to-end hit should be
well under 10%; measured on a tiny run before rollout. `transpose` is replaced by column
gather/scatter (four 5-bit extracts per column), which is the same order of work as the
current bit tricks.

What changes with it: `board.rs` (type, tables, `count_empty`, `spawn`, `print`, a
`parse`/`format` using one character per cell, digits then `a`..`h` for ranks 10..17, so
existing 16-hex-digit position files still parse), `ai.rs` (`HashMap<Board, ..>`),
`main.rs` (the win is now `max_rank >= 17`; `report` gains 65536 and 131072 rows),
`ntuple.rs` (`indices` goes through the abstraction in section 4.3 to get a nibble
board), `dist.rs` (pool wire format, chunk fields), `bot/bot.js` (sends `g` for 65536).
The net file format is unchanged: weights are indexed by 4-bit abstracted boards exactly
as today, so the trained stage 0 and 1 weights stay valid without conversion.

Regression oracle: the current u64 engine moves into the test module verbatim. A test
plays a few thousand random moves on both engines with the same RNG and asserts identical
boards and scores while no tile exceeds 32768, and the same for the row tables cell by
cell. Deterministic self-play also gives a bench check: the same seed must give the same
score on the old and new binaries as long as no game hits 65536.

## 4. Stages 3 and 4

### 4.1 Keying

Unchanged rule, taken from the *raw* board: `stage = (max_rank - 13).clamp(0, stages-1)`.
32768 boards are index 2, 65536 boards index 3. Today `indices` computes the stage from
the *downgraded* board, which sends nearly every 32768 board to stage 1's weights (only a
32768 board with no missing rank, which almost never exists, reaches index 2). That is
fine while stage 2 runs and stage 2's weights are the ones learning 32768 play, but it
means the stage split for 32768 has to be switched on deliberately at the stage 3 start
(section 8), not inferred from the net's stage count.

I considered splitting stages further on the second chain tile (16384 present or not, as
Yeh et al. did). Each stage costs 512 MB on every worker twice (net and mirror), so it is
not in the plan; the abstraction in 4.3 is meant to do that job through shared weights.

### 4.2 Weights, freezing, initialisation

- Stage 3 training: net expanded to 3 stages, `freeze=2` (stages 0 and 1 fixed).
- Stage 4 training: expanded to 4 stages, `freeze=3`.
- `expand_stages` seeds a new stage with a copy of the last one (weight promotion,
  Jaskowski 2018). Because of the abstraction below, the copy is a real warm start, not
  just a non-zero init: the new stage sees boards that look exactly like the ones the
  copied weights were trained on.
- Frozen stages never change on any worker, so `diff`, `snapshot` and the mirror are
  restricted to the unfrozen range. That caps worker RSS at (stages * 512 MB) for the
  net plus 512 MB per trainable stage for the mirror, instead of double. At 4 stages:
  about 2.5 GB. Without this it would be 4 GB, which some farm machines cannot spare.
- New CLI: `g2048 stages FILE N` expands (or reports) a saved net's stage count. It
  replaces the `train --resume --stages` detour and runs on the server before the
  coordinator restarts.

### 4.3 Big-tile abstraction for the value function

`abstract(b: Board) -> u64` is the only place the net sees a board, and it produces the
4-bit index board:

```
loop while max_rank(b) > 14:
    find the largest rank below max that is absent from the board
    none found: break                      (see below)
    every tile above it moves down one rank
return the 16 nibbles
```

For boards up to 16384 it is the identity. For a 32768 board it is exactly today's
`downgrade`, so stage 1's and stage 2's learned behaviour is untouched. For a 65536 board
it downgrades twice when two gaps exist and once otherwise. A live 65536 board always has
at least one gap (16 cells cannot hold 16 distinct tiles and an empty cell or a merge
pair), so the result always fits in nibbles; a full board of 16 distinct tiles is dead and
its value is irrelevant.

Why downgrading rather than macroxue's anchor canonicalisation (chain to 15,14,13...,
free tiles kept): anchor canonicalisation collides at rank 16 whenever the chain ends in
a duplicate (two 16384s under 65536 and 32768 both want nibble 14), so it needs the gap
shift anyway; and downgrading is what the existing weights were trained on, which makes
the stage copy a working warm start. The information loss is the known one: a 32768
board missing 4096 and one missing 8192 can map to the same nibble board. Stages keep the
absolute level apart (index from the raw max rank), the loss is within a stage only, and
Yeh et al. and Guei et al. train with the same loss.

The warm start matters more than it looks. The OTD optimistic init (320000 spread over 64
weights) is far below the value of a healthy 32768 board, so a stage whose weights for
"boards with a 15 in them" were never trained would refuse the merge that creates the
big tile; that refusal is what `downgrade` was written for. With the abstraction, a fresh
65536 board evaluates as a familiar 32768 board with stage 3's weights copied into stage
4, so the merge into 65536 is valued correctly from the first game.

### 4.4 Learning rate and TC

The OTD schedule stays (0.1, 0.01, 0.001 per 64 weights at 50% and 75% of `goal`). Two
additions:

- `stage_start=<episodes>` in the job: progress is `(episodes - stage_start) / goal`, so
  a stage can start without editing the `.episodes` file on the server.
- No TC. Per-worker TC accumulators are what broke stage 1. The only acceptable form is
  shared accumulators, and the one place that sees every update is the coordinator: it
  could keep `sum(delta)` and `sum(|delta|)` per weight from the delta stream and scale
  each incoming delta per weight before applying it. Zero extra wire cost upstream, but
  the worker must then receive the per-weight taken amounts to stay consistent with the
  master (the reply becomes delta-sized, doubling download traffic for the sender), and
  it costs 4 GB of accumulators on the coordinator at 4 stages. Designed, not planned:
  the alpha schedule already does the job the paper used TC for. Listed for review.

## 5. Restart-pool pipeline

Today each worker keeps a private pool of stage-entry boards (the first board where 16384
or 32768 appeared), capped at 100k per stage, reservoir sampled, picked uniformly over
non-empty stages, and lost on restart. That gives stage 2 its 90% restarts. It is not
enough for stages 3 and 4: entry boards are the *easy* end of the stage, a greedy net
rarely gets from a 32768 entry board to the interesting 14-cell 32768 build, a fresh
stage-4 net reaches 65536 so rarely that its pool would stay empty for hours, and every
worker restart throws its harvest away.

### 5.1 What gets harvested

Following section 2 point 1, the unit of progress is the chain state. For a board define:

- `max`: rank of the largest tile.
- `floor`: rank of the smallest tile of the chain, where the chain is max, max-1, max-2,
  ... as long as each rank is present exactly once.
- `top_free`: the largest rank below `floor` (0 if none).

Bucket key = `(max, floor, top_free)` packed in a u16. Within a stage the key walks
through the build cycle: top_free climbs towards floor, they merge, floor rises,
top_free drops. A worker records the post-spawn board the first time each key appears in
an episode (a small set of seen keys per episode). Only boards with `max >= 14` are kept;
below that fresh play covers everything.

The old "first board of the stage" is the key `(max, max, small)` and stays a member of
the pool, so stage 2 style restarts remain available.

### 5.2 Worker side

- Per chunk, a per-key reservoir of at most 64 boards, uploaded with the chunk's delta as
  a new `POST /positions` (binary: count, then key u16 + board u128 each; a few KB).
- Every sync, `GET /positions?n=N&min_stage=S` returns N boards sampled by the
  coordinator, N from the job (`pool_n`, default 4000: 72 KB). They merge into the local
  pool, which is now a map key -> reservoir of `pool_cap` (default 2000) boards.
- The local pool is saved in the cache dir next to `net.bin` and reloaded on restart.
- Restart picking: `restart` is the probability of a restart as today; `restart_stage=S`
  restricts picks to keys with `max - 13 >= S`; the key is chosen uniformly among eligible
  non-empty keys, then a board uniformly within it. Uniform over keys is the rarity
  weighting: a key that fresh play hits once an hour gets the same share as one hit every
  second. `restart_weight=deep` (optional, later) would tilt towards higher `floor`.

### 5.3 Coordinator side

- Global pool: key -> reservoir of `POOL_CAP` (100k) boards, merged from uploads with
  reservoir sampling, so late uploads and early uploads have equal weight. Memory: 16
  bytes per board, a few tens of keys populated, under 100 MB.
- Persisted to `MASTER.pool` on every save and restored on start, like `.episodes` and
  `.best`.
- `GET /positions` sampling: pick a key uniformly among those with `max - 13 >= S`, then
  a board; repeat N times. Cheap, no index needed.
- Status page: a `POOL` block with the populated keys as `32768 | floor 8192 | free 2048:
  41,203 boards`, and the per-stage restart outcomes from 5.4.

### 5.4 Chunk statistics for restarts

Only fresh games count towards the results table today, which is right for "how strong is
the net" and useless for "is stage 4 learning". `Chunk` gains, per start stage 2 and 3,
the restart episode count and how many reached the next tile (`r2=count,reached`,
`r3=count,reached` on the wire; older fields keep their names). `reached` grows from 5 to
7 entries (65536, 131072); the parser already tolerates missing trailing entries, so a
new coordinator reads old workers and vice versa during the rollout minute.

### 5.5 Seeding stage 4 with search

A stage-3 greedy net will produce very few 65536 boards for a long time. Two seeding
paths, both using the same code:

1. Farm harvest: `search_p=0.02 search=1 search_stage=2` makes 2% of the restart episodes
   that start at stage 2 or later play with 1-ply expectimax (section 6) instead of
   greedy. Those episodes still learn, still harvest, and reach 65536 much more often.
2. Offline: `g2048 harvest POOL_FILE --net FILE --depth 3 --url URL --token-file F`
   plays boards from a pool file (or from `GET /positions`) with deep search, uploads
   every new key it reaches through the same `POST /positions`. It runs on the server's
   idle cores or a laptop; nothing else in the farm has to know.

Optional augmentation, off by default (`upgrade_p`): take a stage-3 pool board whose free
tiles are all at least two ranks below the floor, raise every chain tile one rank, and
use the result as a stage-4 start. It produces plausible but not observed 65536 boards
(the chain sits one rank higher than the free tiles would normally allow). Listed because
it bootstraps stage 4 instantly; not recommended until real harvested boards are compared
against it.

### 5.6 Bootstrapping stage 3 itself

The build-7 workers hold their stage 2 pools in memory only and lose them when they exit
for the update, so stage 3 starts with an empty pool. That is fine: about a quarter of
fresh games reach 32768, so the stage-entry key `(15, 15, small)` refills at hundreds of
boards per second farm-wide and the deeper keys follow as restart episodes play through.
From build 8 on, pools survive restarts (saved next to `net.bin`, and the global one in
`MASTER.pool`), so the stage 4 start does not pay this again.

## 6. TD with search (optional, late stages only)

The mechanism: for an episode flagged "search", moves are chosen by `Ai::best_move` with
the net at depth `search` (chance layers, as `--depth` counts them), and TD(0) updates
are unchanged: the target is `r + V(after')` with the greedy one-ply value, so the value
function stays the plain afterstate value the greedy policy and the pool need. A
`search_target=1` variant would use the search value as the target (TreeStrap style);
cheaper to try later than to design now.

Cost, from the code's search shape (each chance layer multiplies by about 2 * empty
cells * 4 moves): depth 1 is 20x to 120x a greedy move, depth 2 another 25x to 250x on top,
fewer with few empty cells, which is the endgame. So depth 1 is affordable as a fraction
of late-stage episodes, depth 2 is a harvest tool, and nothing below `search_stage` ever
searches. Job keys: `search` (depth, 0 = off), `search_stage` (index), `search_p`
(fraction of restart episodes at or above that stage that search). Per episode the flag
is drawn once, so chunk statistics can report searched and greedy restarts separately.

Guei et al. report that search-selected moves during TD improve the late-stage net but
not the early one, which matches restricting it to stages 2 and 3.

## 7. Protocol, job keys, build

All new traffic uses the existing curl-driven HTTP with the bearer token, the same
`handle` dispatcher and binary bodies. New endpoints: `POST /positions`, `GET
/positions`. Changed bodies: `POST /delta` query gains `r2`, `r3` and two more `reached`
entries. `GET /net`, `/deltas`, `/job`, `/beat`, `/status`, `/shutdown` are unchanged.

| job key | default | meaning |
|---|---|---|
| `freeze` | 0 | stages below this stay fixed (exists) |
| `restart` | 0 | probability an episode starts from the pool (exists) |
| `restart_stage` | 1 | only pool keys at this stage index or above are picked |
| `pool_n` | 4000 | boards fetched per sync |
| `pool_cap` | 2000 | boards kept per key on a worker |
| `stage_start` | 0 | episodes at which this stage began, for the OTD schedule |
| `search` | 0 | expectimax depth for searched episodes; 0 = greedy only |
| `search_stage` | 2 | searched episodes only when starting at this stage or above |
| `search_p` | 0 | fraction of eligible restart episodes that search |
| `upgrade_p` | 0 | fraction of stage-4 restarts synthesised by chain upgrade |

Two safety nets on the coordinator, both cheap:

- The effective job carries `net_stages=N`, generated from the master, never set by the
  operator. A worker whose net has a different stage count discards it and downloads
  the master again. Today a worker with fewer stages would silently drop the deltas for
  the stages it lacks (`apply` skips out-of-range indices) and drift.
- Delta entries that fall in a frozen stage (`index < freeze * stage_size`) are dropped
  before `apply`, and the count of dropped entries goes back in the reply so the worker
  can log it. `freeze` is a worker-side rule today; enforcing it at the master means a
  misconfigured or outdated worker cannot write into a finished stage.

`BUILD` goes from 7 to 8. Workers on build 7 keep working against a build-8 coordinator
until `farm set version=8`: they never call `/positions`, and the extra query fields are
ignored by the old parser. They do compute the stage from the downgraded board, so they
would keep training stage 1's weights on 32768 boards; the rollout below pauses them
first and the coordinator-side `freeze` catches anything that slips through. The reverse
(build-8 worker, build-7 coordinator) is never needed because the coordinator restarts
first.

## 8. Rollout

Stage 3 start, once stage 2 hits its goal:

1. `farm pause`. Build and publish the build-8 binaries (Linux and the Windows exe).
2. `POST /shutdown` saves the master and the delta log, then `systemctl stop` the unit
   so nothing races the next step (the log restore only applies if the saved net is the
   one the log ends at; expanding stages keeps the existing indices, so it still does).
3. `g2048 stages nets/otd.bin 3` on the server: index 2 copied from index 1, the file
   grows to 1.5 GB.
4. `systemctl start`: the coordinator comes up on build 8 with the 3-stage net and
   `net_stages=3` in the effective job.
5. `farm set version=8 freeze=2 restart=0.9 restart_stage=2 stage_start=<episodes>
   goal=<N> pause=0`. Workers see `version` before `pause`, exit for the update, and the
   new build downloads the 3-stage net (a full download, since the stage count changed)
   and trains.
6. Watch the status page's POOL block and the `r2` restart outcome column.

Stage 4 start: same sequence with `g2048 stages nets/otd.bin 4`, `freeze=3`, `restart_stage=3`,
plus `search_p=0.02 search=1` for a while so the 65536 buckets fill, and the offline
`harvest` on the server if they fill too slowly.

## 9. Testing plan

`cargo test --release` must pass at every commit. New tests, all tiny:

- Engine oracle (section 3): 5-bit engine against the retained u64 engine over random
  play; every 20-bit line against `slide_row`; transpose and column moves round trip;
  65536 + 65536 merges into rank 17 with 131072 points; `count_empty` on 5-bit boards.
- `abstract`: identity below 32768; equals old `downgrade` on the three existing cases;
  a 65536 board with two gaps downgrades twice to max 14; one gap gives max 15; every
  generated live 65536 board fits (no cell above 15, distinct ranks stay distinct).
- Stage index from the raw board: 32768 boards go to index 2, 65536 to index 3, clamped
  when the net is smaller.
- Frozen-range `diff`/`snapshot`: training with `freeze=1` produces no delta in stage 0
  and the mirror covers only trainable stages.
- Pool: key computation on hand-built boards (chain, floor, top_free, duplicate ends the
  chain); reservoir merge keeps caps; `/positions` body round trip; sampling respects
  `min_stage`; save/load of the pool file.
- Chunk query round trip with the new fields and with the old 5-entry `reached`.
- Coordinator drops delta entries in frozen stages and reports the count; a worker whose
  stage count differs from `net_stages` re-downloads.
- Search episode: a 1-ply searched `train_episode` on a 4-tuple net runs and updates the
  same weights a greedy one would touch.

Tiny runs (not in `cargo test`, run by hand and in a `farm/tiny.sh`):

- `g2048 train` with `--tuples 4 --stages 4 --restart 0.5 --search 1 --search-stage 0`
  for a few thousand games: exercises the abstraction, the pool keys and search without
  the 512 MB tables.
- Coordinator on `127.0.0.1` plus a worker with `--url http://127.0.0.1:PORT` and
  `secs=5`, two rounds: the worker uploads positions, fetches them back, and the pool file
  appears next to the master. Two workers with different names check that `share` and
  the pool merge still agree.
- Old and new binaries, `bench 64 1` with the current stage 2 net if available on the
  reviewer's machine: same scores while no game reaches 65536.

## 10. Risks and questions for review

1. **u128 board.** Largest blast radius in the plan and unavoidable for 65536. Is a 2x
   engine slowdown acceptable if the end-to-end training rate drops by, say, 5%? The
   alternative of keeping u64 with a 65536 mask has the same table size and worse
   code; I do not recommend it.
2. **Stage index from the raw board.** This changes which weights 32768 boards use the
   moment the 3-stage net is loaded. It is exactly the intended stage 3 switch, but it
   must not be deployed while stage 2 is still meant to be learning 32768 play. The
   rollout order in section 8 takes care of it; worth a second pair of eyes.
3. **Abstraction choice.** Iterated downgrade over anchor canonicalisation, for the warm
   start and continuity. If stage 4 turns out to need finer distinctions (the gap
   ambiguity), the fallback is a stage split on the second chain tile at +1 GB per
   worker.
4. **Pool key granularity.** `(max, floor, top_free)` gives a few dozen live keys per
   stage. Too fine and the near-dead keys get as much restart weight as the healthy
   ones; too coarse and the deep states drown. The uniform-over-keys rule is a first
   guess; the status block exists so we can see the distribution and change the rule
   without a protocol change (the key is opaque to the coordinator).
5. **Search cost in training.** Depth 1 for 2% of late restarts is cheap; anything more
   competes with throughput. Defaults are off.
6. **Chain-upgrade augmentation.** Fast bootstrap for stage 4 but synthetic. Off by
   default; needs an A/B against harvested boards before use.
7. **Memory.** 4 stages is 2 GB of net per worker plus 512 MB mirror plus the 16 MB
   tables. Machines under 4 GB free should get `stop.NAME=1` at the stage 4 start.
8. **Coordinator-side TC.** Designed in 4.4, not planned. Say if you want it built.

## 11. Later, out of scope for this branch

- Play-time endgame DP in the style of `Tuple10` on top of our net-driven search for the
  final 131072 attempts (a 4 GB table is fine on the server).
- `search_target=1` (search value as TD target).
- Restart weighting by chain depth, once the pool distribution is visible.
- A `positions`/`endgame` refresh: both read and write the new one-char-per-cell format
  and stop at 131072 instead of 65536.
