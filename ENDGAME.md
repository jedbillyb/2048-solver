# Reaching 65536: what macroxue does, what we built, what it measures

Branch `endgame-65536`. Goal: raise the 65536 rate of the n-tuple player, which was 0 in
every measurement so far (600 full games at depth 2, 100 playouts from 32768 boards at
depth 2 and 3, extra stage-3 training).

## 1. Findings from macroxue/2048-ai

Read from the source (`node.h`, `board.h`, `tuple.h`, `plan.h`, `2048.cc`, `array.h`),
compiled and run here for the numbers marked "measured here".

**Board.** 5 bits per cell, lines of 20 bits, `move_map[1 << 20]`. 65536 and 131072 are
ordinary ranks. Nothing else about the engine is special.

**Evaluation.** One line table, applied to the four rows and the four columns in a fixed
orientation (large tiles to the top-left). For a line a,b,c,d with tile cost
T(r) = r * 2^r: T(a) plus, for each adjacent pair, `T(x) + T(y)` if x >= y else
`(T(x) - T(y)) * 12`, plus `T(x)` again if x == y. That is all: no empty-cell count, no
smoothness. It is a corner-forcing monotonicity score. Ported here as `--eval mx`
(`ai.rs`, `mx_line`).

**Search.** Plain expectimax, depth d = d chance layers, spawn-probability cut at
`1 / 2^(d + 4)`, a 4M-entry transposition table keyed by board, and two cuts: a node
whose static score is below twice the cost of the largest tile is not expanded
(`pass_score`), and a lost board scores `-max(2^17, 2 * T(max))`. Depth 8 runs at 17
moves/s; depth 3 at about 6000.

**Lookup tables, the part that matters.** `Tuple<T0..T15>` fixes some cells as anchors
(parameter 1) and gives every other cell a cap on the rank it may hold. Two shapes are in
use:

| table | anchors | free cells and caps | entries | file |
|---|---|---|---|---|
| Tuple10 | top-left 3x2 block (6) | (3,0):10 (3,1):9, row 2: 10 9 8 8, row 3: 8 8 8 8 | 2.12 G | 4.2 GB |
| Tuple11 small | block without its bottom-right cell (5) | (3,0):7 (2,1):8 (3,1):7, row 2: 7 7 7 7, row 3: 6 6 6 6 | 1.22 G | 2.4 GB |
| Tuple11 big (2022) | same 5 | caps 9/10/9, 9 9 9 9, 8 8 8 8 | 21.8 G | 27 GB RAM |

Each entry is the best move (2 bits) and the probability of reaching the goal (14 bits).
The value is an exact expectimax over the sub-game, memoised on the free cells only:

- `Prefill` replaces the anchors by the canonical staircase 15, 14, 13, ... The anchors'
  real values never enter the table, so one table serves every corner from 2048 up.
- `IsRegular` keeps the sub-game inside the abstraction: anchors distinct where they touch
  and in order, every free cell below `anchor_rank - offset` where `anchor_rank` is the
  smallest anchor (so free tiles never merge into anchors), and `anchor_rank` at most
  `kMaxAnchorRank` (10 for Tuple10: the block ends in 1024 or less).
- `HasSameTops` rejects any move that slides an anchor.
- `IsGoal`: the next chain tile stands where the chain continues, with the staircase under
  it that the following merges need. For Tuple10 that is `anchor - 1` below the block at
  (0,2) with `anchor - 2` at (1,2) and a mergeable pair of `anchor - 3`, or the same down
  column 3. For Tuple11 it is the sixth block tile at (2,1).
- `TryMoves` / `TryTiles`: max over moves, average over spawns (0.9 / 0.1), probability 1
  at the goal, 0 when no move keeps the position regular. Computed lazily from whatever
  position comes up, every reachable position on the way is stored, the table is saved on
  exit (sparse, 512-entry chunks).
- At play time all eight orientations are tried. Tuple11 is followed when its probability
  is at least 0.9, else Tuple10 whenever it sees any way to the goal (p > 0), else search.

**What the tables are worth, measured here** (their binary, `-d3`, seeds 2001+, this
machine):

| macroxue depth 3 | games | mean score | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|
| tables disabled (`SuggestMove` goes straight to search) | 40 | 134,018 | 10.0% | 0% | 0% |
| tables on (published, 1000 games) | 1000 | 493,058 | 81.4% | 52.5% | 1.3% |
| tables on (measured here) | see section 4 | | | | |

Their evaluation and search on their own are weak: at depth 3 they are roughly where our
hand heuristic is. Everything from 134k to 493k, and all of the 65536s, comes from the
tables. The 2019 to 2022 jump (417k to 493k at depth 3, 0.02% to 3.5% 65536 at depth 8
versus the OTD paper) came from making Tuple11 bigger.

**Why this is the right target for us.** The n-tuple net at depth 2 already reaches
32768 in 60% of games, more than macroxue at depth 5. What it cannot do is the part the
tables do: when six large tiles sit in the corner and the remaining ten cells have to
produce the next 512 or 1024 without disturbing them, the outcome is decided tens to
hundreds of moves ahead. A learned value over the whole board plus two or three plies
does not see it; an exact solution of the ten-cell sub-game does. Building a second
32768 next to the first means winning that sub-game many times in a row with one cell
fewer, which is exactly where our games end (average 10k moves survived from the first
32768, about two thirds of the way).

## 2. What was built

`src/endgame.rs`, no new dependencies, builds as before for Windows.

- `Layout`: the three shapes above as `block10` (Tuple10), `five7` (Tuple11 small) and
  `snake8` (macroxue's Snake9 from `snake.h`, for a net that keeps its chain along the top
  row instead of in a block), each with a smaller variant by lowering the number
  (`block8`, `five5`, `snake6`), the regular and goal rules ported one for one, plus one
  extra rule: top-row anchors must not be smaller than the anchor rank, so a table is
  never consulted on a board where a free tile could merge into an anchor.
- `Table`: one `AtomicU16` per entry, allocated zeroed so untouched pages cost nothing;
  the same lazy expectimax DP as `TryMoves`/`TryTiles`, lock-free (two threads solving the
  same region store the same values). Saved as sparse 4096-entry chunks to
  `DIR/<layout>.bin`, loaded on start.
- `Lookup`: all tables over all eight orientations, the move mapped back to the board's
  own orientation, hit counters, per-layout thresholds as macroxue (0.9 for `five`, any
  positive probability otherwise).
- `Ai::best_move` asks the lookup first on boards whose largest tile is at least
  `--lookup-rank` (default 15, so play before the first 32768 is unchanged byte for byte)
  and falls back to the usual expectimax. `search_move` is the old path.
- CLI: `--tables DIR` turns it on for `bench`, `endgame`, `positions`, `serve`;
  `--layouts`, `--lookup-rank`, `--lookup-min`. `positions --rank R` saves boards where
  rank R first appears. `formation POS_FILE` says which layouts the saved boards fit, so
  the net's corner shape can be checked before relying on a table. `probe POS_FILE`
  prints the table's move and probability per board. `--eval mx` and a fixed `--depth`
  for the heuristic player exist for the stand-in measurements below.

## 3. Validation of the port

macroxue's binary was compiled from the cached source and run on this machine.

- `./2048 -d3 -v 2050` reaches 65536 (score 1,133,392, 36,430 moves, 9,789 moves/s), as
  the README promises. Its verbose log gives, for every move, the board, the move and
  the goal probability of the table that answered (24,742 table moves, 11,036 searched).
- 300 of those Tuple10 boards were fed to `g2048 probe` with a fresh `block10` table. The
  table applied to all 300. The move agrees on 261 (87%). The probabilities are all
  about 9% higher here (0.109 against 0.100, 0.1052 against 0.096), never lower: their
  `SetProb` truncates to 1/16000 at every level of the recursion and the error compounds
  over the few hundred levels a sub-game takes; this port rounds. The disagreeing moves
  are near-ties under that bias, and both implementations break exact ties by move
  order, which differs.
- 276 of the Tuple11 boards against a fresh `five7` table: applies to all 276, same move
  on 201 (73%), probability within 0.0015 on 209 and higher here on the other 67, never
  lower. These boards sit near probability 1, where several moves tie exactly and the
  move order decides, so the move agreement is as good as it can be read.
- The two players reproduce each other: see the first two rows of the results table.

## 4. Results

All runs: this 4-core machine, `--eval mx --depth 3` (macroxue's evaluation and depth,
no net), 40 games, seeds 2001 to 2040 (`bench 40 2001`), tables consulted from the first
move (`--lookup-rank 1`) as macroxue does.

| player | mean score | 16384 | 32768 | 65536 | moves/s |
|---|---|---|---|---|---|
| macroxue binary, tables disabled | 134,018 | 10.0% | 0% | 0% | 5,300 (1 thread) |
| macroxue binary, Tuple10 + Tuple11 | 379,388 | 40.0% | 32.5% | 0% | 5,200 (1 thread) |
| macroxue published, same binary, 1000 games | 493,058 | 81.4% | 52.5% | 1.3% | 6,461 |
| g2048 `--eval mx`, no tables | 155,403 | 12.5% | 0% | 0% | 4,820 (4 threads) |
| g2048 `--eval mx --tables`, block10 + five7, tables built during the run | 413,178 | 77.5% | 37.5% | 0% | 1,594 (4 threads, solving included) |
| g2048 same, tables loaded from disk | 383,615 | 72.5% | 37.5% | 0% | 12,879 (4 threads) |

So the port does what the original does: the same evaluation and search jump from no
32768 at all to one game in three, and the best games (839,920 and 819,680 with 32768)
are of the kind that go on to 65536 with more games. 65536 itself was not seen in 40
games; macroxue needs about 80 games per 65536 at this depth. With the tables on disk
the player is 2.7x faster than without them (12,879 against 4,820 moves/s), because a
table move costs a few lookups where a searched move costs thousands of evaluations;
53% of all moves came from a table. The two with-table runs differ a little (413k and
384k) because entries solved in another order break ties differently.

Table building: block10 solved 369M positions and five7 286M during the 40 games (about
3M positions/s per thread); on disk 3.2 GB and 1.9 GB (sparse chunks), 4.2 GB and 2.4 GB
of address space in memory. Later runs load the files and solve only what is new.

**What this does and does not show.** It shows the tables are correct and are what
makes macroxue's endgame work. It does not show the 65536 rate of the n-tuple net with
tables: the net is not in the repository and nothing here plays like it. The expected
effect on the net is bounded by two things that only the real measurement can settle:
whether the net's corner shape is one of the three layouts (`formation` tells), and how
often a game that has made 32768 reaches a position the tables cover.

## 5. How to verify on the real net

Build once; the first table-building run is slow (minutes of solving per new region) and
needs about 7 GB of RAM on top of the net; later runs load the tables.

```sh
cargo build --release
B=./target/release/g2048; NET=nets/otd-stage2.bin

# 1. The fixed board set: 1000 boards where 32768 first appears, 2-ply, seeds fixed by the command.
$B positions pos32k.txt 1000 --net $NET --depth 2 --rank 15

# 2. Does the net's corner shape fit the tables at all? Expect most boards under block10 or snake8.
$B formation pos32k.txt

# 3. Baseline, unchanged code path (depth 2): expect 65536 0.0%, about 10,000 average moves.
$B endgame pos32k.txt --net $NET --depth 2

# 4. Tables once a 32768 is on the board (default --lookup-rank 15; the opening is unchanged).
$B endgame pos32k.txt --net $NET --depth 2 --tables tables

# 5. Tables throughout, as macroxue plays (also changes the opening).
$B endgame pos32k.txt --net $NET --depth 2 --tables tables --lookup-rank 1

# 6. If formation shows a snake, add the snake table.
$B endgame pos32k.txt --net $NET --depth 2 --tables tables --layouts block10,five7,snake8

# 7. Full games, same seeds as before, with and without.
$B bench 600 201 --net $NET --depth 2
$B bench 600 201 --net $NET --depth 2 --tables tables
```

Each `endgame` run prints the 65536 rate, average moves survived and moves/s, then a
line `endgame tables: X of Y moves from a table` and the table sizes. The expectations:

- Step 4 should raise average moves survived from about 10,000 towards the 15,000 a
  second 32768 needs, and the table line should show a large share of moves answered
  once the block has six large tiles. If the share is near zero, the net does not form
  the block: go to step 6, and send me the `formation` output.
- The 65536 rate: macroxue converts 32768 boards at about 4% per attempt at depth 5 to 8.
  With the net's stronger opening and these tables at depth 2, anything from 1% up on
  1000 boards (10 or more) is a measurable gain over 0 of 300 so far; under 5 of 1000 is
  not distinguishable from no effect and the next step is depth 3 in the endgame
  (`--endgame-depth 3`) together with the tables.
- Speed: table moves are free; expect the moves/s of step 4 to be within a few percent
  of step 3 once the tables are built.

## 6. If the tables alone are not enough

The rest of macroxue's edge is depth (3.5% at depth 8 against 1.3% at depth 3) and the
larger five-anchor table (27 GB). Both apply on top of this branch without code changes:
`--endgame-depth N` and `--layouts block10,five8` once the memory is there (the layout
bound would need raising from 7 to 8 in `Layout::parse`; `five8` is 11 GB of address
space). A 65536 as a real tile needs the u128 board of `DESIGN.md` Release B on branch
`stage3`; nothing here depends on it, since the game ends at the merge that makes 65536.

## 7. Second round: hand the endgame to macroxue's player

Result of section 5 on the real net (2026-10-01): tables answered 1.2% of moves, 0 of
1000 boards reached 65536 with or without them, and at the first 32768 no layout fitted
(`block10` 0%, `five7` 1.4%, `snake8` 0%). The first-32768 board is uninformative, since
the chain that built the tile has just been consumed, but 1.2% of moves against about
70% in macroxue's own games says the net plays the endgame in a shape the tables never
see. The tables need a player that forms the block; the net has no drive to do so.

So the next experiment keeps the net for the part it is best at and hands the rest to
the player that is known to convert: `--endgame-eval mx` switches, from the first board
holding `--endgame-rank` (default 15), to macroxue's evaluation, no move reward, its loss
value, its spawn cut `1 / 2^(depth + 4)`, depth 3 (or `--endgame-depth`), and the tables
from the same board on. Before that board nothing changes. macroxue converts a 32768
board to 65536 about 2.5% of the time at depth 3 and about 4% at depth 5 to 8, and its
games enter that phase from the same "32768 plus leftovers" boards.

```sh
B=./target/release/g2048; NET=nets/otd-stage2.bin

# A. From the 1000 fixed 32768 boards (both handover ranks are the same here, since every
#    board already holds 32768): macroxue's player from move one, depth 3 and depth 4.
$B endgame pos32k.txt --net $NET --depth 2 --endgame-eval mx --endgame-depth 3 --tables tables
$B endgame pos32k.txt --net $NET --depth 2 --endgame-eval mx --endgame-depth 4 --tables tables

# B. Fresh games, same seeds as the 559,404 baseline: handover at 32768 and at 16384.
$B bench 600 201 --net $NET --depth 2
$B bench 600 201 --net $NET --depth 2 --endgame-eval mx --endgame-rank 15 --tables tables
$B bench 600 201 --net $NET --depth 2 --endgame-eval mx --endgame-rank 14 --tables tables
```

Expectations, from macroxue's measured rates:

| run | 65536 | other signs |
|---|---|---|
| A, depth 3 | about 25 of 1000 (2.5%); 10 or more is a clear gain over 0 of 1000 | table share of moves far above 1.2%; average moves survived up from 10,000 |
| A, depth 4 | about 30 of 1000; 13x slower per move than depth 3 | |
| B, handover at 32768 | 60% of games reach 32768, about 2.5% of those convert: 8 to 10 of 600 | mean score near the 559k baseline, since the opening is the net's |
| B, handover at 16384 | macroxue converts 16384 to 65536 about 1.6% of the time (1.3 of 81.4), the net reaches 16384 in about 95%: 8 to 10 of 600 | 32768 rate near 60% if their 16384 to 32768 conversion (64%) matches the net's; a lower mean score says the net was the better 16384 player |

If A gives 10 or more, 65536 is reached and the branch does what it was for. The two B
rows then say where the handover belongs: equal 65536 counts with a higher mean score
for the rank-15 run means hand over as late as possible. If A gives under 5, macroxue's
player does not convert from the net's boards either, and the remaining lever is depth
(`--endgame-depth 5` is their 2.7% to 3.3% range, at 500 moves/s per thread).

Speed in the handed-over phase is about 5,000 moves/s per thread at depth 3 with the
tables loaded; run A is 10 to 15 million moves, a few minutes on 8 threads. The tables
grow as new regions come up and are saved at the end of every run.

## 8. Results of the handover, and the bigger five-anchor table

Fresh games, `bench 600 201`, same seeds throughout (2026-10-01):

| 600 fresh games | 65536 | 32768 | mean score | time |
|---|---|---|---|---|
| net alone, depth 2 | 0 | 59.8% | 559,404 | 14 min |
| macroxue from 32768, depth 4 | 7 | | 565,099 | 13 min |
| macroxue from 32768, depth 5 | 8 (1.3%) | | 575,216 | 19 min |
| macroxue from 16384, depth 3 | 2 | 41.5% | 451k | |
| macroxue from 16384, depth 5 | running | | | |

Caveat on the depth-4 and depth-5 rows: until 18512ca, `--endgame-depth` also raised the
net's own depth from 16384 on (its older meaning), so those runs played 16384 to 32768
with the net at depth 4 or 5, not at depth 2. Their 65536 counts are still macroxue's
conversions from the handover point, but the mean scores and the time include a deeper
net phase, and the rows should be re-run after the fix before being compared with later
settings.

65536 is reached: about one fresh game in 80, from none. Given a 32768 the conversion
is about 2.2% (8 of roughly 360), against 3.6% for macroxue's own program at depth 5
from boards it shaped itself. The 16384 handover at depth 3 shows the transfer penalty
directly: macroxue converts its own 16384 boards to 32768 64% of the time at that depth,
the net's boards only 41.5%.

**Levers left**, in order of expected effect:

1. The bigger five-anchor table, macroxue's one documented doubling (2019 to 2022 only
   `Tuple11` grew, cap 7 to 9, and depth-5 conversion went 2.0% to 3.6%). `five8` and
   `five9` now parse: the five and snake layouts place their canonical anchors at 15
   downwards whatever the cap, as macroxue does, so the existing `five7` and `snake8`
   files stay valid. Memory of address space, of which about three quarters get touched:
   `five7` 2.4 GB, `five8` 11 GB (about 9 GB resident), `five9` 44 GB (about 33 GB). With
   the laptop at 13.5 GB and OCI at 23.4 GB less 7 GB for the coordinator, `five8` fits
   on either with training stopped on the laptop; `five9` fits nowhere here.

   ```sh
   # conversion from the fixed boards, then fresh games; first run builds five8 (minutes)
   $B endgame pos32k.txt --net $NET --depth 2 --endgame-eval mx --endgame-depth 5 --tables tables --layouts block10,five8
   $B bench 600 201 --net $NET --depth 2 --endgame-eval mx --endgame-rank 15 --endgame-depth 5 --tables tables --layouts block10,five8
   ```

   Expected: half of macroxue's doubling, so conversion from about 2.2% towards 3%, 11 or
   more of 600 fresh games. Keep `five7` out of the layout list when `five8` is in: the
   larger table covers the smaller one's boards.
2. Earlier handover at depth 5 (running). Depth 3 says the net is the better 16384
   player at equal depth; depth 5 starts 14 points higher for macroxue, so it may break
   even on 32768 and gain on shape.
3. Depth 6 or 7: about 15% relative per ply at 3x the time each.

Measuring: 7 against 8 cannot be read. For conversions use `pos32k.txt` (1000 attempts);
for the overall rate and mean score, the fresh-game bench, and 1200 games (`bench 1200
201`) once two settings are within a few counts of each other.
