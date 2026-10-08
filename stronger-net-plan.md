# Path to 131072: stronger-evaluation plan (2026-10-05)

## Where we actually are
Five tuning levers are exhausted (all flat on the 65536 reach, ~1 to 1.8%; 131072 = 0.00%):
training volume, endgame-depth, endgame-rank, endgame-table growth, main-search-depth.
Root cause: reaching 65536 is gated by the **net's mid-game evaluation quality**, which has
plateaued. The endgame levers only improve survival after 16384, never the reach to 65536.

## Key correction from reading the code
The net already uses **Matsuzaki's 8x6-tuple set** (ntuple.rs:18), the exact architecture
behind the best published n-tuple results (Guei et al. 2021). So "bigger/different architecture"
is NOT an obvious free win, and the one clear upgrade (7-cell tuples) is RAM-prohibitive here:
`TUPLE_SIZE` is hardcoded `1<<24` (16^6); 7-cell = 16^7 = 16x bigger = ~1GB per tuple per stage,
so an 8-tuple 4-stage 7-cell net needs ~32GB on a 23GB box. Not viable on OCI.

## Remaining honest levers, cheapest first

### 1. Temporal-coherence (TC) learning is currently OFF  [CHEAP, UNTRIED HERE]
The live farm job has `tc=0`. TC adapts a per-weight learning rate from each weight's error
history and is a well-known quality/convergence win for n-tuple TD. Flag exists (`--tc 1`,
job key `tc=1`). This is the cheapest untried thing and runs on the existing farm.
- Risk: unknown why it is off; may have been a deliberate choice. Test on a fresh net or a
  forked snapshot, not the live otd.bin, so a bad run cannot regress production.

### 2. More stages (finer high-tile specialization)  [CHEAP]
Net is 4 stages split at 16384/32768/65536 (ntuple.rs FIRST_STAGE_RANK=14). More stages give
high-tile boards their own specialized weights. `--stages N` supported; `g2048 stages FILE N`
grows a saved net. Cheap to try, composes with TC.

### 3. Far more training  [SLOW on this farm]
Strongest published runs use billions of self-play games; we are at ~514M total. But the farm
is 2 small machines (OCI + optiplex; jed-xps offline), and training is flat on 65536 at current
settings, so raw volume alone is the weakest bet UNLESS paired with TC (#1) or more stages (#2).

### 4. Bigger 7-cell tuples  [NOT VIABLE on OCI]
~32GB RAM. Would need a bigger box. Park unless hardware changes.

## Suggested sequence (if you choose to pursue 131072)
1. Fork the current net snapshot. Train a short run with `tc=1` (and optionally `--stages 5+`)
   on OCI only, niced 15, farm otherwise paused. Watch the self-play 65536 rate.
2. If the net's self-play reach lifts above today's flat 0% (farm) / ~1.8% (bench), commit to a
   full retrain. If still flat, TC is not the lever and 131072 is realistically out of reach on
   this hardware with n-tuples; accept 65536 ~1.8% as the ceiling.
3. Any promising net: bench max-perf (rank 14, endgame-depth 6, block10) and look for the first
   131072 sightings and a 65536 rate meaningfully above 1.8%.

## Honest expectation
Even for SOTA programs the 131072 tile is extremely rare. These levers are incremental and may
move the ceiling only modestly. TC (#1) is the single highest-value / lowest-cost test and is
where I would start.
