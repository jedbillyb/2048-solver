# 2048-solver

An AI aiming for the highest possible score on [play2048.co](https://play2048.co), Classic
rules only (no undo, no power-ups).

## Goal
The 131072 tile, the largest a 4x4 board can hold. Nobody, human or AI, has reported it
without undo. Milestones on the way, with the records to beat:

| Milestone | Record to beat |
|---|---|
| 32768 in most games | 72% (Guei et al. 2021, 6-ply search) |
| 65536 | best human 1,314,064 (2048verse, 2023) |
| Beat the AI record | 1,704,908 ([macroxue/2048-ai](https://github.com/macroxue/2048-ai)) |
| 131072 | theoretical max score 3,932,100 |

## Status
- [x] Rules extracted from the live site's JS (`RULES.md`)
- [x] Rust bitboard engine matching those rules
- [x] Expectimax with a hand-tuned heuristic (milestone 1)
- [x] TD-trained n-tuple network (afterstate TD(0), 8 board symmetries)
- [x] Distributed training farm (coordinator + workers on Linux and Windows)
- [x] OTD stage 1: Matsuzaki 8x6-tuple net, 100M games (finished 2026-09-28; the TC phase
      broke it, so stage 1 is the pre-TC net, see below)
- [ ] OTD stage 2: net for boards that already hold 16384 (running)
- [ ] Deep search test (6-ply with tile downgrading)
- [ ] Browser bot that plays play2048.co

## Run
```
cargo test --release
cargo run --release -- train nets/main.bin 3000000 --tuples 8  # self-play TD training, saves every ~100k games
cargo run --release -- bench 16 --net nets/main.bin --depth 2   # expectimax on top of the net
cargo run --release -- bench 16                                 # hand-heuristic expectimax
cargo run --release -- serve --net nets/main.bin --depth 3      # move server for the browser bot (127.0.0.1:20480)
```
`bench` plays games in parallel and prints mean/median score and how often each tile was reached.
Weights live in `nets/` (gitignored; 512 MB for the 8-tuple net).

## Training on several machines

`g2048 coord` holds the master net on an always-on server; `g2048 worker` runs on any
number of machines (Linux or Windows), trains its own copy in 2-minute chunks, uploads
only the largest weight changes (capped at `send_mb`, the rest carries over to the next
chunk) and pulls everyone else's. Syncing runs in the background while the next chunk
trains. Workers speak HTTPS through the system `curl`.

```sh
g2048 coord --net nets/otd.bin --token-file ~/.config/g2048/token   # server, behind nginx /g2048/
g2048 worker --url https://HOST/g2048 --token-file TOKEN_FILE       # each machine
```

- **Learning schedule:** with `schedule=otd` in the job, the coordinator follows the OTD
  recipe over `goal` games: alpha 0.1, cut 10x at 50% and 75%. The paper's final TC phase
  is left out on purpose: TC accumulators live on each worker, start at zero and reset on
  every restart, so each weight's first TC step runs at the full rate. When stage 1 hit
  90% on 2026-09-27 that took the farm's mean score from 302k to 35k within an hour.
- **Stages:** `freeze=N` keeps stages below N fixed. Stage 2 runs with `freeze=1
  restart=0.9`: 90% of games start from a saved board where 16384 first appeared and only
  the stage-2 weights learn; the other 10% are fresh games that keep the pool of starting
  boards varied.
- **Auto-update:** a worker runs under a small supervisor. Bump `const BUILD` in
  `src/dist.rs`, publish the new binaries, then `farm set version=N`; every worker saves
  its net, updates (Windows downloads the new exe) and restarts on its own.
- **Per machine:** `threads.NAME=N`, `priority.NAME=<Windows class>` and `stop.NAME=1` in
  the job.
- **Coordinator restarts:** `POST /shutdown` saves the net and the delta log, then exits
  so systemd starts the new binary; workers carry on instead of downloading the net again.
- **Status:** progress and ETA, the best game so far, per-machine CPU, temperature and
  RAM, and results (games/s, mean score, tile rates, best score).
- **Speed:** on Linux the weights sit on 2 MB huge pages (about 4% faster).

`farm/farm.sh` controls the farm from any machine with the token:
```
farm watch | status               live overview / once
farm dell                         admin PowerShell command: installs a Windows worker as a boot task
farm job | set k=v                show / change the job
farm pause | resume               every machine
farm limit NAME N | full NAME     thread cap for one machine
farm kill NAME                    stop one machine's worker remotely
farm start | stop                 this laptop's own worker
```

## Results so far
| Player | Games | Mean score | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|
| Net after 100k training games, 1-ply greedy | last 10k | 65,493 | 67.5% | 10.9% | 0% | 0% |
| Same net + expectimax depth 2 | 8 | 125,954 | 100% | 62.5% | 0% | 0% |
| Net after 3M games, 1-ply greedy | last 10k | 162,533 | 94.1% | 78.1% | 24.8% | 0% |
| Same net + expectimax depth 2 | 32 | 298,250 | 100% | 100% | 81.2% | 0% |
| 3-stage net, +1.7M games with endgame restarts, depth 2 | 32 | 336,879 | 100% | 100% | 93.8% | 0% |
| Same, depth 3 | 8 | 304,457 | 100% | 100% | 75.0% | 0% |
| OTD 8-tuple net at 51.6M games (farm), 1-ply greedy | recent chunks | 288,598 | 98.2% | 93.2% | 74.2% | 0.05% |
| OTD stage 1, pre-TC (~80M games), 1-ply greedy | 200 | - | - | 98.5% | 92.0% | 0% |
| Same, expectimax depth 2 | 200 | 334,883 | 100% | 99.5% | 95.5% | 0% |
| OTD stage 2 at 7.7M games (farm), 1-ply greedy, fresh games | recent chunks | 291,383 | 98.5% | 94.7% | 79.7% | 1.41% |
| OTD stage 1 after the TC phase (100M games), 1-ply greedy | 200 | - | - | 0% | 0% | 0% |

Best single game so far: **805,224 with the 32768 tile** (stage 2 farm training game on
the laptop, 1-ply greedy, no search, 2026-09-28).

Stage 1 alone stops at 16384: fresh games almost never get further, which is what stage 2
is for.
