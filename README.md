# 2048-solver

An AI aiming for the highest possible score on [play2048.co](https://play2048.co).

## Status
- [x] Rules extracted from the live site's JS (`RULES.md`)
- [x] Rust bitboard engine matching those rules
- [x] Expectimax with a hand-tuned heuristic (milestone 1)
- [x] TD-trained n-tuple network (4 six-tuples x 8 symmetries, afterstate TD(0))
- [ ] Browser bot that plays play2048.co (Classic mode)

## Run
```
cargo test --release
cargo run --release -- train nets/main.bin 3000000          # self-play TD training, saves every ~100k games
cargo run --release -- bench 16 --net nets/main.bin --depth 2 # expectimax on top of the net
cargo run --release -- bench 16                               # hand-heuristic expectimax
cargo run --release -- serve --net nets/main.bin --depth 3    # move server for the browser bot (127.0.0.1:20480)
```
`bench` plays games in parallel and prints mean/median score and how often each tile was reached.
Weights live in `nets/` (gitignored, ~270 MB each).

## Results so far
| Player | Games | Mean score | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| Net after 100k training games, 1-ply greedy | last 10k | 65,493 | 67.5% | 10.9% | 0% |
| Same net + expectimax depth 2 | 8 | 125,954 | 100% | 62.5% | 0% |
| Net after 3M games, 1-ply greedy | last 10k | 162,533 | 94.1% | 78.1% | 24.8% |
| Same net + expectimax depth 2 | 32 | 298,250 | 100% | 100% | 81.2% |
| 3-stage net, +1.7M games with endgame restarts, depth 2 | 32 | 336,879 | 100% | 100% | 93.8% |
| Same, depth 3 | 8 | 304,457 | 100% | 100% | 75.0% |

32768 column is 0% everywhere so far: games top out around 360-386k, just before the
final merge chain. Next: the 8-tuple network (`--tuples 8`), then stages + restarts on top.
