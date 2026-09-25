# 2048-ai

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
```
`bench` plays games in parallel and prints mean/median score and how often each tile was reached.
Weights live in `nets/` (gitignored, ~270 MB each).

## Results so far
| Player | Games | Mean score | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| Net after 100k training games, 1-ply greedy | last 10k | 65,493 | 67.5% | 10.9% | 0% |
| Same net + expectimax depth 2 | 8 | 125,954 | 100% | 62.5% | 0% |
