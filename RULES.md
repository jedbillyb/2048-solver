# play2048.co rules (extracted from site JS, 2026-09-25)

Source: the site's JS bundle (`/assets/MessageBroker-*.js`), not redistributed here.

## Core game (same in every mode)
- 4x4 board, starts with 2 random tiles.
- Spawn after every move that changes the board: a uniformly random empty cell,
  value 2 with probability 0.9, else 4 (`t.float()<.9?2:4`).
- Slide: a tile that was produced by a merge this move cannot merge again.
  Score increases by the value of each new merged tile.
- Win state triggers on the first 2048; play can continue.
- RNG is seeded per game (`_rng:{seed,...}` stored in the gameplay state).

## Modes
- `classic`: no power-ups. Target for the bot first.
- `standard`: power-ups. Creating a tile of value 128 / 256 / 512 by merging
  grants +1 use to the power-up in that tier with the fewest uses left (cap 2 each).

| Earned at | Power-up | Start uses |
|---|---|---|
| 128 | undo | 2 |
| 256 | swapTwoTiles | 1 |
| 256 | teleportTileToEmptyCell | 1 |
| 256 | rotateOuterRingOfBoard | 1 |
| 256 | mergeAnyTwoAdjacentTiles | 0 |
| 512 | removeTilesByValue | 0 |
| 512 | bomb | 0 |
