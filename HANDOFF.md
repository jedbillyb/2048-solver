# Handoff from the laptop Claude session (2026-10-01 night)

User: jed. Rules: short replies, adhd format (action first, numbered steps, max 5 items, end with one next action), no em/en dashes or "--" in prose. Never start stage 4 (needs Release B, u128). Commits here are signed automatically with OCI-only subkey 1F1786AA9A2785B0 (no passphrase). Always push after committing. Never add Co-Authored-By lines.

## Where things stand
Goal: 65536 then 131072. Branch endgame-65536 (this worktree, ~/2048-endgame). Net = ../2048-solver/nets/otd-stage2.bin.
Handover to macroxue player: `--endgame-eval mx --endgame-rank R --endgame-depth N --tables tables`.

Results, 1000 fixed first-32768 boards (pos32k.txt), handover d5, block10+five7: **19/1000 reach 65536** (d4 15, d3 8). This is the baseline.
600 fresh games seeds 201-800 (bench 600 201): rank15 d5 59.8% 32768, 8/600 65536, mean 575k. **rank14 d5 65.3%, 8/600, mean 598k = best full-game setting.**

## Running tonight (both scripts restart Minecraft + g2048-worker themselves)
1. five8.sh -> five8.log: pos32k d5 with --layouts block10,five8 (bigger table, Fable 3cef163). Win = 30+/1000.
2. passbig.sh -> passbig.log: starts after five8.sh exits; same plus `--pass-score big` (macroxue pass_score cut, ported in src/ai.rs + src/main.rs, UNCOMMITTED, same change sits uncommitted in the laptop worktree /mnt/shared/projects/2048-ai-endgame). Binary g2048-pass.
If a script dies: `sudo systemctl start g2048-worker; sudo systemctl restart minecraft-fabric.service` (never systemctl start for MC, it is a silent no-op). Verify with `pgrep -af fabric-server-launch`.

## Decide after results
- five8 >= 30/1000: table coverage was the limit; keep five8 (needs ~12 GB with block10, only OCI with MC stopped). five9 needs ~40 GB, no machine has it.
- five8 ~ 19 and pass-score big clearly higher: pass_score was the missing piece; confirm with bench 600 201 --endgame-rank 14 --endgame-depth 5 ... --pass-score big (needs MC stopped again; ask user first).
- Both ~19: report; next lever is depth 6.

## Stage 3b training
Farm: `curl`/coordinator g2048-coord on this box. At ~69% at 10:30 UTC, 32768->65536 restart rate flat 15.4%. Finish steps at 100% are in stage3b-plan.md (written for the laptop; steps 2-3 need the laptop, so only do step 1 snapshot here and leave the rest for the user).
