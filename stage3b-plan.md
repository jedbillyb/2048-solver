Stage 3b (build 10, started 2026-10-01 12:52 NZDT). Baseline restart "from 32768 -> 65536" = ~15% (flat all of old stage 3).
Hourly: farm status. Note the ALL "-> 65536" restart rate and progress. Speak only if something changed meaningfully (rate trend, machine offline, stall) - else one line.
When progress hits 100%:
1. OCI snapshot: cp nets/otd.bin{,.seq,.episodes,.job,.pool} -> nets/otd-stage3b.bin* only if otd.bin mtime unchanged across the copy; md5.
2. rsync --partial to laptop nets/otd-stage3b.bin, verify md5.
3. farm stop (laptop worker); bench 600 201 --net nets/otd-stage3b.bin --depth 2 > nets/bench-otd-stage3b-d2-s201.out; farm start.
   Compare (seeds 201-800): stage2 59.8% / 559,404; stage2-final 58.7% / 561,042; old stage3 58.3% / 545,621. Also report 65536 rate.
4. Memory project_2048_ai.md + PushNotification with the result. Then CronDelete the hourly job.
Do NOT start stage 4 (needs Release B, u128).

ALSO (endgame-65536 verify, started 2026-10-01 18:31): detached /mnt/shared/projects/2048-ai-endgame/verify.sh writes verify.log; "== DONE" at the end, then it runs `farm start`. When DONE: report baseline vs tables 65536 rate + avg moves + table share + formation summary; PushNotification. If verify.sh is gone without DONE, make sure the laptop worker is running (farm start).

ALSO (handover runs, started 2026-10-01 ~20:40): detached /mnt/shared/projects/2048-ai-endgame/handover.sh writes handover.log (A d3, B rank15, B rank14, A d4, "== DONE", then farm start). Report each new section's result once (65536 count, mean, table share). Compare B against baseline 559,404 / 59.8% 32768. If handover.sh is gone without DONE, run farm start.

ALSO (deeper runs, started 2026-10-01 ~21:00): detached /mnt/shared/projects/2048-ai-endgame/deeper.sh (A d5 done: 19/1000; bench killed, depth bug) -> now deeper2.sh writes deeper2.log (A d5 on pos32k, B 600 201 rank15 d4, "== DONE", then farm start). Compare: A d4 15/1000, d3 8/1000; B rank15 d3 2/600 mean 531,798. If deeper.sh gone without DONE, farm start.

ALSO (d5 bench, started 2026-10-01 ~22:05): deeper3.sh writes deeper3.log (600 201 rank15 mx d5, then farm start). Compare d4: 7/600, mean 565,099; net alone 0, 559,404.

ALSO (rank14 d5 bench, started 2026-10-01 ~22:50): deeper4.sh writes deeper4.log, then farm start. Compare rank15 d5: 59.8% 32768, 8/600, 575,216; rank14 d3 was 41.5%, 2/600, 451k.
- deeper5.sh: waits for deeper4, then pos32k d5 --layouts five8 (laptop worker stopped, farm start at end). Log deeper5.log.
- OCI five8.sh (~/2048-endgame, log five8.log): MC + g2048-worker stopped, pos32k d5 block10,five8. Script restarts both at end; if it dies, run sudo systemctl start g2048-worker; sudo systemctl restart minecraft-fabric.service
- deeper6.sh: pos32k d5 block10,five7 --pass-score small (uncommitted ai.rs change). Log deeper6.log.
