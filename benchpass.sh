#!/bin/sh
# 600-game full-game confirmation: best setting (rank14 d5 mx, block10+five8) plus --pass-score big.
cd ~/2048-endgame
L=benchpass.log
tmux send-keys -t mcfabric "save-all flush" Enter 2>/dev/null; sleep 10
sudo systemctl stop minecraft-fabric.service g2048-worker.service
{ echo "== bench 600 201 rank14 d5 block10,five8 --pass-score big on OCI"; date +%T; free -g | head -2
  ./g2048-pass bench 600 201 --net ../2048-solver/nets/otd-stage2.bin --endgame-eval mx --endgame-rank 14 --endgame-depth 5 --tables tables --layouts block10,five8 --pass-score big
  echo "== DONE"; date +%T; } > $L 2>&1
sudo systemctl start g2048-worker.service
sudo systemctl restart minecraft-fabric.service
sleep 40; pgrep -af fabric-server-launch >> $L; systemctl is-active g2048-worker >> $L
