#!/bin/sh
# Lever 3: depth-6 conversion on the 1000 fixed boards, block10+five8, --pass-score big.
# Sharp signal vs the depth-5 number (27/1000). ~3x depth 5 in time.
cd ~/2048-endgame
L=depth6.log
tmux send-keys -t mcfabric "save-all flush" Enter 2>/dev/null; sleep 10
tmux send-keys -t mcfabric "stop" Enter 2>/dev/null; sleep 12
sudo systemctl stop minecraft-fabric.service g2048-worker.service 2>/dev/null
{ echo "== pos32k endgame-depth 6, block10,five8, --pass-score big on OCI"; date +%T; free -g | head -2
  ./g2048-pass endgame pos32k.txt --net ../2048-solver/nets/otd-stage2.bin --depth 2 --endgame-eval mx --endgame-depth 6 --tables tables --layouts block10,five8 --pass-score big
  echo "== DONE"; date +%T; } > $L 2>&1
sudo systemctl start g2048-worker.service
sudo systemctl restart minecraft-fabric.service
sleep 40; pgrep -af fabric-server-launch >> $L; systemctl is-active g2048-worker >> $L
