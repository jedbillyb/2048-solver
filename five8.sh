#!/bin/sh
cd ~/2048-endgame
L=five8.log
tmux send-keys -t mcfabric "save-all flush" Enter; sleep 15
sudo systemctl stop minecraft-fabric.service g2048-worker.service
{ echo "== A d5 block10+five8 on OCI"; date +%T; free -g | head -2
  ./target/release/g2048 endgame pos32k.txt --net ../2048-solver/nets/otd-stage2.bin --depth 2 --endgame-eval mx --endgame-depth 5 --tables tables --layouts block10,five8
  echo "== DONE"; date +%T; } > $L 2>&1
sudo systemctl start g2048-worker.service
sudo systemctl restart minecraft-fabric.service
sleep 40; pgrep -af fabric-server-launch >> $L; systemctl is-active g2048-worker >> $L
