#!/bin/sh
# Runs after five8.sh: pos32k d5 block10,five8 with --pass-score big (macroxue BIG_TUPLES config).
cd ~/2048-endgame
while pgrep -f five8.sh >/dev/null || [ ! -x g2048-pass ]; do sleep 60; done
L=passbig.log
sudo systemctl stop minecraft-fabric.service g2048-worker.service
{ echo "== A d5 block10,five8 --pass-score big on OCI"; date +%T
  ./g2048-pass endgame pos32k.txt --net ../2048-solver/nets/otd-stage2.bin --depth 2 --endgame-eval mx --endgame-depth 5 --tables tables --layouts block10,five8 --pass-score big
  echo "== DONE"; date +%T; } > $L 2>&1
sudo systemctl start g2048-worker.service
sudo systemctl restart minecraft-fabric.service
sleep 40; pgrep -af fabric-server-launch >> $L; systemctl is-active g2048-worker >> $L
