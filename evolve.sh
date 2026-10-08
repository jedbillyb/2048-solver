#!/bin/bash
# ============================================================================
# evolve.sh - 24/7 autonomous top-stage evolution for the 2048 net.
#
# Goal: push the 65536 -> 131072 regime by sharpening ONLY the top stage
# (stage 3) while the lower stages stay byte-frozen (--freeze 3), so the early
# game can never regress (catastrophic forgetting is impossible by construction).
#
# Each generation: resume from the champion, train the top stage for GAMES
# games, evaluate the result on a fixed 32768 board set, and promote it to
# champion only if it reaches 65536/131072 more often. Pure hill-climb.
#
# PRODUCTION SAFETY (hard rule): this box runs the nz-vehicle-finder API.
#   - training is niced (15) and pinned to cores 2,3 so cores 0,1 + nice-0
#     production always preempt.
#   - a RAM-floor watchdog KILLS the training child the moment free RAM drops
#     below RAM_FLOOR_MB, and the loop waits for RAM to recover. A 2048 job must
#     NEVER starve the API into swap again.
#
# Tunables live in evolve-params.env (re-read every generation) so Claude can
# change strategy between check-ins without restarting the loop.
# ============================================================================
set -u

ROOT=/home/ubuntu/2048-endgame
NETS=/home/ubuntu/2048-solver/nets
BIN=$ROOT/target/release/g2048
SEEDNET=$NETS/otd-stage4-bench.bin        # read-only clean 4-stage baseline
CHAMP=$NETS/otd-evolve-champion.bin       # best net found so far
WORK=$NETS/otd-evolve-work.bin            # current generation output
EVALSET=$ROOT/pos32k-eval.txt             # fixed board set for the fast metric
FULLSET=$ROOT/pos32k.txt
PARAMS=$ROOT/evolve-params.env
CSV=$NETS/evolve-log.csv
DLOG=$NETS/evolve-decisions.log
STATE=$NETS/evolve-state.env              # champion metric cache + gen counter

log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$DLOG"; }

# --- one-time housekeeping: reclaim dead experiment nets so there is room -----
# (broken / regressed / superseded-by-final / concluded-experiment outputs; all
#  regenerable, none referenced by any service. otd.bin and the -final archives
#  are kept.)
reclaim(){
  cd "$NETS" || return
  for f in otd-stage1-tc-broken.bin otd-stage2.bin otd-stage3.bin \
           otd-stage3-overrun.bin otd-stage3b.bin otd-tc-experiment.bin \
           otd-topstage.bin; do
    [ -f "$f" ] && { rm -f "$f" && log "reclaimed $f"; }
  done
}

# --- free-RAM watchdog: returns only when free RAM is above the floor ---------
wait_for_ram(){
  local floor=$1
  while :; do
    local avail; avail=$(free -m | awk '/Mem:/{print $7}')
    [ "$avail" -ge "$floor" ] && return 0
    log "RAM $avail MB < floor $floor MB - holding (production first)"
    sleep 30
  done
}

# --- score a net on the eval board set: prints "p65 p131 avgmoves" ------------
metric(){
  local net=$1 set=$2
  taskset -c 2,3 nice -n 15 "$BIN" endgame "$set" --net "$net" 2>/dev/null \
    | awk '/boards/{for(i=1;i<=NF;i++){if($i=="65536")p65=$(i+1); if($i=="131072")p131=$(i+1); if($i=="moves")mv=$(i+1)} gsub("%","",p65); gsub("%","",p131); print p65+0, p131+0, mv+0}'
}

# combined climb score: 131072 dominates, then 65536 (scaled so it never ties out)
score(){ awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f", a*1000 + b}'; }

# --- run ONE training generation into $WORK, with the RAM watchdog ------------
# returns 0 if it finished, 1 if it was killed for RAM / died.
train_gen(){
  local games=$1 freeze=$2 rp=$3 rs=$4 alpha=$5 seed=$6 floor=$7
  local alphaflag=""; [ "$alpha" != "auto" ] && alphaflag="--alpha $alpha"
  taskset -c 2,3 nice -n 15 "$BIN" train "$WORK" "$games" \
      --resume "$CHAMP" --stages 4 --freeze "$freeze" \
      --restart "$rp" --restart-stage "$rs" $alphaflag --seed "$seed" \
      >> "$NETS/evolve-train.log" 2>&1 &
  local tp=$!
  # watchdog: kill the trainer if free RAM dips under the floor
  while kill -0 "$tp" 2>/dev/null; do
    local avail; avail=$(free -m | awk '/Mem:/{print $7}')
    if [ "$avail" -lt "$floor" ]; then
      log "RAM $avail MB < floor $floor MB DURING train - killing gen (production first)"
      kill "$tp" 2>/dev/null; sleep 2; kill -9 "$tp" 2>/dev/null
      wait "$tp" 2>/dev/null
      return 1
    fi
    sleep 10
  done
  wait "$tp"; return $?
}

# ============================== bootstrap ====================================
reclaim
[ -f "$EVALSET" ] || head -n 160 "$FULLSET" > "$EVALSET"
[ -f "$CSV" ] || echo "ts,gen,games,freeze,restart_p,restart_stage,alpha,seed,p65,p131,avgmoves,result" > "$CSV"

if [ ! -f "$CHAMP" ]; then
  log "no champion yet - seeding from $(basename "$SEEDNET")"
  cp "$SEEDNET" "$CHAMP" || { log "FATAL: could not copy seed (disk?)"; exit 1; }
fi

# champion metric cache
if [ -f "$STATE" ]; then source "$STATE"; else
  read CP65 CP131 CPMV < <(metric "$CHAMP" "$EVALSET")
  GEN=0
  { echo "CP65=$CP65"; echo "CP131=$CP131"; echo "CPMV=$CPMV"; echo "GEN=$GEN"; } > "$STATE"
  log "champion baseline: 65536 ${CP65}%  131072 ${CP131}%  avgmoves ${CPMV}"
fi
CBEST=$(score "$CP131" "$CP65")

log "=== evolve loop armed (pid $$) champion score $CBEST ==="

# =============================== main loop ===================================
while :; do
  # defaults; evolve-params.env overrides any of these each generation
  ENABLED=1; GAMES=4000000; FREEZE=3; RESTART_P=0.6; RESTART_STAGE=2
  ALPHA=auto; GEN_SLEEP=5; RAM_FLOOR_MB=3000
  [ -f "$PARAMS" ] && source "$PARAMS"

  if [ "$ENABLED" != 1 ]; then log "disabled via params - idling 60s"; sleep 60; continue; fi

  wait_for_ram "$RAM_FLOOR_MB"

  GEN=$((GEN+1)); SEED=$((1000+GEN))
  log "gen $GEN start: games=$GAMES freeze=$FREEZE restart=$RESTART_P/stage$RESTART_STAGE alpha=$ALPHA seed=$SEED"
  t0=$(date +%s)
  if ! train_gen "$GAMES" "$FREEZE" "$RESTART_P" "$RESTART_STAGE" "$ALPHA" "$SEED" "$RAM_FLOOR_MB"; then
    log "gen $GEN aborted (RAM/err) - cooling 120s"; rm -f "$WORK"; sleep 120; continue
  fi
  secs=$(( $(date +%s) - t0 ))

  read W65 W131 WMV < <(metric "$WORK" "$EVALSET")
  WSCORE=$(score "$W131" "$W65")
  log "gen $GEN done ${secs}s: work 65536 ${W65}%  131072 ${W131}%  avgmoves ${WMV}  (champ 65536 ${CP65}% 131072 ${CP131}%)"

  if awk -v w="$WSCORE" -v c="$CBEST" 'BEGIN{exit !(w>c)}'; then
    mv -f "$WORK" "$CHAMP"
    CP65=$W65; CP131=$W131; CPMV=$WMV; CBEST=$WSCORE
    { echo "CP65=$CP65"; echo "CP131=$CP131"; echo "CPMV=$CPMV"; echo "GEN=$GEN"; } > "$STATE"
    log "gen $GEN PROMOTED -> champion  65536 ${CP65}%  131072 ${CP131}%"
    echo "$(date -u +%FT%TZ),$GEN,$GAMES,$FREEZE,$RESTART_P,$RESTART_STAGE,$ALPHA,$SEED,$W65,$W131,$WMV,PROMOTED" >> "$CSV"
  else
    rm -f "$WORK"
    { echo "CP65=$CP65"; echo "CP131=$CP131"; echo "CPMV=$CPMV"; echo "GEN=$GEN"; } > "$STATE"
    log "gen $GEN kept champion (no improvement)"
    echo "$(date -u +%FT%TZ),$GEN,$GAMES,$FREEZE,$RESTART_P,$RESTART_STAGE,$ALPHA,$SEED,$W65,$W131,$WMV,kept" >> "$CSV"
  fi
  sleep "$GEN_SLEEP"
done
