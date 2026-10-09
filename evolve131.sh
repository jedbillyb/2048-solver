#!/bin/bash
# ============================================================================
# evolve131.sh - 24/7 engine aimed straight at the 131072 TILE.
#
# Strategy (user steer 2026-10-08 "whatever gets to 131, ok with the 68k rate
# for now"): the ~1.8% rate of REACHING 65536 is accepted as-is. The goal is to
# CONVERT a 65536 board into 131072. Self-play almost never reaches a 65536
# board, so stage 3 (the 65536->131072 regime) has never been trained. This
# engine fixes that:
#
#   Phase 1 (grow):  generate 65536 boards (positions --rank 16) in resumable
#                    chunks, appending to pos65k.txt, until TARGET_BOARDS.
#   Phase 2 (train): seed the restart pool with those boards (--restart-file)
#                    and train ONLY stage 3 (--freeze 3 --restart-stage 3) so
#                    the net learns 65536->131072 from real 65536 positions,
#                    while stages 0-2 (the whole early+mid game) stay byte-frozen.
#                    Promote to champion only if 131072% on a fixed eval subset
#                    improves (tiebreak: longer survival from 65536 boards).
#
# PRODUCTION SAFETY (hard): niced 15 + taskset -c 2,3, RAM-floor watchdog that
# kills the child if free RAM drops below RAM_FLOOR_MB. Never starve the API.
# ============================================================================
set -u

ROOT=/home/ubuntu/2048-endgame
NETS=/home/ubuntu/2048-solver/nets
BIN=$ROOT/target/release/g2048
SEEDNET=$NETS/otd-stage4-bench.bin
CHAMP=$NETS/otd-evolve-champion.bin
WORK=$NETS/otd-evolve-work.bin
POS=$ROOT/pos65k.txt            # the 65536 board library (grows in phase 1)
EVALSET=$ROOT/pos65k-eval.txt   # fixed subset, frozen at the phase-1->2 switch
PARAMS=$ROOT/evolve131-params.env
CSV=$NETS/evolve131-log.csv
DLOG=$NETS/evolve-decisions.log
STATE=$NETS/evolve131-state.env

log(){ echo "$(date -u +%FT%TZ) [131] $*" | tee -a "$DLOG"; }
nboards(){ local n; n=$(grep -c . "$POS" 2>/dev/null); echo "${n:-0}"; }

wait_for_ram(){ local floor=$1; while :; do local a; a=$(free -m|awk '/Mem:/{print $7}'); [ "$a" -ge "$floor" ] && return 0; log "RAM $a<$floor MB - holding"; sleep 30; done; }

# metric on a 65536 board set: prints "p131 avgmoves"
metric(){ taskset -c 2,3 nice -n 15 "$BIN" endgame "$2" --net "$1" 2>/dev/null \
  | awk '/boards/{for(i=1;i<=NF;i++){if($i=="131072")p=$(i+1); if($i=="moves")m=$(i+1)} gsub("%","",p); print p+0, m+0}'; }

score(){ awk -v p="$1" -v m="$2" 'BEGIN{printf "%.1f", p*1000000 + m}'; }

# run a child under the RAM watchdog; returns child's exit (1 if RAM-killed)
guarded(){ local floor=$1; shift; taskset -c 2,3 nice -n 15 "$@" & local tp=$!;
  while kill -0 "$tp" 2>/dev/null; do local a; a=$(free -m|awk '/Mem:/{print $7}');
    if [ "$a" -lt "$floor" ]; then log "RAM $a<$floor during child - killing"; kill "$tp" 2>/dev/null; sleep 2; kill -9 "$tp" 2>/dev/null; wait "$tp" 2>/dev/null; return 1; fi
    sleep 10; done; wait "$tp"; }

# -------------------------------- bootstrap --------------------------------
[ -f "$CHAMP" ] || cp "$SEEDNET" "$CHAMP" || { log "FATAL: no champion / disk"; exit 1; }
[ -f "$CSV" ] || echo "ts,phase,gen,p131,avgmoves,result" > "$CSV"
touch "$POS"
log "=== evolve131 armed (pid $$) boards=$(nboards) ==="

# =============================== main loop ===================================
while :; do
  ENABLED=1; TARGET_BOARDS=150; GEN_COUNT=40; GAMES=250000
  FREEZE=3; RESTART_P=0.9; RESTART_STAGE=3; EVAL_N=60
  GEN_SLEEP=5; RAM_FLOOR_MB=3000
  GEN_ENDGAME_DEPTH=5; GEN_LAYOUTS=five8; GEN_FROM_FILE=$ROOT/pos32k.txt  # phase1: harvest 65536 by playing FROM 32768 seeds with mx + five8 table
  [ -f "$PARAMS" ] && source "$PARAMS"
  [ "$ENABLED" != 1 ] && { log "disabled - idle 60s"; sleep 60; continue; }
  wait_for_ram "$RAM_FLOOR_MB"

  nb=$(nboards)
  # ---------------- Phase 1: grow the 65536 board library ----------------
  if [ "$nb" -lt "$TARGET_BOARDS" ]; then
    tmp=$(mktemp "$ROOT/.pos65k.XXXX")
    if guarded "$RAM_FLOOR_MB" "$BIN" positions "$tmp" "$GEN_COUNT" --net "$CHAMP" --rank 16 \
         --from-file "$GEN_FROM_FILE" \
         --endgame-eval mx --endgame-rank 15 --endgame-depth "$GEN_ENDGAME_DEPTH" \
         --tables "$ROOT/tables" --layouts "$GEN_LAYOUTS" --lookup-rank 15; then
      cat "$tmp" >> "$POS"; sort -u "$POS" -o "$POS"
      log "phase1 grow: boards now $(nboards)/$TARGET_BOARDS"
    else
      log "phase1 chunk aborted (RAM) - cooling 120s"; sleep 120
    fi
    rm -f "$tmp"; sleep "$GEN_SLEEP"; continue
  fi

  # ---- phase 1 -> 2 switch: freeze a fixed eval subset, baseline champion ----
  if [ ! -f "$EVALSET" ]; then
    head -n "$EVAL_N" "$POS" > "$EVALSET"
    read CP131 CPMV < <(metric "$CHAMP" "$EVALSET")
    GEN=0; echo "CP131=$CP131" > "$STATE"; echo "CPMV=$CPMV" >> "$STATE"; echo "GEN=$GEN" >> "$STATE"
    log "phase2 begin: champion baseline 131072 ${CP131}%  avgmoves ${CPMV} on $EVAL_N boards"
  fi
  source "$STATE"; CBEST=$(score "$CP131" "$CPMV")

  # ---------------- Phase 2: train the 65536 -> 131072 conversion ----------
  GEN=$((GEN+1)); SEED=$((2000+GEN))
  log "phase2 gen $GEN: freeze $FREEZE restart $RESTART_P/stage$RESTART_STAGE seed $SEED games $GAMES"
  if ! guarded "$RAM_FLOOR_MB" "$BIN" train "$WORK" "$GAMES" --resume "$CHAMP" --stages 4 \
       --freeze "$FREEZE" --restart "$RESTART_P" --restart-stage "$RESTART_STAGE" \
       --restart-file "$POS" --seed "$SEED" >> "$NETS/evolve131-train.log" 2>&1; then
    log "gen $GEN aborted (RAM) - cooling 120s"; rm -f "$WORK"; sleep 120; continue
  fi
  read W131 WMV < <(metric "$WORK" "$EVALSET"); WSCORE=$(score "$W131" "$WMV")
  log "gen $GEN done: work 131072 ${W131}%  avgmoves ${WMV}  (champ ${CP131}% / ${CPMV})"
  if awk -v w="$WSCORE" -v c="$CBEST" 'BEGIN{exit !(w>c)}'; then
    mv -f "$WORK" "$CHAMP"; CP131=$W131; CPMV=$WMV
    echo "CP131=$CP131" > "$STATE"; echo "CPMV=$CPMV" >> "$STATE"; echo "GEN=$GEN" >> "$STATE"
    log "gen $GEN PROMOTED -> champion 131072 ${CP131}% avgmoves ${CPMV}"
    echo "$(date -u +%FT%TZ),2,$GEN,$W131,$WMV,PROMOTED" >> "$CSV"
  else
    rm -f "$WORK"; echo "GEN=$GEN" >> "$STATE"
    log "gen $GEN kept champion"
    echo "$(date -u +%FT%TZ),2,$GEN,$W131,$WMV,kept" >> "$CSV"
  fi
  sleep "$GEN_SLEEP"
done
