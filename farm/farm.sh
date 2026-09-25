#!/bin/sh
# Control the training farm from any machine holding ~/.config/g2048/token.
#   farm.sh watch             live overview of every machine (refreshes every 5s)
#   farm.sh status            the same, once
#   farm.sh dell              the command to paste into PowerShell on a Windows PC
#   farm.sh job               current settings
#   farm.sh set k=v [k=v..]   change settings (alpha, restart, secs, send_mb, pause)
#   farm.sh pause | resume    stop / restart training on every machine
#   farm.sh limit NAME N      run machine NAME on N threads (from its next round)
#   farm.sh full NAME         back to every thread on NAME
#   farm.sh kill NAME         close the worker on NAME (it needs a manual start after)
#   farm.sh start | stop      this laptop's own worker
set -e
URL=https://server.jedbillyb.com/g2048
TOKEN_FILE=$HOME/.config/g2048/token
AUTH="Authorization: Bearer $(cat "$TOKEN_FILE")"
DIR=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
CACHE=$HOME/.cache/g2048
api() { curl -sS -H "$AUTH" "$@"; }

set_job() {
    job=$(api "$URL/job")
    for kv in "$@"; do
        k=${kv%%=*}
        job=$(printf '%s\n' "$job" | grep -v "^$k=" || true)
        job=$(printf '%s\n%s' "$job" "$kv")
    done
    printf '%s\n' "$job" | grep . | api --data-binary @- "$URL/job"
}

unset_job() {
    api "$URL/job" | grep -v "^$1=" | api --data-binary @- "$URL/job"
}

worker_pid() { pgrep -f "release/g2048 worker" || true; }

overview() {
    if ! api -f --max-time 10 "$URL/status?color=1" 2>/dev/null; then
        echo "server coordinator not reachable (not started yet, or down)"
    fi
    [ -n "$(worker_pid)" ] || printf '\n\033[33mthis laptop: worker NOT running (start it: farm start)\033[0m\n'
}

case "$1" in
    status) overview ;;
    watch) while :; do out=$(overview 2>&1); clear; printf '%s\n' "$out"; sleep 5; done ;;
    dell)
        echo "Paste into PowerShell on the Dell (keep the window open; closing it stops training):"
        echo
        echo "mkdir -Force \$HOME\\g2048 | Out-Null; cd \$HOME\\g2048; iwr https://server.jedbillyb.com/g2048/files/g2048.exe -OutFile g2048.exe; powercfg /change standby-timeout-ac 0; .\\g2048.exe worker --url $URL --token $(cat "$TOKEN_FILE") --cache cache" ;;
    job) api "$URL/job" ;;
    set) shift; set_job "$@" ;;
    pause) set_job pause=1 ;;
    limit) set_job "threads.$2=$3" >/dev/null; echo "$2 will use $3 threads from its next round (up to 2 min)" ;;
    full) unset_job "threads.$2" >/dev/null; echo "$2 back to full power from its next round (up to 2 min)" ;;
    kill)
        set_job "stop.$2=1" >/dev/null
        echo "stopping $2 at the end of its round (up to 3 min)..."
        i=0
        until api "$URL/status" | grep -q "^$2 .*\(OFFLINE\|stopped\)" || [ $i -ge 36 ]; do sleep 5; i=$((i+1)); done
        unset_job "stop.$2" >/dev/null
        echo "done; start it again on the machine itself" ;;
    resume) set_job pause=0 ;;
    start)
        [ -n "$(worker_pid)" ] && { echo "already running"; exit 0; }
        mkdir -p "$CACHE"
        nohup "$DIR/target/release/g2048" worker --url "$URL" --token-file "$TOKEN_FILE" --cache "$CACHE" >> "$CACHE/worker.log" 2>&1 &
        echo "laptop worker started, log: $CACHE/worker.log" ;;
    stop)
        pid=$(worker_pid)
        [ -z "$pid" ] && { echo "not running"; exit 0; }
        kill $pid && echo "laptop worker stopped" ;;
    *) sed -n '2,14p' "$0" | sed 's/^# //' ;;
esac
