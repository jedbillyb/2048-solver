#!/bin/sh
# Control the training farm from any machine holding ~/.config/g2048/token.
#   farm.sh status            every worker's speed and results
#   farm.sh job               current settings
#   farm.sh set k=v [k=v..]   change settings (alpha, restart, secs, send_mb, pause)
#   farm.sh pause | resume    stop / restart training on every machine
#   farm.sh start | stop      this laptop's own worker
set -e
URL=https://server.jedbillyb.com/g2048
TOKEN_FILE=$HOME/.config/g2048/token
AUTH="Authorization: Bearer $(cat "$TOKEN_FILE")"
DIR=$(cd "$(dirname "$0")/.." && pwd)
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

worker_pid() { pgrep -f "release/g2048 worker" || true; }

case "$1" in
    status) api "$URL/status" ;;
    job) api "$URL/job" ;;
    set) shift; set_job "$@" ;;
    pause) set_job pause=1 ;;
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
    *) sed -n '2,8p' "$0" | sed 's/^# //' ;;
esac
