#!/bin/sh
# Shows every running g2048 job with progress and time left.
cd "$(dirname "$0")"
found=0
for pid in $(pgrep -x g2048); do
  found=1
  args=$(tr '\0' ' ' < /proc/$pid/cmdline | sed 's|^\./target/release/||')
  state=$(awk '{print $3}' /proc/$pid/stat)
  secs=$(ps -o etimes= -p $pid | tr -d ' ')
  echo "● $args"
  [ "$state" = "T" ] && echo "  PAUSED (resume: kill -CONT $pid)"
  case "$args" in
    *" train "*)
      out=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="train") print $(i+1)}')
      total=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="train") print $(i+2)}')
      log="${out%.bin}.log"
      last=$(grep " games " "$log" | tail -1)
      done_g=$(echo "$last" | awk '{print $1}'); t=$(echo "$last" | awk '{print $3}' | tr -d s)
      if [ -n "$done_g" ] && [ "$t" -gt 0 ]; then
        left=$(( (total - done_g) * t / done_g ))
        echo "  $done_g / $total games ($((100 * done_g / total))%), about $((left / 60)) min left"
        echo "  latest: $(echo "$last" | cut -c30-)"
      fi ;;
    *" endgame "*|*" bench "*|*" positions "*)
      prog=$(cat /tmp/g2048-progress-$pid 2>/dev/null)
      echo "  running $((secs / 60))m$((secs % 60))s ${prog:+ - $prog}" ;;
  esac
done
[ $found = 0 ] && echo "nothing running"; exit 0
