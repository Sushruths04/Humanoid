#!/bin/bash
# monitor.sh — append hourly summary to stage1_monitor.log
LOG=~/_setup_logs/stage1.log
OUT=~/_setup_logs/stage1_monitor.log

if [ ! -f "$LOG" ]; then echo "No log found"; exit 1; fi

CLEAN=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG")
LAST_ITER=$(echo "$CLEAN" | grep 'Learning iteration' | tail -1 | grep -oP '\d+/\d+')
LAST_REWARD=$(echo "$CLEAN" | grep 'Mean reward' | tail -1 | awk '{print $NF}')
LAST_FELL=$(echo "$CLEAN" | grep 'Episode_Termination/fell' | tail -1 | awk '{print $NF}')
LAST_BR=$(echo "$CLEAN" | grep 'Episode_Termination/board_range' | tail -1 | awk '{print $NF}')
LAST_TIMEOUT=$(echo "$CLEAN" | grep 'Episode_Termination/timeout' | tail -1 | awk '{print $NF}')
LAST_LEN=$(echo "$CLEAN" | grep 'Mean episode length' | tail -1 | awk '{print $NF}')
CHUNKS_DONE=$(echo "$CLEAN" | grep 'saved model' | wc -l)
DOCKER_UP=$(docker ps --format '{{.Status}}' 2>/dev/null | head -1)
ELAPSED=$(ps -o etimes= -p $(docker inspect --format '{{.State.Pid}}' $(docker ps -q) 2>/dev/null | head -1) 2>/dev/null | awk '{h=int($1/3600);m=int(($1%3600)/60);printf "%dh%02dm",h,m}')
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

printf "| %s | %s | %s | rew=%s fell=%s br=%s to=%s len=%s | %s |\n" \
  "$TIMESTAMP" "$ELAPSED" "$LAST_ITER" "$LAST_REWARD" "$LAST_FELL" "$LAST_BR" "$LAST_TIMEOUT" "$LAST_LEN" "$DOCKER_UP" >> "$OUT"

echo "Snapshot saved: iter=$LAST_ITER reward=$LAST_REWARD fell=$LAST_FELL board_range=$LAST_BR elapsed=$ELAPSED"
