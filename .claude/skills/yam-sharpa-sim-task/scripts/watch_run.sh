#!/bin/bash
# Launch a simulation in the background with a log, then print its progress until it ends.
#
#   .claude/skills/yam-sharpa-sim-task/scripts/watch_run.sh app/config/<name>.json [--frames N] [--every 30]
#
# The log goes to output/logs/<name>.log. Progress is the solver's last frame line (frame, outer and Newton
# iterations, PCG average, contact pairs, alpha). The watcher reports an abort (consecutive steps at the
# outer-iteration cap) and the elapsed time at the end. To stop a run early, kill the process by its pid
# (printed below), never with a pattern that could match this shell.
set -u
cfg=${1:?config path}; shift
every=30; frames=""
while [ $# -gt 0 ]; do case "$1" in --frames) frames="--frames $2"; shift 2;; --every) every=$2; shift 2;; *) shift;; esac; done
name=$(basename "$cfg" .json); mkdir -p output/logs; log=output/logs/$name.log; rm -f "$log"
setsid nohup build/release/cs_run "$cfg" $frames > "$log" 2>&1 < /dev/null &
sleep 2
pid=$(ps -eo pid,args | awk -v c="$cfg" '$2 ~ /\/cs_run$/ && index($0, c) {print $1; exit}')
echo "started $name (pid ${pid:-?}), log $log"; t0=$(date +%s)
while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do
  sleep "$every"
  line=$(tr '\r' '\n' < "$log" | grep -E "frame +[0-9]+ " | tail -1 | cut -c26-110)
  echo "$(( ($(date +%s) - t0) / 60 )) min: $line"
done
echo "finished after $(( ($(date +%s) - t0) / 60 )) min: $(tr '\r' '\n' < "$log" | grep -E "frame +[0-9]+ " | tail -1 | cut -c26-110)"
tr '\r' '\n' < "$log" | grep -iE "capped|abort|fatal|error" | tail -3
