#!/usr/bin/env bash
# Launch the BEHAVIOR env-server pool for integrated (online) RL, and emit the
# YAML block the trainer config needs.
#
# One server = one process = one booted activity (OmniGibson locks HEADLESS at the
# first boot and update_task() cannot switch activities), serving one session at a
# time. Rollout concurrency is therefore exactly the number of servers, so give
# each activity as many replicas as the GRPO group_size you want to run in
# parallel — otherwise the group serializes.
#
# Boot is ~5 min per server (shader compile) and each holds a full IsaacSim, so
# start with a couple of activities and grow once the loop is verified.
#
# Usage (inside the unified container):
#   scripts/launch_behavior_env_servers.sh \
#       --activities picking_up_trash,bringing_in_wood \
#       --replicas 4 --base-port 18800 --obs-mode full
#
# Then paste the printed `tools.behavior.servers:` block into the trainer config,
# or point the config at the generated servers.yaml.
set -euo pipefail

ACTIVITIES=""
REPLICAS=2
BASE_PORT=18800
OBS_MODE=full
LOG_DIR=${LOG_DIR:-/data/behavior-data/env_servers}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --activities) ACTIVITIES="$2"; shift 2 ;;
    --replicas)   REPLICAS="$2";   shift 2 ;;
    --base-port)  BASE_PORT="$2";  shift 2 ;;
    --obs-mode)   OBS_MODE="$2";   shift 2 ;;
    --log-dir)    LOG_DIR="$2";    shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ACTIVITIES" ]] || { echo "--activities is required" >&2; exit 2; }

mkdir -p "$LOG_DIR"
OUT_YAML="$LOG_DIR/servers.yaml"
: > "$OUT_YAML"
echo "tools:" >> "$OUT_YAML"
echo "  behavior:" >> "$OUT_YAML"
echo "    servers:" >> "$OUT_YAML"

port=$BASE_PORT
pids=()
declare -A urls_by_activity

IFS=',' read -ra ACTS <<< "$ACTIVITIES"
for act in "${ACTS[@]}"; do
  for ((r = 0; r < REPLICAS; r++)); do
    log="$LOG_DIR/${act}_${port}.log"
    echo "starting $act on :$port  (log: $log)"
    with-env python -m rlinf.envs.behavior.env_server \
      --activity "$act" --port "$port" --obs-mode "$OBS_MODE" \
      > "$log" 2>&1 &
    pids+=($!)
    urls_by_activity[$act]+="http://127.0.0.1:${port} "
    port=$((port + 1))
  done
done

# Wait for the readiness line rather than sleeping a guessed interval — boot time
# varies with shader-cache state from ~40 s (warm) to ~5 min (cold).
echo "waiting for ${#pids[@]} servers to boot (up to 15 min)..."
deadline=$((SECONDS + 900))
for act in "${ACTS[@]}"; do
  for url in ${urls_by_activity[$act]}; do
    p=${url##*:}
    log="$LOG_DIR/${act}_${p}.log"
    until grep -q BEHAVIOR_ENV_SERVER_READY "$log" 2>/dev/null; do
      if (( SECONDS > deadline )); then
        echo "TIMEOUT waiting for $act on :$p — see $log" >&2
        exit 1
      fi
      # A server that died must not be waited on for the full 15 min.
      if ! curl -sf "http://127.0.0.1:${p}/health" >/dev/null 2>&1 \
         && ! kill -0 "$(pgrep -f "env_server.*--port $p" | head -1)" 2>/dev/null; then
        echo "server for $act on :$p exited during boot — see $log" >&2
        tail -20 "$log" >&2
        exit 1
      fi
      sleep 5
    done
    echo "  ready: $act :$p"
  done
done

for act in "${ACTS[@]}"; do
  echo "      ${act}:" >> "$OUT_YAML"
  for url in ${urls_by_activity[$act]}; do
    echo "        - ${url}" >> "$OUT_YAML"
  done
done

echo
echo "all ${#pids[@]} servers ready. Config block written to $OUT_YAML:"
echo
cat "$OUT_YAML"
echo
echo "PIDs: ${pids[*]}"
echo "Stop them with: pkill -f rlinf.envs.behavior.env_server"
wait
