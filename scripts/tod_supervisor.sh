#!/bin/bash
# Keep the prior-information sweep filling whatever GPU capacity is going spare.
#
# This box is shared and the site expects fair usage, so this is deliberately a modest
# consumer, not an opportunistic one. At most ONE worker per card and MAX_WORKERS in
# total, however much capacity appears to be going spare. When a card frees up the sweep
# will spread onto it; when it fills again the extra worker finishes its current cell
# and is not replaced. Nothing is ever killed to make room.
#
# Stacking workers on one card was tried and withdrawn once, on the grounds that it
# doubles this user's share of a card for no wall-clock benefit when the card is
# contended. That reasoning rested on a cell peaking at 3-5 GB of VRAM. MEASURED on
# 2026-11-17, a REDD cell holds 552-702 MiB -- about a tenth of that -- and sits at 99%
# of ONE CPU core, so the bottleneck is the dataloader, not the GPU. Under those numbers
# stacking costs a neighbour very little and should scale nearly linearly, so it is
# being retried WITH MEASUREMENT: compare s/epoch before and after, and drop back to
# MAX_PER_GPU=1 if epoch time degrades. Defaults below stay at the conservative setting;
# the stacked configuration is passed in at launch so this file still documents the
# cautious baseline.
#
# Safety rules, in order of importance:
#   * HEADROOM_MIB is checked against FREE VRAM, and a card must look free on
#     STABLE_POLLS consecutive polls before a worker is added. One poll is not enough:
#     another user's job between allocations reads as free and we would start a cell
#     that then OOMs them (and us). This repo has already lost 942 cells to an OOM storm.
#   * HOME_GPU always keeps one worker, so the sweep makes progress even when every
#     card looks busy.
#   * Workers claim cells with mkdir locks (scripts/sweep_parallel.sh), so adding or
#     losing one can never duplicate or corrupt a cell.
#
# Run:  setsid nohup bash scripts/tod_supervisor.sh > .sweep-tod/supervisor.log 2>&1 &
# Stop: touch .sweep-tod/STOP     (workers finish their current cell, then exit)
set -u

JOBLIST=${JOBLIST:-.sweep-tod/joblist.txt}
LOCKDIR=${LOCKDIR:-.sweep-tod/locks}
LOGDIR=${LOGDIR:-.sweep-tod/logs}
STOPFILE=${STOPFILE:-.sweep-tod/STOP}
PY=${PY:-./.venv/bin/python}
EPOCHS=${EPOCHS:-100}
P_ES=${P_ES:-8}
P_RLR=${P_RLR:-3}

GPUS=${GPUS:-"0 1 2"}
HOME_GPU=${HOME_GPU:-0}
# Demanding far more free VRAM than a cell needs is deliberate: it is the margin that
# keeps us from crowding a neighbour. A REDD cell measures 552-702 MiB (see header), so
# 15000 is ~20x headroom; lower it when stacking, but never below a few GB of slack.
HEADROOM_MIB=${HEADROOM_MIB:-15000}
# Below this, an extra worker on a card is retired (not killed -- just not replaced).
LOW_WATER_MIB=${LOW_WATER_MIB:-6000}
# One per card, hard. See the fair-usage note in the header.
MAX_PER_GPU=${MAX_PER_GPU:-1}
MAX_WORKERS=${MAX_WORKERS:-2}
STABLE_POLLS=${STABLE_POLLS:-3}
POLL=${POLL:-120}

mkdir -p "$LOCKDIR" "$LOGDIR" "$(dirname "$STOPFILE")"
rm -f "$STOPFILE"

declare -A PIDGPU=()      # pid -> gpu
declare -A STABLE=()      # gpu -> consecutive polls seen free
for g in $GPUS; do STABLE[$g]=0; done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1; }

reap() {                   # drop finished workers from the table
  for pid in "${!PIDGPU[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "[$(date +%H:%M:%S)] worker $pid (gpu${PIDGPU[$pid]}) exited"; unset 'PIDGPU[$pid]'; }
  done
}

release_orphan_locks() {
  # A worker that dies mid-cell leaves the lock behind with no owner and no .pt, which
  # would block that cell for the rest of the sweep. Only a lock whose cell has neither
  # a finished .pt nor a live run_one_expe is released.
  local d n c ds app ws m seed da
  for d in "$LOCKDIR"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); c=${n#*__}
    ds=${c%%_*}; c=${c#*_}
    seed=${c##*_}; c=${c%_*}
    app=${c%%_*}; c=${c#*_}
    ws=${c%%_*}; m=${c#*_}
    da=$app; [ "$app" = "WashingMachine" ] && da=WasherDryer
    [ -f "result-redd-tod/${ds}_${app}_10s/$ws/${m}_${seed}.pt" ] && continue
    [ -f "result-redd-tod/${ds}_${da}_10s/$ws/${m}_${seed}.pt" ] && continue
    pgrep -f -- "--appliance $app --window_size $ws --name_model $m --seed $seed" >/dev/null && continue
    rmdir "$d" 2>/dev/null && echo "[$(date +%H:%M:%S)] released orphan lock $n"
  done
}

count_on() {               # workers currently on gpu $1
  local g=$1 n=0
  for pid in "${!PIDGPU[@]}"; do [ "${PIDGPU[$pid]}" = "$g" ] && n=$((n+1)); done
  echo $n
}

spawn() {
  local g=$1
  JOBLIST="$JOBLIST" LOCKDIR="$LOCKDIR" LOGDIR="$LOGDIR" PY="$PY" \
  EPOCHS="$EPOCHS" P_ES="$P_ES" P_RLR="$P_RLR" \
    setsid nohup bash scripts/sweep_parallel.sh worker "$g" >> "$LOGDIR/worker_gpu${g}.out" 2>&1 < /dev/null &
  local pid=$!
  PIDGPU[$pid]=$g
  echo "[$(date +%H:%M:%S)] + worker pid $pid on gpu$g (now $(count_on "$g") there, ${#PIDGPU[@]} total)"
}

remaining() { 
  # cells in the list with no .pt on disk yet
  awk '{print}' "$JOBLIST" | while read -r ds a ws m s root; do
    da=$a; [ "$a" = "WashingMachine" ] && da=WasherDryer
    [ -f "$root/${ds}_${a}_10s/$ws/${m}_${s}.pt" ] && continue
    [ -f "$root/${ds}_${da}_10s/$ws/${m}_${s}.pt" ] && continue
    echo x
  done | wc -l
}

# Adopt workers already running, so a supervisor restart does not spawn duplicates
# alongside them -- which is how a card ends up with two of this user's workers on it.
while read -r pid g; do
  [ -z "${pid:-}" ] && continue
  PIDGPU[$pid]=$g
  echo "adopted existing worker pid $pid on gpu$g"
done < <(ps -eo pid,cmd | awk '$0 ~ /[s]weep_parallel\.sh worker [0-9]+$/ {print $1, $NF}')

echo "=== supervisor started $(date -Is) ==="
echo "joblist $JOBLIST ($(wc -l < "$JOBLIST") cells), home gpu $HOME_GPU,"
echo "add a worker at >=${HEADROOM_MIB}MiB free for $STABLE_POLLS polls, max $MAX_WORKERS workers"

while true; do
  reap
  release_orphan_locks

  if [ -f "$STOPFILE" ]; then
    echo "[$(date +%H:%M:%S)] stop file present; not spawning. ${#PIDGPU[@]} worker(s) still finishing."
    [ ${#PIDGPU[@]} -eq 0 ] && { echo "=== supervisor stopped $(date -Is) ==="; exit 0; }
    sleep "$POLL"; continue
  fi

  left=$(remaining)
  if [ "$left" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] all cells present; waiting for ${#PIDGPU[@]} worker(s) to exit"
    [ ${#PIDGPU[@]} -eq 0 ] && { echo "=== supervisor finished $(date -Is) ==="; exit 0; }
    sleep "$POLL"; continue
  fi

  # The home GPU always keeps one worker, so progress never stalls on a busy box.
  if [ "$(count_on "$HOME_GPU")" -eq 0 ] && [ ${#PIDGPU[@]} -lt "$MAX_WORKERS" ]; then
    spawn "$HOME_GPU"
  fi

  status=""
  for g in $GPUS; do
    f=$(free_mib "$g"); [ -z "$f" ] && continue
    n=$(count_on "$g")
    status="$status gpu$g=${f}MiB/${n}w"

    if [ "$f" -ge "$HEADROOM_MIB" ]; then
      STABLE[$g]=$(( ${STABLE[$g]} + 1 ))
    else
      STABLE[$g]=0
    fi

    if [ "${STABLE[$g]}" -ge "$STABLE_POLLS" ] \
       && [ "$n" -lt "$MAX_PER_GPU" ] \
       && [ ${#PIDGPU[@]} -lt "$MAX_WORKERS" ] \
       && [ "$left" -gt "$(( ${#PIDGPU[@]} ))" ]; then
      echo "[$(date +%H:%M:%S)] gpu$g free (${f}MiB) for ${STABLE[$g]} polls -> adding a worker"
      spawn "$g"
      STABLE[$g]=0        # re-earn the headroom before adding another
    fi
  done

  echo "[$(date +%H:%M:%S)] $left cells left, ${#PIDGPU[@]} worker(s);$status"
  sleep "$POLL"
done
