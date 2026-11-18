#!/bin/bash
# Multi-GPU sweep runner.
#
# Replaces the serial chain. Work is a flat job list; each worker walks it and claims a
# cell by atomically creating a lock directory (mkdir is atomic on POSIX), so workers
# never collide and the split self-balances -- a GPU that draws cheap cells simply takes
# more of them. Any worker count works, and a killed run is resumed by relaunching:
# completion is judged by a .pt.done marker on disk, not by the lock.
#
# Run under tmux (see `launch` below) so it survives the terminal going away. The
# previous serial chain was started with setsid+nohup and still died with its session;
# tmux is the thing that actually detaches here.
#
# Usage:
#   bash scripts/sweep_parallel.sh launch          # build job list, start one worker per GPU
#   bash scripts/sweep_parallel.sh status          # progress
#   bash scripts/sweep_parallel.sh stop            # stop all workers
#   bash scripts/sweep_parallel.sh worker <gpu>    # (internal)
set -u

JOBLIST=${JOBLIST:-.sweep/joblist.txt}
LOCKDIR=${LOCKDIR:-.sweep/locks}
LOGDIR=${LOGDIR:-.sweep/logs}
PY=${PY:-./.venv/bin/python}
GPUS=${GPUS:-"0 1 2"}      # gpu id per worker; repeat an id to share a card
EPOCHS=${EPOCHS:-100}
P_ES=${P_ES:-8}
P_RLR=${P_RLR:-3}
SESSION=${SESSION:-nilmsweep}
# Free-RAM floor before a worker starts a cell. UK-DALE needs a bigger reserve because
# its window tensors are roughly an order of magnitude larger than REDD's.
MIN_FREE_GB=${MIN_FREE_GB:-5}
UKDALE_MIN_FREE_GB=${UKDALE_MIN_FREE_GB:-10}
# Free-VRAM floor, in MiB. A UK-DALE cell peaks around 5.7 GB of VRAM against REDD's
# ~2 GB, and this card has 23.7 GB, so four UK-DALE cells at once do not fit. Workers
# therefore wait for VRAM as well as RAM, which serialises the big cells automatically
# instead of letting them all OOM.
MIN_FREE_VRAM=${MIN_FREE_VRAM:-3000}
UKDALE_MIN_FREE_VRAM=${UKDALE_MIN_FREE_VRAM:-7000}

REDD_MODELS="TCN_KL_aug NILMFormer BERT4NILM BiGRU BiLSTM TCN TCN_KL_scratch TCN_aug_curated TCN_KL_aug_curated"
RF_MODELS="TCN_rf TCN_KL_scratch_rf TCN_aug_curated_rf TCN_KL_aug_rf TCN_KL_aug_curated_rf"
REDD_APPS="Fridge Microwave Dishwasher WashingMachine"
UKDALE_APPS="Fridge Kettle Microwave Dishwasher WashingMachine"
UKDALE_WINDOWS=${UKDALE_WINDOWS:-"512"}

# Directory name for the skip check: run_one_expe maps the WashingMachine config onto
# the WasherDryer channel on REDD, so that is the directory it writes.
dir_app() { [ "$1" = "WashingMachine" ] && echo WasherDryer || echo "$1"; }

build_joblist() {
  mkdir -p "$(dirname "$JOBLIST")"
  : > "$JOBLIST"
  # REDD first (cheap cells, and the tables depend on them), UK-DALE last.
  for ws in 512 256 128; do
    for m in $REDD_MODELS; do
      for a in $REDD_APPS; do
        for s in 0 1 2; do
          echo "REDD $a $ws $m $s result-redd-fix" >> "$JOBLIST"
        done
      done
    done
  done
  for ws in 512 256 128; do
    for m in $RF_MODELS; do
      for a in $REDD_APPS; do
        for s in 0 1 2; do
          echo "REDD $a $ws $m $s result-redd-archfix" >> "$JOBLIST"
        done
      done
    done
  done
  for ws in $UKDALE_WINDOWS; do
    for m in $REDD_MODELS; do
      for a in $UKDALE_APPS; do
        for s in 0 1 2; do
          echo "UKDALE $a $ws $m $s result-ukdale-fix" >> "$JOBLIST"
        done
      done
    done
  done
  echo "job list: $(wc -l < "$JOBLIST") cells -> $JOBLIST"
}

# Echo the .pt path for a cell, preferring one that already exists on disk.
cell_pt() {     # dataset app ws model seed root
  local ds=$1 a=$2 ws=$3 m=$4 s=$5 root=$6
  local p="$root/${ds}_${a}_10s/$ws/${m}_${s}.pt"
  [ -f "$p" ] && { echo "$p"; return 0; }
  if [ "$ds" = "REDD" ]; then
    local q="$root/${ds}_$(dir_app "$a")_10s/$ws/${m}_${s}.pt"
    [ -f "$q" ] && { echo "$q"; return 0; }
  fi
  echo "$p"; return 1
}

# A cell counts as finished only when its .pt is accompanied by a .pt.done marker the
# worker writes after run_one_expe exits 0. The .pt alone is NOT enough: the trainer
# checkpoints weights mid-run, so a killed cell leaves a ~400 KB .pt with no
# test_metrics_timestamp in it. Judging on existence alone made two reference cells
# (TCN_KL_scratch seed 2, Dishwasher and Microwave) permanently skipped -- they were
# silently absent from every table until a scan for partial checkpoints found them.
cell_done() {   # dataset app ws model seed root
  local pt
  pt=$(cell_pt "$@") || return 1
  [ -f "$pt.done" ] && return 0
  return 1
}

worker() {
  local gpu=$1
  mkdir -p "$LOCKDIR" "$LOGDIR"
  local log="$LOGDIR/gpu${gpu}_$$.log"
  echo "=== worker gpu$gpu started $(date -Is) ===" >> "$log"
  local n=0
  local pass=0
  # Several passes over the list, because a cell that fails releases its lock but the
  # worker has already moved past that line. Re-walking picks up anything that failed
  # transiently (a stale cache entry, a contended GPU) without a manual relaunch. A pass
  # that claims nothing means the work is genuinely done, and MAXPASS stops a cell that
  # fails every time from looping forever.
  local MAXPASS=${MAXPASS:-50}
  while [ $pass -lt $MAXPASS ]; do
   pass=$((pass+1))
   local claimed=0
   echo "=== worker gpu$gpu pass $pass $(date -Is) ===" >> "$log"
   while read -r ds a ws m s root <&9; do
    [ -z "${ds:-}" ] && continue
    cell_done "$ds" "$a" "$ws" "$m" "$s" "$root" && continue
    local lock="$LOCKDIR/${root//\//_}__${ds}_${a}_${ws}_${m}_${s}"
    mkdir "$lock" 2>/dev/null || continue       # another worker owns it
    claimed=$((claimed+1))
    # Memory gate. This box has 31 GB and no usable swap, and a UK-DALE cell holds far
    # more in RAM than a REDD one (32k-132k windows against 1.5k-10k), so several large
    # cells starting together can exhaust the machine -- which has already cost one
    # reaped process. Wait for room instead of racing into the OOM killer.
    need=$MIN_FREE_GB
    [ "$ds" = "UKDALE" ] && need=$UKDALE_MIN_FREE_GB
    waited=0
    while [ "$(awk '"'"'/MemAvailable/ {print int($2/1048576)}'"'"' /proc/meminfo)" -lt "$need" ]; do
      if [ $waited -ge 1800 ]; then
        echo "[$(date +%H:%M:%S)] gpu$gpu proceeding after ${waited}s wait (low mem)" >> "$log"
        break
      fi
      sleep 30; waited=$((waited+30))
    done
    vneed=$MIN_FREE_VRAM
    [ "$ds" = "UKDALE" ] && vneed=$UKDALE_MIN_FREE_VRAM
    vwaited=0
    while true; do
      vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1)
      [ -z "$vfree" ] && break                 # no nvidia-smi: do not block the sweep
      [ "$vfree" -ge "$vneed" ] && break
      if [ $vwaited -ge 1800 ]; then
        echo "[$(date +%H:%M:%S)] gpu$gpu proceeding after ${vwaited}s wait (low VRAM ${vfree}MiB)" >> "$log"
        break
      fi
      sleep 30; vwaited=$((vwaited+30))
    done
    echo "[$(date +%F' '%H:%M:%S)] gpu$gpu START $root $ds $a ws=$ws $m seed=$s" >> "$log"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu $PY -m scripts.run_one_expe \
      --dataset "$ds" --sampling_rate 10s --appliance "$a" \
      --window_size "$ws" --name_model "$m" --seed "$s" \
      --result_path "$root" \
      --epochs "$EPOCHS" --p_es "$P_ES" --p_rlr "$P_RLR" \
      --shuffle_train --single_gpu --cache_data \
      >> "$log" 2>&1 < /dev/null
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] gpu$gpu FAILED rc=$rc $root $ds $a ws=$ws $m seed=$s" >> "$log"
      rmdir "$lock" 2>/dev/null                 # let a later pass retry it
    else
      n=$((n+1))
      # Mark the cell finished only now, after a clean exit. See cell_done().
      pt=$(cell_pt "$ds" "$a" "$ws" "$m" "$s" "$root") && : > "$pt.done"
      echo "[$(date +%H:%M:%S)] gpu$gpu DONE  $root $ds $a ws=$ws $m seed=$s" >> "$log"
    fi
   done 9< "$JOBLIST"
   [ $claimed -eq 0 ] && break
  done
  echo "=== worker gpu$gpu finished $(date -Is), trained $n ===" >> "$log"
}

case "${1:-}" in
  launch)
    build_joblist
    # Locks are advisory only; the .pt.done marker is the real completion marker, so
    # clearing them on launch just lets an interrupted cell be retried.
    rm -rf "$LOCKDIR"; mkdir -p "$LOCKDIR" "$LOGDIR"
    # Stop cleanly first. Killing the tmux session has been observed to leave worker
    # shells and their python children alive; they then race the new workers for the
    # same cells, and the lock directory is cleared below, which removes the only thing
    # keeping the two sets apart. Kill by pattern and verify before starting.
    tmux kill-session -t "$SESSION" 2>/dev/null
    sleep 1
    pkill -f "sweep_parallel.sh worker" 2>/dev/null
    sleep 1
    pkill -f "scripts.run_one_expe" 2>/dev/null
    sleep 3
    left=$(ps -eo cmd | grep -c "[s]cripts.run_one_expe")
    if [ "$left" -gt 0 ]; then
      echo "WARNING: $left run_one_expe processes survived the stop; not launching." >&2
      echo "Investigate with: ps -eo pid,etime,cmd | grep run_one_expe" >&2
      exit 1
    fi
    # Window names are per WORKER, not per GPU, so the same GPU id may appear more than
    # once in GPUS without the window names colliding.
    i=0
    for g in $GPUS; do
      if [ "$i" = "0" ]; then
        tmux new-session -d -s "$SESSION" -n "w${i}_gpu${g}" \
          "PY='$PY' bash scripts/sweep_parallel.sh worker $g"
      else
        tmux new-window -t "$SESSION" -n "w${i}_gpu${g}" \
          "PY='$PY' bash scripts/sweep_parallel.sh worker $g"
      fi
      i=$((i+1))
    done
    sleep 2
    tmux list-windows -t "$SESSION"
    echo "launched. follow with: tmux attach -t $SESSION   (detach: ctrl-b d)"
    ;;
  worker)  worker "${2:?gpu id}" ;;
  status)
    tot=$(wc -l < "$JOBLIST" 2>/dev/null || echo 0)
    for root in result-redd-fix result-redd-archfix result-ukdale-fix; do
      printf '%-22s %4d cells\n' "$root" "$(find $root -name '*.pt' 2>/dev/null | wc -l)"
    done
    echo "job list total: $tot"
    echo "running:        $(ls -d $LOCKDIR/*/ 2>/dev/null | wc -l) claimed"
    echo "failures:       $(grep -h FAILED $LOGDIR/*.log 2>/dev/null | wc -l)"
    tmux list-windows -t "$SESSION" 2>/dev/null || echo "no tmux session"
    ;;
  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "not running"
    pkill -f "scripts.run_one_expe" 2>/dev/null
    ;;
  *) echo "usage: $0 {launch|status|stop|worker <gpu>}"; exit 2 ;;
esac
