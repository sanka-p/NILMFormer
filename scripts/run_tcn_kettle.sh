#!/bin/bash
#
# TCN ablation arms 1-3 on UK-DALE / Kettle.
#
#   TCN             no curation, no augmentation, no KL features
#   TCN_KL_scratch  + Karhunen-Loeve input features
#   TCN_KL_aug      + synthetic training data (replaces the real windows)
#
# The fourth rung, the pretrained TCN_KL, already exists in
# 2207-results/result-synagg-overfit/ and is not retrained here.
#
# Same dataset variant, sampling rate and recipe as the baselines in result-synagg/, so
# the rows drop straight into scripts/make_table.py.
#
# Usage:
#   bash scripts/run_tcn_kettle.sh                     # 3 windows x 3 seeds x 3 arms = 27 runs
#   WINDOW_SIZES="256" SEEDS="0" bash scripts/run_tcn_kettle.sh   # single cell
#   PY=env/bin/python bash scripts/run_tcn_kettle.sh   # force a specific interpreter
#   FORCE=1 bash scripts/run_tcn_kettle.sh             # redo runs that already completed
#
# Resumable: a COMPLETED run is skipped and a partial one redone (the trainer rewrites the
# same .pt on every best-loss epoch, so mere existence proves nothing). Safe to re-run
# after an interruption on this shared box.

set -u

# shellcheck source=scripts/tcn_ablation_lib.sh
. "$(dirname "$0")/tcn_ablation_lib.sh"

DATASET="UKDALE"
APPLIANCE="Kettle"
SAMPLING_RATE="10s"
RESULT_PATH="result-tcn-ablation/"

read -r -a WINDOW_SIZES <<< "${WINDOW_SIZES:-128 256 512}"
read -r -a SEEDS        <<< "${SEEDS:-0 1 2}"
read -r -a MODELS       <<< "${MODELS:-TCN TCN_KL_scratch TCN_KL_aug}"

tcn_preflight "$0" || exit 1

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/tcn_kettle_$(date +%Y%m%d_%H%M%S).log"

_STOP=0
trap '_STOP=1; echo "Interrupted - stopping after the current run."' INT TERM

n_done=0
n_skip=0
n_fail=0

echo "Logging to ${RUN_LOG}"
echo "Interpreter: ${PY}  ($("$PY" -c 'import torch; print("torch", torch.__version__)' 2>/dev/null))"
echo "Grid: windows=[${WINDOW_SIZES[*]}] seeds=[${SEEDS[*]}] models=[${MODELS[*]}]"
[ -n "${FORCE:-}" ] && echo "FORCE set: existing results will be overwritten."

# Window and seed outermost so the three arms of a cell run back to back: they share the
# cached activation segments (results/tcn_aug_cache) and a warm page cache for UK-DALE.
for window_size in "${WINDOW_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for model in "${MODELS[@]}"; do
            [ "$_STOP" -eq 1 ] && break 3

            out="${RESULT_PATH}${DATASET}_${APPLIANCE}_${SAMPLING_RATE}/${window_size}/${model}_${seed}.pt"
            if [ -f "$out" ] && [ -z "${FORCE:-}" ]; then
                if tcn_is_complete "$out"; then
                    echo "skip  ${model} ws=${window_size} seed=${seed}  (complete)"
                    n_skip=$((n_skip + 1))
                    continue
                fi
                echo "redo  ${model} ws=${window_size} seed=${seed}  (partial .pt, no test metrics)"
            fi

            echo "run   ${model} ws=${window_size} seed=${seed}  ($(date +%H:%M:%S))"
            if PYTHONPATH=. "$PY" -m scripts.run_one_expe \
                    --dataset "$DATASET" \
                    --sampling_rate "$SAMPLING_RATE" \
                    --appliance "$APPLIANCE" \
                    --window_size "$window_size" \
                    --name_model "$model" \
                    --seed "$seed" \
                    --result_path "$RESULT_PATH" >> "$RUN_LOG" 2>&1; then
                n_done=$((n_done + 1))
                echo "  ok  -> ${out}"
            else
                # Capture before anything else: n_fail=... would reset $?.
                rc=$?
                n_fail=$((n_fail + 1))
                echo "  FAILED (exit ${rc}) - see ${RUN_LOG}"
            fi
        done
    done
done

echo
echo "Kettle ablation finished: ${n_done} run, ${n_skip} skipped, ${n_fail} failed."
echo "Log: ${RUN_LOG}"
if [ "$n_fail" -gt 0 ]; then
    exit 1
fi
echo
echo "Next:"
echo "  PYTHONPATH=. ${PY} -m scripts.make_table --refresh"
