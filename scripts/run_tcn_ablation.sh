#!/bin/bash
#
# TCN ablation: isolate what the pretrained TCN_KL gains from curation, augmentation
# and the Karhunen-Loeve front end.
#
#   TCN             no KL, no augmentation, no curation
#   TCN_KL_scratch  + KL features
#   TCN_KL_aug      + synthetic training data (replaces the real windows, as TCN_KL did)
#   TCN_KL          + curation, external recipe  (already in 2207-results/)
#
# Same grid, sampling rate and recipe as the baselines in result-synagg/, so the rows are
# directly comparable. Resumable: completed runs are skipped, partial ones redone.
#
# Usage:
#   bash scripts/run_tcn_ablation.sh          # full grid, 5 appliances x 3 windows x 3 seeds x 3 arms
#   bash scripts/run_tcn_ablation.sh smoke    # one cell (Kettle / ws=256 / seed 0)
#   PY=env/bin/python bash scripts/run_tcn_ablation.sh   # force an interpreter
#   FORCE=1 bash scripts/run_tcn_ablation.sh             # redo runs that already completed
#   RESULT_PATH=result-curated-scaling/ DATASET=... bash scripts/run_tcn_ablation.sh
#   APPLIANCES="Fridge Microwave" MODELS="TCN" bash scripts/run_tcn_ablation.sh   # scope the grid

set -u

# shellcheck source=scripts/tcn_ablation_lib.sh
. "$(dirname "$0")/tcn_ablation_lib.sh"

RESULT_PATH="${RESULT_PATH:-result-tcn-ablation/}"
DATASET="${DATASET:-UKDALE}"
SAMPLING_RATE="10s"

# REDD has no kettle, and its washer/dryer is one combined meter (WasherDryer).
if [ "$DATASET" = "REDD" ]; then
    _default_apps="WasherDryer Dishwasher Microwave Fridge"
    _smoke_app="Fridge"      # the only REDD appliance with ample curated activations
else
    _default_apps="WashingMachine Dishwasher Kettle Microwave Fridge"
    _smoke_app="Kettle"
fi

if [ "${1:-}" = "smoke" ]; then
    APPLIANCES=("$_smoke_app")
    WINDOW_SIZES=(256)
    SEEDS=(0)
else
    read -r -a APPLIANCES  <<< "${APPLIANCES:-$_default_apps}"
    read -r -a WINDOW_SIZES <<< "${WINDOW_SIZES:-128 256 512}"
    read -r -a SEEDS        <<< "${SEEDS:-0 1 2}"
fi
read -r -a MODELS <<< "${MODELS:-TCN TCN_KL_scratch TCN_KL_aug}"

n_done=0
n_skip=0
n_fail=0

tcn_preflight "$0" || exit 1

_STOP=0
trap '_STOP=1; echo "Interrupted - finishing current run then stopping."' INT TERM

echo "Interpreter: ${PY}  ($("$PY" -c 'import torch; print("torch", torch.__version__)' 2>/dev/null))"
[ -n "${FORCE:-}" ] && echo "FORCE set: existing results will be overwritten."

for appliance in "${APPLIANCES[@]}"; do
    for window_size in "${WINDOW_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for model in "${MODELS[@]}"; do
                [ "$_STOP" -eq 1 ] && exit 1

                out="${RESULT_PATH}${DATASET}_${appliance}_${SAMPLING_RATE}/${window_size}/${model}_${seed}.pt"
                if [ -f "$out" ] && [ -z "${FORCE:-}" ]; then
                    if tcn_is_complete "$out"; then
                        echo "Skipping (complete): $out"
                        n_skip=$((n_skip + 1))
                        continue
                    fi
                    echo "Redoing (partial .pt, no test metrics): $out"
                fi

                echo "Running $model / $appliance / ws=$window_size / seed=$seed ..."
                if PYTHONPATH=. "$PY" -m scripts.run_one_expe \
                        --dataset "$DATASET" \
                        --sampling_rate "$SAMPLING_RATE" \
                        --appliance "$appliance" \
                        --window_size "$window_size" \
                        --name_model "$model" \
                        --seed "$seed" \
                        --result_path "$RESULT_PATH"; then
                    n_done=$((n_done + 1))
                    echo "Done: $model / $appliance / ws=$window_size / seed=$seed"
                else
                    # Capture before anything else: n_fail=... would reset $?.
                    rc=$?
                    n_fail=$((n_fail + 1))
                    echo "FAILED (exit ${rc}): $model / $appliance / ws=$window_size / seed=$seed"
                fi
            done
        done
    done
done

echo
echo "TCN ablation finished: ${n_done} run, ${n_skip} skipped, ${n_fail} failed."
[ "$n_fail" -gt 0 ] && exit 1
exit 0
