#!/usr/bin bash

_STOP=0
_CHILD_PID=""
trap '_STOP=1; [ -n "$_CHILD_PID" ] && kill "$_CHILD_PID" 2>/dev/null; echo "Interrupted — stopping experiments."' INT TERM

# Global parameters
declare -a SEEDS=(0 1 2)

declare -a DATASETS_1=("REFIT")
declare -a APPLIANCES_1=("WashingMachine" "Dishwasher" "Kettle" "Microwave")

declare -a DATASETS_2=("UKDALE")
declare -a APPLIANCES_2=("WashingMachine" "Dishwasher" "Kettle" "Microwave" "Fridge")

declare -a MODELS_1=("BiLSTM" "FCN" "CNN1D" "UNetNILM" "DAResNet" "BERT4NILM" "DiffNILM" \
                     "TSILNet" "Energformer" "BiGRU" "STNILM" "NILMFormer")
declare -a WINDOW_SIZES_1=("128" "256" "512" "360" "720")

declare -a MODELS_2=("ConvNet" "ResNet" "Inception")
declare -a WINDOW_SIZES_2=("day" "week" "month")

# Run experiments
# Data is processed once per (dataset, appliance, window, seed); models loop inside Python.
run_batch() {
  local -n arr_datasets=$1
  local -n arr_appliances=$2
  local -n arr_windows=$3
  local model_group=$4

  for dataset in "${arr_datasets[@]}"; do
    for appliance in "${arr_appliances[@]}"; do
      for win in "${arr_windows[@]}"; do
        for seed in "${SEEDS[@]}"; do
          [ "$_STOP" -eq 1 ] && return 1
          echo "Running group: dataset=$dataset appliance=$appliance win=$win seed=$seed group=$model_group"
          uv run -m scripts.run_group_expe \
            --dataset "$dataset" \
            --sampling_rate "1min" \
            --appliance "$appliance" \
            --window_size "$win" \
            --seed "$seed" \
            --model_group "$model_group" &
          _CHILD_PID=$!
          wait "$_CHILD_PID"
          _CHILD_PID=""
          [ "$_STOP" -eq 1 ] && return 1
        done
      done
    done
  done
}

#####################################
# Run all possible experiments
#####################################
run_batch DATASETS_1 APPLIANCES_1 WINDOW_SIZES_1 nilm || return 1 2>/dev/null || exit 1
run_batch DATASETS_1 APPLIANCES_1 WINDOW_SIZES_2 tser || return 1 2>/dev/null || exit 1
run_batch DATASETS_2 APPLIANCES_2 WINDOW_SIZES_1 nilm || return 1 2>/dev/null || exit 1
run_batch DATASETS_2 APPLIANCES_2 WINDOW_SIZES_2 tser || return 1 2>/dev/null || exit 1