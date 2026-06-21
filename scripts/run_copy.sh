#!/bin/bash

for appliance in "WashingMachine" "Dishwasher" "Kettle" "Microwave" "Fridge"; do
    echo "Running experiment for $appliance..."
    uv run -m scripts.run_one_expe \
        --dataset "UKDALE" \
        --sampling_rate "10s" \
        --appliance "$appliance" \
        --window_size 256 \
        --name_model NILMFormer \
        --seed 1
    echo "Done: $appliance"
done
