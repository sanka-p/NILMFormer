#!/bin/bash

for appliance in "WashingMachine" "Dishwasher" "Kettle" "Microwave"; do
    echo "Running experiment for $appliance..."
    uv run -m scripts.run_one_expe \
        --dataset "REFIT" \
        --sampling_rate "10s" \
        --appliance "$appliance" \
        --window_size 128 \
        --name_model BERT4NILM \
        --seed 0
    echo "Done: $appliance"
done
