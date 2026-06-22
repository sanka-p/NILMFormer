#!/bin/bash

for appliance in "WasherDryer"; do
    for window_size in 128 256 512; do
        for seed in 0 1 2; do
            echo "Running experiment for $appliance ws=$window_size seed=$seed model=TCN_KL..."
            uv run -m scripts.run_one_expe \
                --dataset "REDD" \
                --sampling_rate "10s" \
                --appliance "$appliance" \
                --window_size "$window_size" \
                --name_model "TCN_KL" \
                --seed "$seed"
            echo "Done: $appliance ws=$window_size seed=$seed model=TCN_KL"
        done
    done
done
