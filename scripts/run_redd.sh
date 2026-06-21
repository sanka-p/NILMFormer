#!/bin/bash

for appliance in "WashingMachine" "Dishwasher" "Microwave" "Fridge"; do
    for window_size in 128 256 512; do
        for seed in 0 1 2; do
            for model in "NILMFormer" "BERT4NILM" "BiLSTM" "BiGRU" "CNN1D" "DAResNet"; do
                echo "Running experiment for $appliance ws=$window_size seed=$seed model=$model..."
                uv run -m scripts.run_one_expe \
                    --dataset "REDD" \
                    --sampling_rate "10s" \
                    --appliance "$appliance" \
                    --window_size "$window_size" \
                    --name_model "$model" \
                    --seed "$seed"
                echo "Done: $appliance ws=$window_size seed=$seed model=$model"
            done
        done
    done
done
