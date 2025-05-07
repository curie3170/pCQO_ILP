#!/bin/bash

BASE_INPUT_DIR="/export3/curiekim/fobc/graphs/er_graphs/N_1500"
TARGET_EXEC="./external/pcqo_mis"
ROUNDS=(1000)
BASE_LOG_DIR="/export3/curiekim/fobc/fast_tune/er_graphs/N_1500"

mkdir -p "$BASE_LOG_DIR"

for d in 10 20 30 40 50 60 70; do
    INPUT_DIR="$BASE_INPUT_DIR/$d"
    LOG_DIR="$BASE_LOG_DIR/$d"
    mkdir -p "$LOG_DIR"

    # Select 3 random graphs
    mapfile -t SELECTED_FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.clq" | shuf -n 3)

    for filepath in "${SELECTED_FILES[@]}"; do
        for round in "${ROUNDS[@]}"; do
            log_file="$LOG_DIR/$(basename "$filepath" .clq)_tuningiter$round.log"
            $TARGET_EXEC "$filepath" "$round" &> "$log_file"
        done
    done
done