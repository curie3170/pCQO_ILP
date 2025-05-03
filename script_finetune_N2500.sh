#!/bin/bash

INPUT_DIR="/export2/curiekim/fobc/graphs/er_graphs/N_2500"
TARGET_EXEC="./external/pcqo_mis"
ROUNDS=(1000 5000)
LOG_DIR="/export2/curiekim/fobc/fast_tune/er_graphs/N_2500"


mkdir -p "$LOG_DIR"

for filepath in "$INPUT_DIR"/*.clq; do
    for round in "${ROUNDS[@]}"; do
        log_file="$LOG_DIR/$(basename "$filepath" .clq)_tuningiter$round.log"
        $TARGET_EXEC "$filepath" "$round" &> "$log_file"
    done
done