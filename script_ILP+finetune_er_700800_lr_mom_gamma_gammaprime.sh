#!/bin/bash

#1e-05,0.7,500,7
SUBDIR=$1
LR=$2
MOM=$3
GAMMA=$4
GAMMA_PRIME=$5
# 0.00001 → "0.00001"
LR_FMT=$(printf "%.10f" "$LR" | sed -E 's/0+$//; s/\.$//')
if [ $# -ne 5 ]; then
    echo "Usage: $0 <subdirectory under fobc/graphs/> <lr> <mom> <gamma> <gamma_prime>"
    echo "Example: $0 er_700-800 0.00001 0.7 500 7"
    exit 1
fi


BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRAPH_DIR="${BASE_DIR}/../fobc/graphs/${SUBDIR}"
# FAST_TUNE_DIR="${BASE_DIR}/../fobc/fast_tune/${SUBDIR}"
GUROBI_DIR="${BASE_DIR}/../fobc/gurobi/${SUBDIR}"
LOG_DIR="${BASE_DIR}/../fobc/ILP+fast_tune/${SUBDIR}_${LR_FMT}_${MOM}_${GAMMA}_${GAMMA_PRIME}" 
EXEC_PATH="${BASE_DIR}/external/pcqo_mis"

mkdir -p "$LOG_DIR"
echo "$GRAPH_DIR"
for graph_path in "$GRAPH_DIR"/*.clq; do
    base_name=$(basename "$graph_path" .clq)
    #csv_path="${FAST_TUNE_DIR}/${base_name}_tuningiter${TUNEITER}.csv"
    txt_path="${GUROBI_DIR}/${base_name}.txt"
    log_path="${LOG_DIR}/${base_name}.log"
    log_path_gurobi="${LOG_DIR}/${base_name}_GurobiInit.log"

    # Skip if required files are missing
    # if [ ! -f "$csv_path" ]; then
    #     echo "Skipping $base_name: CSV not found."
    #     continue
    # fi

    if [ ! -f "$txt_path" ]; then
        echo "Skipping $base_name: TXT not found."
        continue
    fi

    # Read last line from CSV and extract hyperparameters
    # IFS=',' read -r LEARNING_RATE MOMENTUM GAMMA GAMMA_PRIME BATCH_SIZE STD _ _ _ _<<< $(tail -n 1 "$csv_path")
    
    # Read binary string
    BINARY_STRING=$(cat "$txt_path")

    # Construct and run command
    #CMD="${EXEC_PATH} ${graph_path} ${LEARNING_RATE} ${MOMENTUM} 50000 500 ${GAMMA} ${GAMMA_PRIME} ${BATCH_SIZE} ${STD} 1 \"${BINARY_STRING}\""
    #5000
    CMD1="${EXEC_PATH} ${graph_path} ${LR} ${MOM}  500000 500 ${GAMMA} ${GAMMA_PRIME} 256 2.25 10"
    CMD2="${EXEC_PATH} ${graph_path} ${LR} ${MOM} 500000 500 ${GAMMA} ${GAMMA_PRIME} 256 2.25 10 \"${BINARY_STRING}\""
    #1000_repeated_fixed_sample
    echo "Running on ${base_name}..."
    eval $CMD1 | tee "$log_path"
    eval $CMD2 | tee "$log_path_gurobi"
done