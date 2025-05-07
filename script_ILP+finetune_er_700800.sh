
# /export2/curiekim/pCQO-mis-benchmark% ./external/pcqo_mis /export2/curiekim/fobc/graphs/er_700-800/ER_700_800_0.15_0.clq 2500
# /export2/curiekim/pCQO-mis-benchmark% ./external/pcqo_mis /export2/curiekim/fobc/graphs/er_700-800/ER_700_800_0.15_0.clq 5000
# /export2/curiekim/pCQO-mis-benchmark% ./external/pcqo_mis /export2/curiekim/fobc/graphs/er_700-800/ER_700_800_0.15_0.clq 7500
# /export2/curiekim/pCQO-mis-benchmark% ./external/pcqo_mis /export2/curiekim/fobc/graphs/er_700-800/ER_700_800_0.15_0.clq 10000
# /export2/curiekim/pCQO-mis-benchmark% ./external/pcqo_mis /export2/curiekim/fobc/graphs/er_700-800/ER_700_800_0.15_0.clq 0.000009 0.9 50000 500 350 7 256 2.25 10 | tee /export2/curiekim/fobc/best_hyperparam/er_700-800/ER_700_800_0.15_0.log

# /export2/curiekim/fobc/fast_tune/er_700-800/ER_700_800_0.15_0_tuningiter1000.csv의 LearningRate,Momentum,Gamma,GammaPrime,BatchSize,Std를 읽고
# /export2/curiekim/fobc/gurobi/er_700-800/ER_700_800_0.15_0.txt를 "1 0 0 1 .."로 읽은 뒤, 

#!/bin/bash

# Usage: ./script.sh er_700-800 1000
SUBDIR=$1
TUNEITER=$2
if [ -z "$SUBDIR" ]; then
    echo "Usage: $0 <subdirectory name under fobc/graphs/ (e.g., er_700-800)> <tuning iteration>"
    exit 1
fi

if [ -z "$TUNEITER" ]; then
    echo "Usage: $0 <subdirectory name under fobc/graphs/ (e.g., er_700-800)> <tuning iteration>"
    exit 1
fi


BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRAPH_DIR="${BASE_DIR}/../fobc/graphs/${SUBDIR}"
FAST_TUNE_DIR="${BASE_DIR}/../fobc/fast_tune/${SUBDIR}"
GUROBI_DIR="${BASE_DIR}/../fobc/gurobi/${SUBDIR}"
LOG_DIR="${BASE_DIR}/../fobc/ILP+fast_tune/${SUBDIR}"
EXEC_PATH="${BASE_DIR}/external/pcqo_mis"
# GRAPH_DIR="/export2/curiekim/fobc/graphs/${SUBDIR}"
# FAST_TUNE_DIR="/export2/curiekim/fobc/fast_tune/${SUBDIR}"
# GUROBI_DIR="/export2/curiekim/fobc/gurobi/${SUBDIR}"
# LOG_DIR="/export2/curiekim/fobc/fast_tune+ILP/${SUBDIR}"
# EXEC_PATH="/export2/curiekim/pCQO-mis-benchmark/external/pcqo_mis"

mkdir -p "$LOG_DIR"
echo "$GRAPH_DIR"
for graph_path in "$GRAPH_DIR"/*.clq; do
    base_name=$(basename "$graph_path" .clq)
    csv_path="${FAST_TUNE_DIR}/${base_name}_tuningiter${TUNEITER}.csv"
    txt_path="${GUROBI_DIR}/${base_name}.txt"
    log_path="${LOG_DIR}/${base_name}_tuningiter${TUNEITER}_repeated_fixed.log"
    log_path_gurobi="${LOG_DIR}/${base_name}_tuningiter${TUNEITER}_repeated_fixed_GurobiInit.log"

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
    CMD1="${EXEC_PATH} ${graph_path} 0.00001 0.9 500000 500 500 5 256 2.25 10"
    CMD2="${EXEC_PATH} ${graph_path} 0.00001 0.9 500000 500 500 5 256 2.25 10 \"${BINARY_STRING}\""
    #1000_repeated_fixed_sample
    echo "Running on ${base_name}..."
    eval $CMD1 | tee "$log_path"
    eval $CMD2 | tee "$log_path_gurobi"
done