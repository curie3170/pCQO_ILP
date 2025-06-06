# #!/bin/bash

# INPUT_DIR="/export2/curiekim/fobc/graphs/er_700-800"
# TARGET_EXEC="./external/pcqo_mis"
# ROUNDS=(1000 5000)
# LOG_DIR="/export2/curiekim/fobc/fast_tune/er_700-800"


# mkdir -p "$LOG_DIR"

# for filepath in "$INPUT_DIR"/*.clq; do
#     for round in "${ROUNDS[@]}"; do
#         log_file="$LOG_DIR/$(basename "$filepath" .clq)_tuningiter"$round".log"
#         $TARGET_EXEC "$filepath" "$round" &> "$log_file"
#     done
# done
#!/bin/bash

INPUT_DIR="/export3/curiekim/fobc/graphs/er_700-800"
TARGET_EXEC="./external/pcqo_mis"
ROUNDS=(1000)
LOG_DIR="/export3/curiekim/fobc/fast_tune/er_700-800"

mkdir -p "$LOG_DIR"

# Select 5 random graphs
mapfile -t SELECTED_FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.clq" | shuf -n 5)

for filepath in "${SELECTED_FILES[@]}"; do
    for round in "${ROUNDS[@]}"; do
        log_file="$LOG_DIR/$(basename "$filepath" .clq)_tuningiter${round}_unfixed.log"
        $TARGET_EXEC "$filepath" "$round" &> "$log_file"
    done
done