#!/bin/bash

# --- Configuration Section ---
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-Martin1007Wang}"
export PYTHONPATH=$PYTHONPATH:/mnt/wangjingxiong/think_on_graph

BASE_WORKDIR="/mnt/wangjingxiong/think_on_graph"
ALL_DATASET_NAMES=("webqsp" "cwq")
SPLIT="train" # Assuming split is the same for all datasets

PATH_DATA_INPUT_BASE="${BASE_WORKDIR}/data/paths"
# This is the top-level directory where strategy-specific folders (containing data from all inputs) will be created.
PREFERENCE_DATASET_OUTPUT_BASE="${BASE_WORKDIR}/data/preference_dataset_v2"

# Python script name
CREATE_PREFERENCE_SCRIPT="workflow/create_preference_dataset_v2.py"

# --- DPO Preference Generation Configurations ---
CANDIDATE_STRATEGIES=("pn_only") # "kg_allhop" "pn_kg_supplement"
POSITIVE_SOURCE_FIELDS=("shortest_paths") # "positive_paths"

MAX_SELECTION_COUNT=3
NUM_SAMPLES_PREFERENCE=-1

LOG_BASE_DIR="${BASE_WORKDIR}/logs_preference_pipeline_all_sources_combined"

# --- Helper Functions ---
log_info() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}
log_error() {
    echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2
}
log_warning() {
    echo "[WARNING] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

check_command_status() {
    local exit_code=$1
    local success_msg="$2"
    local error_msg="$3"
    local log_file="$4"

    if [ "${exit_code}" -eq 0 ]; then
        log_info "${success_msg}"
        return 0
    else
        log_error "${error_msg} (Exit code: ${exit_code})"
        if [ -n "$log_file" ]; then
            log_error "Please check the log: ${log_file}"
        fi
        return 1
    fi
}

# --- Script Execution ---
cd "$BASE_WORKDIR" || { log_error "Base working directory '$BASE_WORKDIR' does not exist! Exiting."; exit 1; }
mkdir -p "$LOG_BASE_DIR" || { log_error "Failed to create log directory '$LOG_BASE_DIR'! Exiting."; exit 1; }

OVERALL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

log_info "===== STARTING PREFERENCE DATASET PROCESSING PIPELINE (ALL SOURCES COMBINED PER STRATEGY) ====="
log_info "Datasets to process: ${ALL_DATASET_NAMES[*]} for split: ${SPLIT}"
log_info "Logging to directory: ${LOG_BASE_DIR}"
echo "------------------------------------------------------------------"

log_info "STEP 1: Collecting prerequisite path data files..."
ALL_INPUT_PATH_FILES_FOR_PYTHON=()
ALL_INPUT_DATASET_NAMES_FOR_PYTHON=() # These names are passed to Python for its internal use (e.g. logging, metadata)

for DATASET_NAME_ITER in "${ALL_DATASET_NAMES[@]}"; do
    log_info "--- Checking path data for DATASET: ${DATASET_NAME_ITER} (Split: ${SPLIT}) ---"
    CURRENT_PATH_DATA_DIR="${PATH_DATA_INPUT_BASE}/${DATASET_NAME_ITER}_${SPLIT}"
    CURRENT_GENERATED_PATH_DATA_FILE="${CURRENT_PATH_DATA_DIR}/path_data.json" # Ensure this filename is correct

    if [ ! -f "$CURRENT_GENERATED_PATH_DATA_FILE" ]; then
        log_warning "Path data file '$CURRENT_GENERATED_PATH_DATA_FILE' for dataset '${DATASET_NAME_ITER}_${SPLIT}' not found! This dataset will be SKIPPED."
        continue
    else
        log_info "Found path data file for ${DATASET_NAME_ITER}_${SPLIT}: $CURRENT_GENERATED_PATH_DATA_FILE"
        ALL_INPUT_PATH_FILES_FOR_PYTHON+=("$CURRENT_GENERATED_PATH_DATA_FILE")
        ALL_INPUT_DATASET_NAMES_FOR_PYTHON+=("${DATASET_NAME_ITER}_${SPLIT}") # Pass a unique name for each dataset
    fi
    echo "------------------------------------------------------------------"
done

if [ ${#ALL_INPUT_PATH_FILES_FOR_PYTHON[@]} -eq 0 ]; then
    log_error "No input path files were found or collected. Cannot proceed."
    exit 1
fi

log_info "Collected input files to be passed to Python script: ${ALL_INPUT_PATH_FILES_FOR_PYTHON[*]}"
log_info "Corresponding dataset names to be passed to Python script: ${ALL_INPUT_DATASET_NAMES_FOR_PYTHON[*]}"
echo "------------------------------------------------------------------"

if [ ! -f "$CREATE_PREFERENCE_SCRIPT" ]; then
    log_error "Python script '$CREATE_PREFERENCE_SCRIPT' not found! Exiting."
    exit 1
fi

TOTAL_STRATEGY_CONFIGS=$(( ${#CANDIDATE_STRATEGIES[@]} * ${#POSITIVE_SOURCE_FIELDS[@]} ))
PROCESSED_STRATEGY_CONFIGS=0
FAILED_STRATEGY_CONFIGS=0

log_info "STEP 2: Creating preference datasets for each strategy configuration (combining all input sources per strategy)..."
for current_strategy in "${CANDIDATE_STRATEGIES[@]}"; do
    for current_positive_source in "${POSITIVE_SOURCE_FIELDS[@]}"; do
        PROCESSED_STRATEGY_CONFIGS=$((PROCESSED_STRATEGY_CONFIGS + 1))
        echo "" # Newline for readability
        log_info "--- Processing Strategy Configuration ${PROCESSED_STRATEGY_CONFIGS}/${TOTAL_STRATEGY_CONFIGS} ---"
        log_info "Candidate Strategy: ${current_strategy}"
        log_info "Positive Source Field: ${current_positive_source}"

        # Log file for this specific strategy run (covers all datasets combined for this strategy)
        # Using SPLIT in the log name to distinguish if you run for different splits later
        CURRENT_RUN_LOG="${LOG_BASE_DIR}/create_preference_${SPLIT}_${current_strategy}_${current_positive_source}_${OVERALL_TIMESTAMP}.log"
        log_info "Log file for this configuration run: ${CURRENT_RUN_LOG}"

        # The Python script, given its argparse, determines its own output subdirectory structure
        # based on args.output_path, args.input_dataset_names (for *its internal loop*),
        # and the current strategy/source.
        # The bash script primarily cares that the Python script runs and where the top-level
        # PREFERENCE_DATASET_OUTPUT_BASE is.
        # The Python script (as I wrote it previously) creates:
        # <args.output_path>/<input_dataset_name>_cand_<strategy>_pos_<source>
        # So, multiple directories will be created by one run of the Python script if multiple inputs are given.

        SAMPLING_ARGS="" # Define your sampling args here if needed:
        # ENABLE_RELATION_SAMPLING_FLAG="--enable_relation_sampling" # Example
        # if [ -n "$ENABLE_RELATION_SAMPLING_FLAG" ]; then
        # SAMPLING_ARGS="${ENABLE_RELATION_SAMPLING_FLAG} --relation_sampling_threshold YOUR_THRESHOLD --num_distractors_to_sample YOUR_NUM_DISTRACTORS"
        # fi

        # Execute Python script
        # Note the quoting around array expansion for paths and names
        # Ensure all arguments are correctly passed.
        (
            python "$CREATE_PREFERENCE_SCRIPT" \
            --input_paths "${ALL_INPUT_PATH_FILES_FOR_PYTHON[@]}" \
            --input_dataset_names "${ALL_INPUT_DATASET_NAMES_FOR_PYTHON[@]}" \
            --output_path "$PREFERENCE_DATASET_OUTPUT_BASE" \
            --candidate_strategy "$current_strategy" \
            --positive_source_field "$current_positive_source" \
            --max_selection_count "$MAX_SELECTION_COUNT" \
            --neo4j_uri "$NEO4J_URI" \
            --neo4j_user "$NEO4J_USER" \
            --neo4j_password "$NEO4J_PASSWORD" \
            --num_samples "$NUM_SAMPLES_PREFERENCE" \
            ${SAMPLING_ARGS}
        ) > "$CURRENT_RUN_LOG" 2>&1
        SCRIPT_EXIT_CODE=$?

        if check_command_status "${SCRIPT_EXIT_CODE}" \
            "Python script for [Strategy: ${current_strategy}, Source: ${current_positive_source}] COMPLETED." \
            "Python script for [Strategy: ${current_strategy}, Source: ${current_positive_source}] FAILED." \
            "$CURRENT_RUN_LOG"; then
            # Success means the script ran. Python script itself handles internal success/failure per dataset.
            log_info "Python script finished. Outputs for this strategy are under: ${PREFERENCE_DATASET_OUTPUT_BASE}"
            log_info "The Python script should have created subdirectories there based on input dataset names and the current strategy."
        else
            FAILED_STRATEGY_CONFIGS=$((FAILED_STRATEGY_CONFIGS + 1))
        fi
        echo "----------------------------------------"
    done # end positive_source loop
done # end strategy loop

echo ""
log_info "===== OVERALL PREFERENCE DATASET PROCESSING PIPELINE SUMMARY ====="
log_info "Input path files passed to Python script runs: ${ALL_INPUT_PATH_FILES_FOR_PYTHON[*]}"
log_info "Preference Dataset Generation Summary (Python script runs):"
log_info "Total strategy configurations attempted: ${PROCESSED_STRATEGY_CONFIGS}"
log_info "Python script runs without error: $((PROCESSED_STRATEGY_CONFIGS - FAILED_STRATEGY_CONFIGS))"
log_info "Python script runs with errors: ${FAILED_STRATEGY_CONFIGS}"
log_info "Logs for each strategy run are in: ${LOG_BASE_DIR}"
log_info "Generated preference datasets (if successful) are under: ${PREFERENCE_DATASET_OUTPUT_BASE}"
log_info "Please check individual logs and output directories for per-dataset status within each strategy run."
echo "=================================================================="

if [ $FAILED_STRATEGY_CONFIGS -gt 0 ]; then
    exit 1
else
    exit 0
fi