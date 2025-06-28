#!/bin/bash

# --- Configuration Section ---

# Set Neo4j Environment Variables (Consider externalizing these for production)
# Uses existing env var if set, otherwise defaults.
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-Martin1007Wang}" # Production: Use env vars or a secure config manager

# Set Base Work Directory
BASE_WORKDIR="/mnt/wangjingxiong/think_on_graph" # Main project directory

# === MODIFICATION: Define all dataset names to process ===
ALL_DATASET_NAMES=("webqsp" "cwq") # Example: add other dataset names here e.g., "dataset2"
# === END MODIFICATION ===

SPLIT="train" # Assuming split is the same for all datasets

# Input path data base directory (where individual dataset path_data.json files are located)
PATH_DATA_INPUT_BASE="${BASE_WORKDIR}/data/paths"

# Output base directory for the combined create_preference_dataset.py script
PREFERENCE_DATASET_OUTPUT_BASE="${BASE_WORKDIR}/data/preference_dataset" # Changed from naive_preference_dataset to match your latest script

# Base name for the COMBINED preference dataset outputs
# This name will be part of the output directory structure created by the Python script.
COMBINED_PREFERENCE_BASE_OUTPUT_NAME="${SPLIT}" # Using the split name (e.g., "train")

# Python script names (assuming they are in a 'workflow' subdirectory or accessible via PATH)
# PREPARE_PATHS_SCRIPT="workflow/prepare_paths.py" # Assuming this is run separately
CREATE_PREFERENCE_SCRIPT="workflow/create_naive_preference_dataset.py" # Your Python script

# --- DPO Preference Generation Configurations ---
CANDIDATE_STRATEGIES=("pn_only") # Add more strategies if needed: "kg_allhop" "pn_kg_supplement"
POSITIVE_SOURCE_FIELDS=("shortest_paths") # Add more sources if needed: "positive_paths"

MAX_SELECTION_COUNT=5
NUM_SAMPLES_PREFERENCE=-1 # -1 for all

# Log Directory
LOG_BASE_DIR="${BASE_WORKDIR}/logs_preference_pipeline" # Centralized logs for this pipeline

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

# Navigate to the working directory or exit
cd "$BASE_WORKDIR" || { log_error "Base working directory '$BASE_WORKDIR' does not exist! Exiting."; exit 1; }

# Create Log Directory
mkdir -p "$LOG_BASE_DIR" || { log_error "Failed to create log directory '$LOG_BASE_DIR'! Exiting."; exit 1; }

# Current Timestamp (for unique log file names if script is run multiple times quickly)
OVERALL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

log_info "===== STARTING COMBINED PREFERENCE DATASET PROCESSING PIPELINE ====="
log_info "Processing datasets to combine: ${ALL_DATASET_NAMES[*]}"
log_info "Split: ${SPLIT}"
log_info "Base Work Directory: ${BASE_WORKDIR}"
log_info "Logging to directory: ${LOG_BASE_DIR}"
echo "------------------------------------------------------------------"

# STEP 1: Collect all input path files
log_info "STEP 1: Collecting prerequisite path data files for each dataset..."
ALL_INPUT_PATH_FILES=() # Initialize an empty array to store paths

for DATASET_NAME_ITER in "${ALL_DATASET_NAMES[@]}"; do
    log_info "--- Checking path data for DATASET: ${DATASET_NAME_ITER} ---"

    # Construct the expected path to the path_data.json file for the current dataset
    CURRENT_PATH_DATA_DIR="${PATH_DATA_INPUT_BASE}/${DATASET_NAME_ITER}_${SPLIT}"
    CURRENT_GENERATED_PATH_DATA_FILE="${CURRENT_PATH_DATA_DIR}/path_data.json"

    if [ ! -f "$CURRENT_GENERATED_PATH_DATA_FILE" ]; then
        log_warning "Path data file '$CURRENT_GENERATED_PATH_DATA_FILE' for dataset '${DATASET_NAME_ITER}' not found!"
        log_warning "This dataset will be SKIPPED for combined processing."
        # Optionally, you could choose to exit if any file is missing:
        # log_error "Exiting because a required path data file is missing."
        # exit 1
        continue # Skip to next dataset
    else
        log_info "Found path data file for ${DATASET_NAME_ITER}: $CURRENT_GENERATED_PATH_DATA_FILE"
        ALL_INPUT_PATH_FILES+=("$CURRENT_GENERATED_PATH_DATA_FILE") # Add found file to the list
    fi
    echo "------------------------------------------------------------------"
done

# Check if any files were collected
if [ ${#ALL_INPUT_PATH_FILES[@]} -eq 0 ]; then
    log_error "No input path files were found or collected. Cannot proceed with combined preference dataset creation."
    exit 1
fi

log_info "Collected input files for combined processing: ${ALL_INPUT_PATH_FILES[*]}"
echo "------------------------------------------------------------------"

# STEP 2: Create ONE Preference Dataset by combining all collected input files
log_info "STEP 2: Running ${CREATE_PREFERENCE_SCRIPT} to create a single COMBINED preference dataset..."

# Check if the Python script exists
if [ ! -f "$CREATE_PREFERENCE_SCRIPT" ]; then
    log_error "Python script '$CREATE_PREFERENCE_SCRIPT' not found! Exiting."
    exit 1
fi

TOTAL_CONFIGS_COMBINED=$(( ${#CANDIDATE_STRATEGIES[@]} * ${#POSITIVE_SOURCE_FIELDS[@]} ))
PROCESSED_CONFIGS_COMBINED=0
FAILED_CONFIGS_COMBINED=0

for strategy in "${CANDIDATE_STRATEGIES[@]}"; do
    for positive_source in "${POSITIVE_SOURCE_FIELDS[@]}"; do
        PROCESSED_CONFIGS_COMBINED=$((PROCESSED_CONFIGS_COMBINED + 1))
        echo ""
        log_info "--- Processing Combined Configuration ${PROCESSED_CONFIGS_COMBINED}/${TOTAL_CONFIGS_COMBINED} ---"
        log_info "Candidate Strategy: ${strategy}"
        log_info "Positive Source Field: ${positive_source}"

        CURRENT_PREFERENCE_LOG="${LOG_BASE_DIR}/create_preference_COMBINED_${COMBINED_PREFERENCE_BASE_OUTPUT_NAME}_${strategy}_${positive_source}_${OVERALL_TIMESTAMP}.log"
        log_info "Log file for this configuration: ${CURRENT_PREFERENCE_LOG}"

        # *** MODIFIED LINE TO MATCH PYTHON SCRIPT'S OUTPUT NAMING CONVENTION ***
        EXPECTED_OUTPUT_SUBDIR_NAME="${COMBINED_PREFERENCE_BASE_OUTPUT_NAME}_cand_${strategy}_pos_${positive_source}"
        FULL_EXPECTED_OUTPUT_DIR="${PREFERENCE_DATASET_OUTPUT_BASE}/${EXPECTED_OUTPUT_SUBDIR_NAME}"

        SAMPLING_ARGS=""
        # if [ -n "$ENABLE_RELATION_SAMPLING_FLAG" ]; then # Check if the flag variable is non-empty
        #     SAMPLING_ARGS="${ENABLE_RELATION_SAMPLING_FLAG} --relation_sampling_threshold ${RELATION_SAMPLING_THRESHOLD} --num_distractors_to_sample ${NUM_DISTRACTORS_TO_SAMPLE}"
        # fi

        # Note: The --input_files argument in the Python script expects nargs='+'
        # Passing the bash array "${ALL_INPUT_PATH_FILES[@]}" will expand to separate arguments.
        ( python "$CREATE_PREFERENCE_SCRIPT" \
            --input_files "${ALL_INPUT_PATH_FILES[@]}" \
            --output_path "$PREFERENCE_DATASET_OUTPUT_BASE" \
            --base_output_name "$COMBINED_PREFERENCE_BASE_OUTPUT_NAME" \
            --candidate_strategy "$strategy" \
            --positive_source_field "$positive_source" \
            --max_selection_count "$MAX_SELECTION_COUNT" \
            --neo4j_uri "$NEO4J_URI" \
            --neo4j_user "$NEO4J_USER" \
            --neo4j_password "$NEO4J_PASSWORD" \
            --num_samples "$NUM_SAMPLES_PREFERENCE" \
            ${SAMPLING_ARGS} \
        ) > "$CURRENT_PREFERENCE_LOG" 2>&1
        
        SCRIPT_EXIT_CODE=$?

        if check_command_status "${SCRIPT_EXIT_CODE}" \
            "COMBINED Preference dataset generation SUCCEEDED for [Strategy: ${strategy}, Source: ${positive_source}]" \
            "COMBINED Preference dataset generation FAILED for [Strategy: ${strategy}, Source: ${positive_source}]" \
            "$CURRENT_PREFERENCE_LOG"; then
            
            if [ -d "$FULL_EXPECTED_OUTPUT_DIR" ]; then
                 log_info "Output directory verified at: ${FULL_EXPECTED_OUTPUT_DIR}"
            else
                 log_warning "Output directory ${FULL_EXPECTED_OUTPUT_DIR} was NOT found, though script exited successfully. Please check script's output logic and naming consistency."
                 # Consider this a failure if the directory is essential and script should have created it
                 # FAILED_CONFIGS_COMBINED=$((FAILED_CONFIGS_COMBINED + 1)) 
            fi
        else
            FAILED_CONFIGS_COMBINED=$((FAILED_CONFIGS_COMBINED + 1))
        fi
        echo "----------------------------------------"
    done # end positive_source loop
done # end strategy loop

echo ""
log_info "===== OVERALL DATA PROCESSING PIPELINE SUMMARY ====="
log_info "Input path files used for combined processing: ${ALL_INPUT_PATH_FILES[*]}"
log_info "Combined Preference Dataset Generation Summary:"
log_info "Total configurations processed: ${PROCESSED_CONFIGS_COMBINED}"
log_info "Successful configurations: $((PROCESSED_CONFIGS_COMBINED - FAILED_CONFIGS_COMBINED))"
log_info "Failed configurations: ${FAILED_CONFIGS_COMBINED}"
log_info "Logs for preference dataset generation are in: ${LOG_BASE_DIR}"
log_info "Generated preference datasets (if successful) are under: ${PREFERENCE_DATASET_OUTPUT_BASE}"
echo "=================================================================="

if [ $FAILED_CONFIGS_COMBINED -gt 0 ]; then
    exit 1
else
    exit 0
fi
