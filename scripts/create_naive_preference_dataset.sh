#!/bin/bash

# --- Configuration Section ---

# Set Neo4j Environment Variables (Consider externalizing these for production)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="Martin1007Wang" # Production: Use env vars or a secure config manager

# Set Base Work Directory
BASE_WORKDIR="/mnt/wangjingxiong/think_on_graph" # Main project directory

# === MODIFICATION: Define all dataset names to process ===
ALL_DATASET_NAMES=("webqsp" "cwq") # Example: add other dataset names here e.g., "dataset2"
# === END MODIFICATION ===

SPLIT="train" # Assuming split is the same for all datasets

# Output base directory for prepare_paths.py
PATH_GENERATION_OUTPUT_BASE="${BASE_WORKDIR}/data/processed"

# Output base directory for create_preference_dataset_with_label.py
PREFERENCE_DATASET_OUTPUT_BASE="${BASE_WORKDIR}/data/naive_preference_dataset"

# Base name for the COMBINED preference dataset outputs
# You can customize this, e.g., by joining dataset names or using a generic name
COMBINED_PREFERENCE_BASE_OUTPUT_NAME="train" # Example: using first dataset name as base for combined

# Python script names (assuming they are in a 'workflow' subdirectory)
PREPARE_PATHS_SCRIPT="workflow/prepare_paths.py"
CREATE_PREFERENCE_SCRIPT="workflow/create_naive_preference_dataset.py" # Your Python script

# --- DPO Preference Generation Configurations ---
CANDIDATE_STRATEGIES=("pn_only")
POSITIVE_SOURCE_FIELDS=("shortest_paths")

# Common parameters for create_preference_dataset_with_label.py
MAX_SELECTION_COUNT=5
NUM_SAMPLES_PREFERENCE=-1 # -1 for all
# Add sampling parameters if you want to enable them by default:
# ENABLE_RELATION_SAMPLING="--enable_relation_sampling" # Uncomment to enable
# RELATION_SAMPLING_THRESHOLD=25
# NUM_DISTRACTORS_TO_SAMPLE=10

# --- Script Execution ---

# Navigate to the working directory
cd "$BASE_WORKDIR" || { echo "Base working directory '$BASE_WORKDIR' does not exist!"; exit 1; }

# Create Log Directory
LOGDIR="${BASE_WORKDIR}/logs"
mkdir -p "$LOGDIR"

# Current Timestamp (for overall script run)
OVERALL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================================="
echo "===== STARTING OVERALL DATA PROCESSING PIPELINE - $(date) ====="
echo "=================================================================="
echo "Processing datasets to combine: ${ALL_DATASET_NAMES[*]}"
echo "Split: ${SPLIT}"
echo "Base Work Directory: ${BASE_WORKDIR}"
echo "Logging to directory: ${LOGDIR}"
echo "------------------------------------------------------------------"

# === MODIFICATION: Collect all input path files first ===
ALL_INPUT_PATH_FILES=() # Initialize an empty array to store paths

for DATASET_NAME in "${ALL_DATASET_NAMES[@]}"; do
    echo ""
    echo "--- Locating path data for DATASET: ${DATASET_NAME} ---"

    # --- Dataset-specific Configurations for path location ---
    PATH_OUTPUT_DIR_NAME="${DATASET_NAME}_${SPLIT}"
    GENERATED_PATH_DATA_FILE="${PATH_GENERATION_OUTPUT_BASE}/${PATH_OUTPUT_DIR_NAME}/path_data.json"
    PATH_LOG="${LOGDIR}/prepare_paths_${DATASET_NAME}_${SPLIT}_${OVERALL_TIMESTAMP}.log" # Log for individual path prep (if active)

    # STEP 1: Generate Path Data (if not already generated) for the current DATASET_NAME
    # This step is run once for the current dataset and split.
    # If STEP 1 is commented out, this script assumes GENERATED_PATH_DATA_FILE already exists.
    echo "STEP 1 (Per Dataset): Checking/Generating path data for ${DATASET_NAME}..."
    echo "Path data file expected at: ${GENERATED_PATH_DATA_FILE}"
    # echo "Log file for path generation (if active): ${PATH_LOG}" # Uncomment if STEP 1 is active

    # # Create the output directory for prepare_paths.py if it doesn't exist
    # mkdir -p "${PATH_GENERATION_OUTPUT_BASE}/${PATH_OUTPUT_DIR_NAME}"

    # # Call prepare_paths.py for the current DATASET_NAME
    # python "$PREPARE_PATHS_SCRIPT" \
    #     --data_path "${BASE_WORKDIR}/data/processed/rmanluo_RoG-${DATASET_NAME}_${SPLIT}" \ # INITIAL_DATA_INPUT_DIR specific to dataset
    #     --dataset_name "$DATASET_NAME" \
    #     --split "$SPLIT" \
    #     --output_path "$PATH_GENERATION_OUTPUT_BASE" \
    #     --output_name "$PATH_OUTPUT_DIR_NAME" \
    #     --neo4j_uri "$NEO4J_URI" \
    #     --neo4j_user "$NEO4J_USER" \
    #     --neo4j_password "$NEO4J_PASSWORD" \
    #     # ... other args for prepare_paths.py ...
    #     2>&1 | tee "$PATH_LOG"

    if [ ! -f "$GENERATED_PATH_DATA_FILE" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Path data file '$GENERATED_PATH_DATA_FILE' for dataset '${DATASET_NAME}' not found!"
        echo "This file is required for combined processing. Please ensure it exists or STEP 1 runs successfully."
        # echo "Check log (if STEP 1 was active): $PATH_LOG" # Uncomment if STEP 1 is active
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Decide if you want to exit or just skip this file
        # exit 1 # Exit if any file is missing
        echo "SKIPPING dataset ${DATASET_NAME} due to missing path file."
        continue # Skip to next dataset if a file is missing
    else
        echo "SUCCESS: Path data file found for ${DATASET_NAME}: $GENERATED_PATH_DATA_FILE"
        ALL_INPUT_PATH_FILES+=("$GENERATED_PATH_DATA_FILE") # Add found file to the list
    fi
    echo "------------------------------------------------------------------"
done

# Check if any files were collected
if [ ${#ALL_INPUT_PATH_FILES[@]} -eq 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: No input path files were found or collected. Cannot proceed with combined preference dataset creation."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

echo ""
echo "Collected input files for combined processing: ${ALL_INPUT_PATH_FILES[*]}"
echo "------------------------------------------------------------------"

# STEP 2: Create ONE Preference Dataset by combining all collected input files
echo "STEP 2: Running ${CREATE_PREFERENCE_SCRIPT} to create a single COMBINED preference dataset..."
echo "Using input files: ${ALL_INPUT_PATH_FILES[*]}"
echo "Base output directory for preference datasets: ${PREFERENCE_DATASET_OUTPUT_BASE}"
echo "Combined output base name: ${COMBINED_PREFERENCE_BASE_OUTPUT_NAME}"
echo "------------------------------------------------------------------"

TOTAL_CONFIGS_COMBINED=$(( ${#CANDIDATE_STRATEGIES[@]} * ${#POSITIVE_SOURCE_FIELDS[@]} ))
CURRENT_CONFIG_COMBINED=0
FAILED_CONFIGS_COMBINED=0

for strategy in "${CANDIDATE_STRATEGIES[@]}"; do
    for positive_source in "${POSITIVE_SOURCE_FIELDS[@]}"; do
        CURRENT_CONFIG_COMBINED=$((CURRENT_CONFIG_COMBINED + 1))
        echo ""
        echo "--- Processing Combined Configuration ${CURRENT_CONFIG_COMBINED}/${TOTAL_CONFIGS_COMBINED} ---"
        echo "Candidate Strategy: ${strategy}"
        echo "Positive Source Field: ${positive_source}"

        # Dynamic log file for this specific preference dataset configuration
        CURRENT_PREFERENCE_LOG="${LOGDIR}/create_preference_COMBINED_${COMBINED_PREFERENCE_BASE_OUTPUT_NAME}_${strategy}_${positive_source}_${OVERALL_TIMESTAMP}.log"
        echo "Log file for this configuration: ${CURRENT_PREFERENCE_LOG}"

        # Output subdir for the combined dataset
        EXPECTED_OUTPUT_SUBDIR="${COMBINED_PREFERENCE_BASE_OUTPUT_NAME}_cand_${strategy}_pos_${positive_source}"
        FULL_EXPECTED_OUTPUT_DIR="${PREFERENCE_DATASET_OUTPUT_BASE}/${EXPECTED_OUTPUT_SUBDIR}"

        SAMPLING_ARGS=""
        # if [ -n "$ENABLE_RELATION_SAMPLING" ]; then
        #     SAMPLING_ARGS="$ENABLE_RELATION_SAMPLING --relation_sampling_threshold $RELATION_SAMPLING_THRESHOLD --num_distractors_to_sample $NUM_DISTRACTORS_TO_SAMPLE"
        # fi

        python "$CREATE_PREFERENCE_SCRIPT" \
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
            $SAMPLING_ARGS \
            2>&1 | tee "$CURRENT_PREFERENCE_LOG"

        if [ ${PIPESTATUS[0]} -eq 0 ] && [ -d "$FULL_EXPECTED_OUTPUT_DIR" ]; then
            echo "SUCCESS: COMBINED Preference dataset generated for [Strategy: ${strategy}, Source: ${positive_source}]"
            echo "Output at: ${FULL_EXPECTED_OUTPUT_DIR}"
        else
            echo "ERROR: COMBINED Preference dataset creation FAILED for [Strategy: ${strategy}, Source: ${positive_source}]"
            echo "Please check the log: ${CURRENT_PREFERENCE_LOG}"
            echo "Expected output directory: ${FULL_EXPECTED_OUTPUT_DIR}"
            FAILED_CONFIGS_COMBINED=$((FAILED_CONFIGS_COMBINED + 1))
        fi
        echo "----------------------------------------"
    done # end positive_source loop
done # end strategy loop

echo ""
echo "=================================================================="
echo "===== OVERALL DATA PROCESSING PIPELINE COMPLETED - $(date) ====="
echo "=================================================================="
echo "Input path files used for combined processing: ${ALL_INPUT_PATH_FILES[*]}"
echo ""
echo "Combined Preference Dataset Generation Summary:"
echo "Total configurations processed: ${CURRENT_CONFIG_COMBINED}"
echo "Successful configurations: $((CURRENT_CONFIG_COMBINED - FAILED_CONFIGS_COMBINED))"
echo "Failed configurations: ${FAILED_CONFIGS_COMBINED}"
echo "Logs for preference dataset generation are in: ${LOGDIR} (prefixed with 'create_preference_COMBINED_')"
echo "Generated preference datasets are under: ${PREFERENCE_DATASET_OUTPUT_BASE}"
echo "=================================================================="

if [ $FAILED_CONFIGS_COMBINED -gt 0 ]; then
    exit 1
else
    exit 0
fi
