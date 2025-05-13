#!/bin/bash

# --- Configuration Section ---

# Set Neo4j Environment Variables (Consider externalizing these for production)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="Martin1007Wang" # Production: Use env vars or a secure config manager

# Set Base Work Directory
BASE_WORKDIR="/mnt/wangjingxiong/think_on_graph" # Main project directory

# Dataset and Split Configuration
DATASET_NAME="webqsp"
SPLIT="train"

# Input for prepare_paths.py (Output of a previous step, or initial dataset)
# Assuming rmanluo_RoG-webqsp_train is a directory containing the data for prepare_paths.py
INITIAL_DATA_INPUT_DIR="${BASE_WORKDIR}/data/processed/rmanluo_RoG-${DATASET_NAME}_${SPLIT}"

# Output base directory for prepare_paths.py
PATH_GENERATION_OUTPUT_BASE="${BASE_WORKDIR}/data/processed" # Changed to be more specific

# Base name for the output of prepare_paths.py (will become a directory)
PATH_OUTPUT_DIR_NAME="${DATASET_NAME}_${SPLIT}"

# Output base directory for create_preference_dataset_with_label.py
PREFERENCE_DATASET_OUTPUT_BASE="${BASE_WORKDIR}/data/preference_dataset"

# Base name for the preference dataset outputs (suffixes will be added by Python script)
PREFERENCE_BASE_OUTPUT_NAME="${DATASET_NAME}_${SPLIT}"

# Python script names (assuming they are in a 'workflow' subdirectory)
PREPARE_PATHS_SCRIPT="workflow/prepare_paths.py"
CREATE_PREFERENCE_SCRIPT="workflow/create_preference_dataset.py" # Your Python script

# --- DPO Preference Generation Configurations ---
# Define strategies and positive source fields to iterate over
# Ensure these values match the choices in your Python script's argparse
CANDIDATE_STRATEGIES=("pn_only" "kg_allhop" "pn_kg_supplement")
POSITIVE_SOURCE_FIELDS=("positive_paths" "shortest_paths")

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

# Current Timestamp (for log file names)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log file for prepare_paths.py
PATH_LOG="${LOGDIR}/prepare_paths_${DATASET_NAME}_${SPLIT}_${TIMESTAMP}.log"

echo "=================================================================="
echo "===== STARTING DATA PROCESSING PIPELINE - $(date) ====="
echo "=================================================================="
echo "Dataset: ${DATASET_NAME}, Split: ${SPLIT}"
echo "Base Work Directory: ${BASE_WORKDIR}"
echo "Logging to directory: ${LOGDIR}"
echo "------------------------------------------------------------------"

# STEP 1: Generate Path Data (if not already generated)
# This step is run once for the dataset and split.
GENERATED_PATH_DATA_FILE="${PATH_GENERATION_OUTPUT_BASE}/${PATH_OUTPUT_DIR_NAME}/path_data.json" # Expected output

# echo "STEP 1: Running ${PREPARE_PATHS_SCRIPT} to generate path data..."
# echo "Output for paths expected at: ${GENERATED_PATH_DATA_FILE}"
# echo "Log file for path generation: ${PATH_LOG}"
# echo "------------------------------------------------------------------"

# # Create the output directory for prepare_paths.py if it doesn't exist
# mkdir -p "${PATH_GENERATION_OUTPUT_BASE}/${PATH_OUTPUT_DIR_NAME}"

# # Call prepare_paths.py
# python "$PREPARE_PATHS_SCRIPT" \
#     --data_path "$INITIAL_DATA_INPUT_DIR" \
#     --dataset_name "$DATASET_NAME" \
#     --split "$SPLIT" \
#     --output_path "$PATH_GENERATION_OUTPUT_BASE" \
#     --output_name "$PATH_OUTPUT_DIR_NAME" \
#     --neo4j_uri "$NEO4J_URI" \
#     --neo4j_user "$NEO4J_USER" \
#     --neo4j_password "$NEO4J_PASSWORD" \
#     --max_path_length 3 \
#     --top_k_relations 5 \
#     --max_pairs 5 \
#     --max_negatives_per_pair 5 \
#     --num_threads 16 \
#     --num_samples -1 \
#     2>&1 | tee "$PATH_LOG"

# Check if path generation was successful
if [ ${PIPESTATUS[0]} -ne 0 ] || [ ! -f "$GENERATED_PATH_DATA_FILE" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Path data generation failed or output file not found."
    echo "Please check the log: $PATH_LOG"
    echo "Expected output file: $GENERATED_PATH_DATA_FILE"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
else
    echo "SUCCESS: Path data generated successfully: $GENERATED_PATH_DATA_FILE"
fi
echo "------------------------------------------------------------------"


# STEP 2: Create Preference Datasets for different configurations
echo "STEP 2: Running ${CREATE_PREFERENCE_SCRIPT} to create preference datasets..."
echo "Using generated path data from: ${GENERATED_PATH_DATA_FILE}"
echo "Base output directory for preference datasets: ${PREFERENCE_DATASET_OUTPUT_BASE}"
echo "------------------------------------------------------------------"

TOTAL_CONFIGS=$(( ${#CANDIDATE_STRATEGIES[@]} * ${#POSITIVE_SOURCE_FIELDS[@]} ))
CURRENT_CONFIG=0
FAILED_CONFIGS=0

for strategy in "${CANDIDATE_STRATEGIES[@]}"; do
    for positive_source in "${POSITIVE_SOURCE_FIELDS[@]}"; do
        CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
        echo ""
        echo "--- Processing Configuration ${CURRENT_CONFIG}/${TOTAL_CONFIGS} ---"
        echo "Candidate Strategy: ${strategy}"
        echo "Positive Source Field: ${positive_source}"

        # Dynamic log file for this specific preference dataset configuration
        CURRENT_PREFERENCE_LOG="${LOGDIR}/create_preference_${DATASET_NAME}_${SPLIT}_${strategy}_${positive_source}_${TIMESTAMP}.log"
        echo "Log file for this configuration: ${CURRENT_PREFERENCE_LOG}"

        # The Python script will create a subdirectory like:
        # ${PREFERENCE_DATASET_OUTPUT_BASE}/${PREFERENCE_BASE_OUTPUT_NAME}_cand_${strategy}_pos_${positive_source}
        EXPECTED_OUTPUT_SUBDIR="${PREFERENCE_BASE_OUTPUT_NAME}_cand_${strategy}_pos_${positive_source}"
        FULL_EXPECTED_OUTPUT_DIR="${PREFERENCE_DATASET_OUTPUT_BASE}/${EXPECTED_OUTPUT_SUBDIR}"


        # Optional: Add sampling args here if they were defined and uncommented above
        SAMPLING_ARGS=""
        # if [ -n "$ENABLE_RELATION_SAMPLING" ]; then
        #    SAMPLING_ARGS="$ENABLE_RELATION_SAMPLING --relation_sampling_threshold $RELATION_SAMPLING_THRESHOLD --num_distractors_to_sample $NUM_DISTRACTORS_TO_SAMPLE"
        # fi

        python "$CREATE_PREFERENCE_SCRIPT" \
            --input_path "$GENERATED_PATH_DATA_FILE" \
            --output_path "$PREFERENCE_DATASET_OUTPUT_BASE" \
            --base_output_name "$PREFERENCE_BASE_OUTPUT_NAME" \
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
            echo "SUCCESS: Preference dataset generated for [Strategy: ${strategy}, Source: ${positive_source}]"
            echo "Output at: ${FULL_EXPECTED_OUTPUT_DIR}"
        else
            echo "ERROR: Preference dataset creation FAILED for [Strategy: ${strategy}, Source: ${positive_source}]"
            echo "Please check the log: ${CURRENT_PREFERENCE_LOG}"
            echo "Expected output directory: ${FULL_EXPECTED_OUTPUT_DIR}"
            FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
        fi
        echo "----------------------------------------"
    done
done

echo ""
echo "=================================================================="
echo "===== DATA PROCESSING PIPELINE SUMMARY - $(date) ====="
echo "=================================================================="
echo "Path data generation log: ${PATH_LOG}"
if [ -f "$GENERATED_PATH_DATA_FILE" ]; then
    echo "Path data file: ${GENERATED_PATH_DATA_FILE}"
else
    echo "Path data file: NOT GENERATED OR ERROR OCCURRED."
fi
echo ""
echo "Preference Dataset Generation Summary:"
echo "Total configurations processed: ${CURRENT_CONFIG}"
echo "Successful configurations: $((CURRENT_CONFIG - FAILED_CONFIGS))"
echo "Failed configurations: ${FAILED_CONFIGS}"
echo "Logs for preference dataset generation are in: ${LOGDIR} (prefixed with 'create_preference_')"
echo "Generated preference datasets are under: ${PREFERENCE_DATASET_OUTPUT_BASE}"
echo "=================================================================="

if [ $FAILED_CONFIGS -gt 0 ]; then
    exit 1
else
    exit 0
fi