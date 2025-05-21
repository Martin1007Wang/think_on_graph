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
INITIAL_DATA_INPUT_DIR="${BASE_WORKDIR}/data/processed/rmanluo_RoG-${DATASET_NAME}_${SPLIT}"

# Output base directory for prepare_paths.py
PATH_GENERATION_OUTPUT_BASE="${BASE_WORKDIR}/data/processed"

# Base name for the output of prepare_paths.py (will become a directory)
PATH_OUTPUT_DIR_NAME="${DATASET_NAME}_${SPLIT}" # This will result in path_data.json inside this dir

# Output base directory for the SFT dataset script
SFT_DATASET_OUTPUT_BASE="${BASE_WORKDIR}/data/naive_instruction_dataset" # Changed for SFT

# Base name for the SFT dataset outputs (suffixes will be added by Python script)
SFT_BASE_OUTPUT_NAME="${DATASET_NAME}_${SPLIT}" # Changed for SFT

# Python script names (assuming they are in a 'workflow' subdirectory)
PREPARE_PATHS_SCRIPT="workflow/prepare_paths.py"
CREATE_SFT_SCRIPT="workflow/create_naive_chat_dataset.py" 

# --- SFT Dataset Generation Configurations ---
CANDIDATE_STRATEGIES=("pn_only" "kg_allhop" "pn_kg_supplement")
POSITIVE_SOURCE_FIELDS=("positive_paths" "shortest_paths")
MAX_SELECTION_COUNT_SFT=3 # Max items in completion for SFT, can be tuned
NUM_SAMPLES_SFT=-1 # -1 for all, or set a number for testing
ENABLE_RELATION_SAMPLING_SFT="" # Set to "--enable_relation_sampling" to enable
RELATION_SAMPLING_THRESHOLD_SFT=20
NUM_DISTRACTORS_TO_SAMPLE_SFT=7

# --- Script Execution ---

# Navigate to the working directory
cd "$BASE_WORKDIR" || { echo "ERROR: Base working directory '$BASE_WORKDIR' does not exist! Exiting."; exit 1; }

# Create Log Directory
LOGDIR="${BASE_WORKDIR}/logs_sft" # Separate logs for SFT
mkdir -p "$LOGDIR"

# Current Timestamp (for log file names)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================================="
echo "===== STARTING SFT DATASET PROCESSING PIPELINE - $(date) ====="
echo "=================================================================="
echo "Dataset: ${DATASET_NAME}, Split: ${SPLIT}"
echo "Base Work Directory: ${BASE_WORKDIR}"
echo "Logging to directory: ${LOGDIR}"
echo "------------------------------------------------------------------"

# STEP 1: Generate Path Data (if not already generated)
# This step is run once for the dataset and split.
# The SFT script will use the same path_data.json as the DPO script.
GENERATED_PATH_DATA_FILE="${PATH_GENERATION_OUTPUT_BASE}/${PATH_OUTPUT_DIR_NAME}/path_data.json"

# Log file for prepare_paths.py (can be shared if run once, or specific if re-run)
PATH_LOG="${LOGDIR}/prepare_paths_${DATASET_NAME}_${SPLIT}_${TIMESTAMP}.log"

# --- UNCOMMENT AND RUN THIS SECTION IF path_data.json IS NOT YET GENERATED ---
# --- OR IF YOU NEED TO REGENERATE IT ---
# echo "STEP 1: Running ${PREPARE_PATHS_SCRIPT} to generate path data..."
# echo "Output for paths expected at: ${GENERATED_PATH_DATA_FILE}"
# echo "Log file for path generation: ${PATH_LOG}"
# echo "------------------------------------------------------------------"
# # Create the output directory for prepare_paths.py if it doesn't exist
# mkdir -p "${PATH_GENERATION_OUTPUT_BASE}/${PATH_OUTPUT_DIR_NAME}"
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
# PIPESTATUS_PREPARE_PATHS=${PIPESTATUS[0]}
# --- END OF PREPARE_PATHS.PY SECTION ---

# For this script, we'll assume path_data.json exists. If not, the user should run Step 1 first.
# For automated pipelines, you'd uncomment the section above and use PIPESTATUS_PREPARE_PATHS.
PIPELINE_STATUS_PREPARE_PATHS=0 # Assuming success if section is commented out
if [ ! -f "$GENERATED_PATH_DATA_FILE" ]; then
    echo "WARNING: Path data file '$GENERATED_PATH_DATA_FILE' not found."
    echo "If you haven't run prepare_paths.py yet, please do so or uncomment Step 1 in this script."
    # Set a status to indicate it wasn't found, if strict checking is needed before proceeding.
    PIPELINE_STATUS_PREPARE_PATHS=1
    # For now, we allow proceeding, but the SFT script will fail if this file is missing.
fi


# Check if path generation was successful (or if the file exists if Step 1 is commented out)
if [ ${PIPELINE_STATUS_PREPARE_PATHS} -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Path data generation failed or output file not found from Step 1."
    echo "Please check the log: $PATH_LOG (if Step 1 was run)"
    echo "Expected input file for Step 2: $GENERATED_PATH_DATA_FILE"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
elif [ -f "$GENERATED_PATH_DATA_FILE" ]; then
    echo "INFO: Using existing path data file: $GENERATED_PATH_DATA_FILE for SFT dataset generation."
else
    echo "ERROR: Critical input file $GENERATED_PATH_DATA_FILE for SFT generation is missing. Exiting."
    exit 1
fi
echo "------------------------------------------------------------------"


# STEP 2: Create SFT Datasets for different configurations
echo "STEP 2: Running ${CREATE_SFT_SCRIPT} to create SFT instruction datasets..."
echo "Using generated path data from: ${GENERATED_PATH_DATA_FILE}"
echo "Base output directory for SFT datasets: ${SFT_DATASET_OUTPUT_BASE}"
echo "------------------------------------------------------------------"

TOTAL_CONFIGS=$(( ${#CANDIDATE_STRATEGIES[@]} * ${#POSITIVE_SOURCE_FIELDS[@]} ))
CURRENT_CONFIG_COUNT=0
FAILED_SFT_CONFIGS=0

for strategy in "${CANDIDATE_STRATEGIES[@]}"; do
    for positive_source in "${POSITIVE_SOURCE_FIELDS[@]}"; do
        CURRENT_CONFIG_COUNT=$((CURRENT_CONFIG_COUNT + 1))
        echo ""
        echo "--- Processing SFT Configuration ${CURRENT_CONFIG_COUNT}/${TOTAL_CONFIGS} ---"
        echo "Candidate Strategy: ${strategy}"
        echo "Positive Source Field: ${positive_source}"

        CURRENT_SFT_LOG="${LOGDIR}/create_sft_${DATASET_NAME}_${SPLIT}_${strategy}_${positive_source}_${TIMESTAMP}.log"
        echo "Log file for this SFT configuration: ${CURRENT_SFT_LOG}"

        # The Python SFT script will create a subdirectory like:
        # ${SFT_DATASET_OUTPUT_BASE}/${SFT_BASE_OUTPUT_NAME}_cand_${strategy}_pos_${positive_source}
        EXPECTED_SFT_OUTPUT_SUBDIR="${SFT_BASE_OUTPUT_NAME}_cand_${strategy}_pos_${positive_source}"
        FULL_EXPECTED_SFT_OUTPUT_DIR="${SFT_DATASET_OUTPUT_BASE}/${EXPECTED_SFT_OUTPUT_SUBDIR}"

        SAMPLING_ARGS_SFT=""
        if [ "$ENABLE_RELATION_SAMPLING_SFT" == "--enable_relation_sampling" ]; then
            SAMPLING_ARGS_SFT="--enable_relation_sampling --relation_sampling_threshold ${RELATION_SAMPLING_THRESHOLD_SFT} --num_distractors_to_sample ${NUM_DISTRACTORS_TO_SAMPLE_SFT}"
        fi

        # Call the SFT Python script
        python "$CREATE_SFT_SCRIPT" \
            --input_path "$GENERATED_PATH_DATA_FILE" \
            --output_path "$SFT_DATASET_OUTPUT_BASE" \
            --base_output_name "$SFT_BASE_OUTPUT_NAME" \
            --candidate_strategy "$strategy" \
            --positive_source_field "$positive_source" \
            --max_selection_count "$MAX_SELECTION_COUNT_SFT" \
            --neo4j_uri "$NEO4J_URI" \
            --neo4j_user "$NEO4J_USER" \
            --neo4j_password "$NEO4J_PASSWORD" \
            --num_samples "$NUM_SAMPLES_SFT" \
            ${SAMPLING_ARGS_SFT} \
            2>&1 | tee "$CURRENT_SFT_LOG"
        
        SCRIPT_EXIT_CODE=${PIPESTATUS[0]}

        if [ ${SCRIPT_EXIT_CODE} -eq 0 ] && [ -d "$FULL_EXPECTED_SFT_OUTPUT_DIR" ]; then
            echo "SUCCESS: SFT dataset generated for [Strategy: ${strategy}, Source: ${positive_source}]"
            echo "Output at: ${FULL_EXPECTED_SFT_OUTPUT_DIR}"
        else
            echo "ERROR: SFT dataset creation FAILED for [Strategy: ${strategy}, Source: ${positive_source}] (Exit code: ${SCRIPT_EXIT_CODE})"
            echo "Please check the log: ${CURRENT_SFT_LOG}"
            echo "Expected output directory: ${FULL_EXPECTED_SFT_OUTPUT_DIR}"
            FAILED_SFT_CONFIGS=$((FAILED_SFT_CONFIGS + 1))
        fi
        echo "----------------------------------------"
    done
done

echo ""
echo "=================================================================="
echo "===== SFT DATASET PROCESSING PIPELINE SUMMARY - $(date) ====="
echo "=================================================================="
if [ -f "$PATH_LOG" ]; then # Check if PATH_LOG was actually created
    echo "Path data generation log (if run in this session): ${PATH_LOG}"
fi
if [ -f "$GENERATED_PATH_DATA_FILE" ]; then
    echo "Path data file used as input: ${GENERATED_PATH_DATA_FILE}"
else
    echo "Path data file: NOT FOUND OR ERROR OCCURRED."
fi
echo ""
echo "SFT Dataset Generation Summary:"
echo "Total configurations processed: ${CURRENT_CONFIG_COUNT}"
echo "Successful configurations: $((CURRENT_CONFIG_COUNT - FAILED_SFT_CONFIGS))"
echo "Failed configurations: ${FAILED_SFT_CONFIGS}"
echo "Logs for SFT dataset generation are in: ${LOGDIR} (prefixed with 'create_sft_')"
echo "Generated SFT datasets are under: ${SFT_DATASET_OUTPUT_BASE}"
echo "=================================================================="

if [ $FAILED_SFT_CONFIGS -gt 0 ]; then
    exit 1
else
    exit 0
fi