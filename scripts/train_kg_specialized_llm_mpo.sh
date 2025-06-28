#!/bin/bash

# ==============================================================================
#  MPO Fine-tuning Launcher Script for KG-Specialized LLM
# ==============================================================================

set -e
set -u

# --- Environment & Path Configurations ---
export BASE_PROJECT_DIR="${BASE_PROJECT_DIR:-/mnt/wangjingxiong/think_on_graph}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="${BASE_PROJECT_DIR}"

# --- Model Configuration ---
export MODEL_PATH="/mnt/data/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit/snapshots/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"

# --- Dataset Configurations ---
DATASET_NAME="train"
DATASET_TYPE="preference_dataset_v2"
CANDIDATE_STRATEGY="pn_only"
POSITIVE_SOURCE_FIELD="shortest_paths"
DATASET_CONFIG_TAG="cand_${CANDIDATE_STRATEGY}_pos_${POSITIVE_SOURCE_FIELD}/hf_dataset"
DATA_PATH_LIST="${BASE_PROJECT_DIR}/data/${DATASET_TYPE}/${DATASET_CONFIG_TAG}"

# --- Script Path Configuration ---
# Assuming your Python script is named mpo_training_script.py now
PYTHON_SCRIPT_PATH="${BASE_PROJECT_DIR}/workflow/finetune_kg_specialized_llm_mpo.py"


# --- Core Training Hyperparameters ---
USE_PEFT=True
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Model Loading
LOAD_IN_4BIT=False
LOAD_IN_8BIT=False

# Training Strategy
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=True
LEARNING_RATE=1e-5
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
BF16=True
ATTN_IMPLEMENTATION="flash_attention_2"

# MPO Specifics (Updated from DPO)
BETA=0.1
SFT_LOSS_WEIGHT=0.05 # New MPO parameter for IPO-style regularization
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512

# Logging & Saving
SAVE_MERGED=True
SAVE_STRATEGY="epoch"
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=10
REPORT_TO="wandb"

# --- Dynamic Naming for Outputs & Logging ---
MODEL_BASENAME=$(basename "$MODEL_PATH")

RUN_SPECIFIC_NAME="${MODEL_BASENAME}"
RUN_SPECIFIC_NAME+="_ep${NUM_TRAIN_EPOCHS}"
RUN_SPECIFIC_NAME+="_loss-mpo" # Indicate MPO loss
RUN_SPECIFIC_NAME+="_b${BETA}"
RUN_SPECIFIC_NAME+="_sft${SFT_LOSS_WEIGHT}" # Add SFT weight to name
RUN_SPECIFIC_NAME+="_lr${LEARNING_RATE}"
if [ "${USE_PEFT}" = "True" ]; then
    RUN_SPECIFIC_NAME+="_lora-r${LORA_R}-a${LORA_ALPHA}"
fi
RUN_SPECIFIC_NAME+="_bf16"

SAVE_PATH="${BASE_PROJECT_DIR}/mpo_models/${DATASET_CONFIG_TAG}/${RUN_SPECIFIC_NAME}"
WANDB_RUN_NAME="${DATASET_CONFIG_TAG}/${RUN_SPECIFIC_NAME}"

# --- Sanity Checks ---
if [ ! -d "${BASE_PROJECT_DIR}" ]; then
  echo "ERROR: BASE_PROJECT_DIR not found at '${BASE_PROJECT_DIR}'" >&2
  exit 1
fi
if [ ! -f "${PYTHON_SCRIPT_PATH}" ]; then
  echo "ERROR: Python script not found at '${PYTHON_SCRIPT_PATH}'" >&2
  exit 1
fi
if [ ! -d "${DATA_PATH_LIST}" ]; then
  echo "ERROR: Dataset directory not found at '${DATA_PATH_LIST}'" >&2
  exit 1
fi
if [ "${REPORT_TO}" = "wandb" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARNING: REPORT_TO is 'wandb', but WANDB_API_KEY is not set. WandB logging may fail." >&2
fi

# --- Display Configuration ---
echo "========================================================"
echo "🚀 Starting MPO Fine-tuning with Accelerate..."
echo "========================================================"
echo "🔹 Model:               ${MODEL_PATH}"
echo "🔹 Dataset:             ${DATASET_CONFIG_TAG}"
echo "🔹 Output Path:         ${SAVE_PATH}"
echo "🔹 WandB Run Name:      ${WANDB_RUN_NAME}"
echo "🔹 LoRA Config:         R=${LORA_R}, Alpha=${LORA_ALPHA}"
echo "🔹 MPO Config:          Beta=${BETA}, SFT Weight=${SFT_LOSS_WEIGHT}"
echo "🔹 Training Config:     Epochs=${NUM_TRAIN_EPOCHS}, LR=${LEARNING_RATE}"
echo "🔹 Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) (single-node)"
echo "========================================================"

# --- Build Command using an Array (More Robust) ---
CMD_ARGS=(
    "--data_path_list" "${DATA_PATH_LIST}"
    "--model_name_or_path" "${MODEL_PATH}"
    "--output_dir" "${SAVE_PATH}"

    # PEFT/LoRA
    "--use_peft" "${USE_PEFT}"
    "--lora_r" "${LORA_R}"
    "--lora_alpha" "${LORA_ALPHA}"
    "--lora_dropout" "${LORA_DROPOUT}"
    "--target_modules" "${TARGET_MODULES}"
    "--save_merged" "${SAVE_MERGED}"

    # Model Loading
    "--load_in_4bit" "${LOAD_IN_4BIT}"
    "--load_in_8bit" "${LOAD_IN_8BIT}"

    # TrainingArguments
    "--bf16" "${BF16}"
    "--num_train_epochs" "${NUM_TRAIN_EPOCHS}"
    "--per_device_train_batch_size" "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    "--gradient_accumulation_steps" "${GRADIENT_ACCUMULATION_STEPS}"
    "--eval_strategy" "no"
    "--save_strategy" "${SAVE_STRATEGY}"
    "--save_total_limit" "${SAVE_TOTAL_LIMIT}"
    "--learning_rate" "${LEARNING_RATE}"
    "--weight_decay" "${WEIGHT_DECAY}"
    "--warmup_ratio" "${WARMUP_RATIO}"
    "--lr_scheduler_type" "${LR_SCHEDULER_TYPE}"
    "--logging_steps" "${LOGGING_STEPS}"
    "--report_to" "${REPORT_TO}"
    "--run_name" "${WANDB_RUN_NAME}"
    "--gradient_checkpointing" "${GRADIENT_CHECKPOINTING}"
    "--dataloader_num_workers" "0"
    "--dataloader_pin_memory" "true"

    # MPO Specific Arguments (Updated)
    "--beta" "${BETA}"
    "--sft_loss_weight" "${SFT_LOSS_WEIGHT}"
    "--max_length" "${MAX_LENGTH}"
    "--max_prompt_length" "${MAX_PROMPT_LENGTH}"
)

if [ -n "${ATTN_IMPLEMENTATION:-}" ]; then
    CMD_ARGS+=("--attn_implementation" "${ATTN_IMPLEMENTATION}")
fi

# --- Launch Training ---
echo "INFO: Executing command..."
# Note: Ensure your accelerate config file is correctly set up.
accelerate launch --config_file accelerate_configs/single_gpu.yaml "${PYTHON_SCRIPT_PATH}" "${CMD_ARGS[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "✅ MPO Fine-tuning completed successfully."
    echo "✅ Model saved to: ${SAVE_PATH}"
    echo "--------------------------------------------------------"
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "❌ MPO Fine-tuning FAILED with exit code $EXIT_CODE."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

exit $EXIT_CODE
