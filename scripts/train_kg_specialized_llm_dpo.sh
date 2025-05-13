#!/bin/bash

# Shell 执行选项：如果任何命令失败，则立即退出；如果使用了未设置的变量，则报错
set -e
set -u

export BASE_PROJECT_DIR="/mnt/wangjingxiong/think_on_graph"
export MODEL_PATH="/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

if [ -z "${BASE_PROJECT_DIR:-}" ]; then
  echo "ERROR: BASE_PROJECT_DIR is not set. Please set it as an environment variable or in the script."
  exit 1
fi
if [ -z "${MODEL_PATH:-}" ]; then
  echo "ERROR: MODEL_PATH is not set. Please set it as an environment variable or in the script."
  exit 1
fi

# --- Environment Settings ---
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1

# --- Path Configurations ---
DATASET_NAME="webqsp"
CANDIDATE_STRATEGY="pn_only"
POSITIVE_SOURCE_FIELD="shortest_paths"
DATA_PATH_LIST="${BASE_PROJECT_DIR}/data/preference_dataset/${DATASET_NAME}_train_cand_${CANDIDATE_STRATEGY}_pos_${POSITIVE_SOURCE_FIELD}"

PYTHON_SCRIPT_PATH="workflow/finetune_kg_specialized_llm_dpo.py"
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_zero2.yaml"

# --- Training & Model Configuration ---
# LoRA Configuration
USE_PEFT=True
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Quantization Configuration
LOAD_IN_4BIT=False
LOAD_IN_8BIT=True
# BNB 4-bit参数（仅当LOAD_IN_4BIT=True时传递）
BNB_4BIT_QUANT_TYPE_ARG="nf4"
BNB_4BIT_COMPUTE_DTYPE_ARG="bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT_ARG=True


# Training Hyperparameters
NUM_TRAIN_EPOCHS=2 # DPO的Epoch通常很少
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=True
LEARNING_RATE=5e-5 # DPO可能需要更小的学习率, e.g., 1e-6 to 5e-6
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0

# Precision
FP16=False
BF16=True # 若GPU支持，推荐使用BF16

# Attention Implementation
ATTN_IMPLEMENTATION="flash_attention_2"

# Saving and Logging
SAVE_MERGED=True
SAVE_STRATEGY="steps"
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=1
LOGGING_STEPS=10
REPORT_TO="wandb"

# DPO Specific Parameters
BETA=0.1
LOSS_TYPE="sigmoid"
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
REFERENCE_FREE=False
LABEL_SMOOTHING=0.0
PRECOMPUTE_REF_LOG_PROBS=True

# Other Configurations
RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"
DATALOADER_NUM_WORKERS=8
DATALOADER_PIN_MEMORY=True
# AUTO_FIND_BATCH_SIZE=False # TrainingArguments参数, 如果需要则传递

# --- Output Configuration ---
MODEL_BASENAME=$(basename "$MODEL_PATH")
SAVE_PATH_BASE="dpo_models_v4/GCR-lora-dpo_v4_${DATASET_NAME}_${CANDIDATE_STRATEGY}_${POSITIVE_SOURCE_FIELD}"
RUN_SPECIFIC_NAME="${MODEL_BASENAME}_epoch${NUM_TRAIN_EPOCHS}_${LOSS_TYPE}_beta${BETA}_lora_r${LORA_R}"
SAVE_PATH="${SAVE_PATH_BASE}/${RUN_SPECIFIC_NAME}"
WANDB_RUN_NAME="${RUN_SPECIFIC_NAME}"

# --- Script Sanity Checks ---
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
  echo "ERROR: Python DPO script not found at '$PYTHON_SCRIPT_PATH'"
  exit 1
fi
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
  echo "ERROR: Accelerate config file not found at '$ACCELERATE_CONFIG_FILE'"
  exit 1
fi
if [ ! -d "$DATA_PATH_LIST" ]; then # 明确检查是否为目录
  echo "ERROR: DPO Dataset directory (DATA_PATH_LIST) not found at '$DATA_PATH_LIST'"
  exit 1
fi


# --- Launch Training ---
echo "========================================================"
echo "Starting DPO Fine-tuning with Accelerate..."
echo "Model: ${MODEL_PATH}"
echo "Dataset Path: ${DATA_PATH_LIST}"
echo "Output Directory: ${SAVE_PATH}"
echo "Wandb Run Name: ${WANDB_RUN_NAME}"
echo "Accelerate Config: ${ACCELERATE_CONFIG_FILE}"
echo "Python Script: ${PYTHON_SCRIPT_PATH}"
echo "Epochs: ${NUM_TRAIN_EPOCHS}, Per-Device Batch Size: ${PER_DEVICE_TRAIN_BATCH_SIZE}, Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective Batch Size (single node): $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "LoRA R: ${LORA_R}, Alpha: ${LORA_ALPHA}"
echo "Quantization: 4-bit (${LOAD_IN_4BIT}), 8-bit (${LOAD_IN_8BIT})"
echo "Precision: BF16 (${BF16}), FP16 (${FP16})"
echo "DPO Beta: ${BETA}, Loss Type: ${LOSS_TYPE}"
echo "========================================================"

CMD="accelerate launch --config_file \"${ACCELERATE_CONFIG_FILE}\" \"${PYTHON_SCRIPT_PATH}\""
CMD+=" --data_path_list \"${DATA_PATH_LIST}\"" # Python脚本的data_path_list参数应为单个目录路径
CMD+=" --model_name_or_path \"${MODEL_PATH}\""
CMD+=" --output_dir \"${SAVE_PATH}\""

# PEFT/LoRA
CMD+=" --use_peft ${USE_PEFT}"
if [ "${USE_PEFT}" = "True" ]; then
    CMD+=" --lora_r ${LORA_R}"
    CMD+=" --lora_alpha ${LORA_ALPHA}"
    CMD+=" --lora_dropout ${LORA_DROPOUT}"
    CMD+=" --target_modules \"${TARGET_MODULES}\""
    CMD+=" --save_merged ${SAVE_MERGED}"
fi

# Quantization
CMD+=" --load_in_4bit ${LOAD_IN_4BIT}"
CMD+=" --load_in_8bit ${LOAD_IN_8BIT}"
if [ "${LOAD_IN_4BIT}" = "True" ]; then
    CMD+=" --bnb_4bit_quant_type \"${BNB_4BIT_QUANT_TYPE_ARG}\""
    CMD+=" --bnb_4bit_compute_dtype \"${BNB_4BIT_COMPUTE_DTYPE_ARG}\""
    CMD+=" --bnb_4bit_use_double_quant ${BNB_4BIT_USE_DOUBLE_QUANT_ARG}"
fi

# TrainingArguments
CMD+=" --fp16 ${FP16}"
CMD+=" --bf16 ${BF16}"
CMD+=" --num_train_epochs ${NUM_TRAIN_EPOCHS}"
CMD+=" --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}"
CMD+=" --per_device_eval_batch_size 1" # DPO评估通常batch_size为1
CMD+=" --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"
CMD+=" --eval_strategy \"no\"" # 如果有评估集并希望评估，改为 "steps" 或 "epoch" 并设置 eval_steps
CMD+=" --save_strategy \"${SAVE_STRATEGY}\""
CMD+=" --save_steps \"${SAVE_STEPS}\""
CMD+=" --save_total_limit ${SAVE_TOTAL_LIMIT}"
CMD+=" --learning_rate ${LEARNING_RATE}"
CMD+=" --weight_decay ${WEIGHT_DECAY}"
CMD+=" --warmup_ratio ${WARMUP_RATIO}"
CMD+=" --lr_scheduler_type \"${LR_SCHEDULER_TYPE}\""
CMD+=" --logging_steps ${LOGGING_STEPS}"
CMD+=" --report_to \"${REPORT_TO}\""
CMD+=" --run_name \"${WANDB_RUN_NAME}\""
CMD+=" --gradient_checkpointing ${GRADIENT_CHECKPOINTING}"

# Other general args (Python script must define these in its HfArgumentParser)
if [ -n "${ATTN_IMPLEMENTATION:-}" ]; then
    CMD+=" --attn_implementation \"${ATTN_IMPLEMENTATION}\""
fi
# 如果Python脚本的ScriptArguments中定义了response_template，则传递
if [ -n "${RESPONSE_TEMPLATE:-}" ]; then
    CMD+=" --response_template \"${RESPONSE_TEMPLATE}\""
fi
CMD+=" --dataloader_num_workers ${DATALOADER_NUM_WORKERS}"
CMD+=" --dataloader_pin_memory ${DATALOADER_PIN_MEMORY}"
# CMD+=" --auto_find_batch_size False" # 如果需要且Python脚本支持

# DPO Specific Arguments (Python script's ScriptArguments must define these)
CMD+=" --beta ${BETA}"
CMD+=" --loss_type \"${LOSS_TYPE}\""
CMD+=" --max_length ${MAX_LENGTH}"
CMD+=" --max_prompt_length ${MAX_PROMPT_LENGTH}"
CMD+=" --reference_free ${REFERENCE_FREE}"
CMD+=" --label_smoothing ${LABEL_SMOOTHING}"
CMD+=" --precompute_ref_log_probs ${PRECOMPUTE_REF_LOG_PROBS}"
# --dpo_alpha, --eval_dataset_path, --generate_during_eval 等会使用Python脚本中的默认值，除非在这里显式传递

# Execute command
echo "INFO: Executing DPO command:"
echo "${CMD}"
eval "${CMD}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "DPO Fine-tuning completed successfully."
    echo "Model saved to: ${SAVE_PATH}"
    echo "--------------------------------------------------------"
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "DPO Fine-tuning FAILED with exit code $EXIT_CODE."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

exit $EXIT_CODE