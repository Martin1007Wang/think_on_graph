#!/bin/bash

# Shell 执行选项：如果任何命令失败，则立即退出；如果使用了未设置的变量，则报错
set -e
set -u

# --- 必须由用户设置的环境变量或在此处直接赋值 ---
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

# --- 并行化和设备配置 ---
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1

# --- 数据路径配置 ---
DATASET_NAME="webqsp"
CANDIDATE_STRATEGY="pn_kg_supplement"
POSITIVE_SOURCE_FIELD="shortest_paths"
DATA_PATH_LIST="${BASE_PROJECT_DIR}/data/instruction_dataset/${DATASET_NAME}_train_sft_instruct_cand_${CANDIDATE_STRATEGY}_pos_${POSITIVE_SOURCE_FIELD}"

SAVE_PROCESSED_TRAIN_DATA_PATH="cache/relation_sft_dataset_processed_by_script_${DATASET_NAME}_${CANDIDATE_STRATEGY}_${POSITIVE_SOURCE_FIELD}"
DATASET_CACHE_DIR="cache/hf_datasets_map_cache"
FORCE_DATA_PROCESSING=False

PYTHON_SCRIPT_PATH="workflow/finetune_kg_specialized_llm_sft.py"
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_zero2.yaml"

# --- 训练参数 ---
NUM_TRAIN_EPOCHS=5
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=True
LEARNING_RATE=5e-5
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0

# --- LoRA 配置 (在模型与输出配置之前定义，因为LORA_R用于RUN_SPECIFIC_NAME) ---
USE_PEFT=True
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SAVE_MERGED=True

# --- 模型与输出配置 ---
MODEL_BASENAME=$(basename "$MODEL_PATH")
SAVE_PATH_BASE="sft_models_v4/GCR-lora-sft_v4_${DATASET_NAME}_${CANDIDATE_STRATEGY}_${POSITIVE_SOURCE_FIELD}"
RUN_SPECIFIC_NAME="${MODEL_BASENAME}_epoch${NUM_TRAIN_EPOCHS}_lora_r${LORA_R}" # SFT 特定命名
SAVE_PATH="${SAVE_PATH_BASE}/${RUN_SPECIFIC_NAME}"
WANDB_RUN_NAME="${RUN_SPECIFIC_NAME}"

# --- 量化配置 ---
LOAD_IN_4BIT=False
LOAD_IN_8BIT=False

BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT=True

# --- 其他配置 ---
ATTN_IMPLEMENTATION="flash_attention_2"
DATALOADER_NUM_WORKERS=8
DATALOADER_PIN_MEMORY=True

# Precision
FP16=False
BF16=True

# Saving and Logging
SAVE_STRATEGY="steps"
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=1
LOGGING_STEPS=50
REPORT_TO="wandb"

# --- 响应模板配置 (关键修改) ---
LLAMA3_SFT_TEMPLATE_STRING="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion}<|eot_id|>"
RESPONSE_TEMPLATE_TO_PASS="${LLAMA3_SFT_TEMPLATE_STRING}"

# --- Script Sanity Checks ---
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
  echo "ERROR: Python SFT script not found at '$PYTHON_SCRIPT_PATH'"
  exit 1
fi
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
  echo "ERROR: Accelerate config file not found at '$ACCELERATE_CONFIG_FILE'"
  exit 1
fi
if [ ! -e "$DATA_PATH_LIST" ]; then
  echo "ERROR: SFT Dataset (DATA_PATH_LIST) not found at '$DATA_PATH_LIST'"
  exit 1
fi

# 构建 accelerate launch 命令
CMD="accelerate launch --config_file \"${ACCELERATE_CONFIG_FILE}\" \"${PYTHON_SCRIPT_PATH}\""

# 数据相关参数
CMD+=" --data_path_list \"${DATA_PATH_LIST}\""
CMD+=" --save_processed_train_dataset_path \"${SAVE_PROCESSED_TRAIN_DATA_PATH}\""
CMD+=" --dataset_cache_dir \"${DATASET_CACHE_DIR}\""
CMD+=" --force_data_processing ${FORCE_DATA_PROCESSING}"

# 模型与输出参数
CMD+=" --model_name_or_path \"${MODEL_PATH}\""
CMD+=" --output_dir \"${SAVE_PATH}\""

# PEFT/LoRA 参数
CMD+=" --use_peft ${USE_PEFT}"
if [ "${USE_PEFT}" = "True" ]; then
    CMD+=" --lora_r ${LORA_R}"
    CMD+=" --lora_alpha ${LORA_ALPHA}"
    CMD+=" --lora_dropout ${LORA_DROPOUT}"
    CMD+=" --target_modules \"${TARGET_MODULES}\""
    CMD+=" --save_merged ${SAVE_MERGED}"
fi

# 量化参数
CMD+=" --load_in_4bit ${LOAD_IN_4BIT}"
CMD+=" --load_in_8bit ${LOAD_IN_8BIT}"
if [ "${LOAD_IN_4BIT}" = "True" ]; then
    CMD+=" --bnb_4bit_quant_type \"${BNB_4BIT_QUANT_TYPE}\""
    CMD+=" --bnb_4bit_compute_dtype \"${BNB_4BIT_COMPUTE_DTYPE}\""
    CMD+=" --bnb_4bit_use_double_quant ${BNB_4BIT_USE_DOUBLE_QUANT}"
fi

# TrainingArguments 参数
CMD+=" --fp16 ${FP16}"
CMD+=" --bf16 ${BF16}"
CMD+=" --num_train_epochs ${NUM_TRAIN_EPOCHS}"
CMD+=" --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}"
CMD+=" --per_device_eval_batch_size 1"
CMD+=" --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"
CMD+=" --eval_strategy \"no\""
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

# 其他一般参数
if [ -n "${ATTN_IMPLEMENTATION:-}" ]; then
    CMD+=" --attn_implementation \"${ATTN_IMPLEMENTATION}\""
fi

# 传递 response_template (关键修改)
if [ -n "${RESPONSE_TEMPLATE_TO_PASS:-}" ]; then
    CMD+=" --response_template \"${RESPONSE_TEMPLATE_TO_PASS}\""
fi

CMD+=" --dataloader_num_workers ${DATALOADER_NUM_WORKERS}"
CMD+=" --dataloader_pin_memory ${DATALOADER_PIN_MEMORY}"

# 执行命令
echo "INFO: Executing SFT command:"
echo "${CMD}"
eval "${CMD}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "SFT Fine-tuning completed successfully."
    echo "Model saved to: ${SAVE_PATH}"
    echo "--------------------------------------------------------"
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "SFT Fine-tuning FAILED with exit code $EXIT_CODE."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

exit $EXIT_CODE