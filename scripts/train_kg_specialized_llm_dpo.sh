export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
DATASET_LIST="data/processed/rmanluo/RoG-webqsp_train"

# Lora 配置 - 优化速度
BATCH_SIZE=16
USE_PEFT=True
EPOCH=10
GRADIENT_CHECKPOINTING=True
GRADIENT_ACCUMULATION_STEPS=16
auto_find_batch_size=False

# 量化配置 - 简化为正确的8位量化设置
LOAD_IN_4BIT=False
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT=True

LOAD_IN_8BIT=True  # 8位量化通常比4位量化更快

# LoRA 参数 - 速度优化
LORA_R=8  # 减小秩以加速
LORA_ALPHA=16
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
SAVE_MERGED=True

CONFIG="accelerate_configs/deepspeed_zero2.yaml"

MODEL_PATH=/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"

ATTN_IMP=flash_attention_2

SAVE_PATH=dpo_models/GCR-lora-$(basename "$MODEL_PATH")
SAVE_NAME=$(basename "$SAVE_PATH")

# DPO特定参数
BETA=0.1
LOSS_TYPE=sigmoid
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256
REFERENCE_FREE=False
LABEL_SMOOTHING=0.0

accelerate launch --config_file ${CONFIG} workflow/finetune_kg_specialized_llm_dpo.py \
    --data_path_list ${DATASET_LIST}  \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --use_peft ${USE_PEFT} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --target_modules "${TARGET_MODULES}" \
    --load_in_4bit ${LOAD_IN_4BIT} \
    --load_in_8bit ${LOAD_IN_8BIT} \
    --save_merged ${SAVE_MERGED} \
    --fp16 True \
    --bf16 False \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --report_to "wandb" \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --auto_find_batch_size ${auto_find_batch_size} \
    --attn_implementation ${ATTN_IMP} \
    --response_template "${RESPONSE_TEMPLATE}" \
    --beta ${BETA} \
    --loss_type ${LOSS_TYPE} \
    --max_length ${MAX_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --reference_free ${REFERENCE_FREE} \
    --label_smoothing ${LABEL_SMOOTHING} \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --run_name "${SAVE_NAME}"