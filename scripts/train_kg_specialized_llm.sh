export HF_ENDPOINT=https://hf-mirror.com

DATASET_LIST="data/processed/rmanluo/RoG-webqsp_train"
MODEL_NAME="msmarco-distilbert-base-tas-b"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang"

# Lora with optimized resources
# BATCH_SIZE=16
# USE_PEFT=True
# EPOCH=10
# GRADIENT_CHECKPOINTING=True
# GRADIENT_ACCUMULATION_STEPS=4
# auto_find_batch_size=True
# CONFIG="accelerate_configs/multi_gpu.yaml"

# Full
BATCH_SIZE=4
USE_PEFT=False
EPOCH=3
GRADIENT_CHECKPOINTING=True
GRADIENT_ACCUMULATION_STEPS=16
auto_find_batch_size=False
CONFIG="accelerate_configs/deepspeed_zero3.yaml"

# Model Configurations

MODEL_PATH=Qwen/Qwen2-0.5B-Instruct
ATTN_IMP=flash_attention_2
RESPONSE_TEMPLATE="<|im_start|>assistant"
CONFIG="accelerate_configs/multi_gpu.yaml"

# MODEL_PATH=Qwen/Qwen2-1.5B-Instruct
# ATTN_IMP=flash_attention_2
# RESPONSE_TEMPLATE="<|im_start|>assistant"
# CONFIG="accelerate_configs/multi_gpu.yaml"

# MODEL_PATH=Qwen/Qwen2-7B-Instruct
# ATTN_IMP=flash_attention_2
# RESPONSE_TEMPLATE="<|im_start|>assistant"
# CONFIG="accelerate_configs/deepspeed_zero3.yaml"

# MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
# ATTN_IMP=flash_attention_2
# RESPONSE_TEMPLATE="[/INST]"
# CONFIG="accelerate_configs/deepspeed_zero3.yaml"

# MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
# ATTN_IMP=flash_attention_2
# RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"
# CONFIG="accelerate_configs/deepspeed_zero3.yaml"

# MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# ATTN_IMP=flash_attention_2
# RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"
# CONFIG="accelerate_configs/deepspeed_zero3.yaml"


SAVE_PATH=save_models/GCR-$(basename "$MODEL_PATH")
SAVE_NAME=$(basename "$SAVE_PATH")

accelerate launch --config_file ${CONFIG} workflow/finetune_kg_specialized_llm.py \
    --data_path_list ${DATASET_LIST}  \
    --encode_model_name ${MODEL_NAME} \
    --neo4j_uri ${NEO4J_URI} \
    --neo4j_user ${NEO4J_USER} \
    --neo4j_password ${NEO4J_PASSWORD} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --use_peft ${USE_PEFT} \
    --bf16 True \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --auto_find_batch_size ${auto_find_batch_size} \
    --neftune_noise_alpha 5 \
    --attn_implementation ${ATTN_IMP} \
    --response_template "${RESPONSE_TEMPLATE}" \
    --run_name ${SAVE_NAME}

