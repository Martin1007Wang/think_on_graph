export TOKENIZERS_PARALLELISM=false
DATASET_LIST="data/processed/rmanluo/RoG-webqsp_train"

# Lora
BATCH_SIZE=8
USE_PEFT=True
EPOCH=10
GRADIENT_CHECKPOINTING=False
GRADIENT_ACCUMULATION_STEPS=16
auto_find_batch_size=True

# Full
# BATCH_SIZE=4
# USE_PEFT=False
# EPOCH=3
# GRADIENT_CHECKPOINTING=True
# GRADIENT_ACCUMULATION_STEPS=16
# auto_find_batch_size=False

# CONFIG="accelerate_configs/multi_gpu.yaml"
CONFIG="accelerate_configs/deepspeed_zero3.yaml"

# Model Configurations

# MODEL_PATH=Qwen/Qwen2-0.5B-Instruct
# MODEL_PATH=Qwen/Qwen2-1.5B-Instruct
# MODEL_PATH=Qwen/Qwen2-7B-Instruct
# RESPONSE_TEMPLATE="<|im_start|>assistant"

# MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
# RESPONSE_TEMPLATE="[/INST]"

MODEL_PATH=/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"

ATTN_IMP=flash_attention_2

SAVE_PATH=cpo_models_with_response_template/GCR-$(basename "$MODEL_PATH")
SAVE_NAME=$(basename "$SAVE_PATH")

accelerate launch --config_file ${CONFIG} workflow/finetune_kg_specialized_llm_cpo.py \
    --data_path_list ${DATASET_LIST}  \
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