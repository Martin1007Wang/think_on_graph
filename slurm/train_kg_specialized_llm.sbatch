#!/bin/bash
#SBATCH --partition=a100x4
#SBATCH --account=wangjian
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=kg_llm_training
#SBATCH --output=logs/kg_llm_training_%j.log
#SBATCH --error=logs/kg_llm_training_%j.err

DATASET_LIST="data/processed/rmanluo/RoG-webqsp_train"

# Full
BATCH_SIZE=4
USE_PEFT=False
EPOCH=3
GRADIENT_CHECKPOINTING=True
GRADIENT_ACCUMULATION_STEPS=16
auto_find_batch_size=False

CONFIG="accelerate_configs/deepspeed_zero3.yaml"

MODEL_PATH=/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"

ATTN_IMP=flash_attention_2

SAVE_PATH=cpo_models/GCR-$(basename "$MODEL_PATH")
SAVE_NAME=$(basename "$SAVE_PATH")

# 创建日志目录
mkdir -p logs

# 输出作业信息
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Training model: ${MODEL_PATH}"

export WANDB_MODE=offline
export WANDB_API_KEY=485437de8d2d866cc352b3046d229da99f2a7386
__conda_setup="$('/home/wangjingxiong/project/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/wangjingxiong/project/anaconda3/etc/profile.d/conda.sh" ]; 
then
        . "/home/wangjingxiong/project/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/wangjingxiong/project/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate GCR

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

echo "Job completed at $(date)"