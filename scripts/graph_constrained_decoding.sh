#!/bin/bash

# 首先监控进程1578608是否结束
echo "正在等待进程 1578608 结束..."
while ps -p 1578608 &>/dev/null; do
    echo "进程 1578608 仍在运行，等待 30 秒后再次检查..."
    sleep 30
done
echo "进程 1578608 已结束，开始执行后续任务..."

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
DATA_PATH=rmanluo
DATA_NAME="RoG-webqsp"
SPLIT="test"
MODEL_NAME="GCR-lora-sft_with_label-Llama-3.1-8B-Instruct"
MODEL_PATH="/mnt/wangjingxiong/think_on_graph/sft_models_v2/GCR-lora-0e9e39f249a16976918f6564b8830bc894c89659/merged_model"
# MODEL_NAME="Llama-3.1-8B-Instruct"
# MODEL_PATH="/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
#MODEL_NAME="GCR-lora-Llama-3.1-8B-Instruct"
#MODEL_PATH="/mnt/wangjingxiong/think_on_graph/dpo_models_v2/GCR-lora-Llama-3.1-8B"
GENERATION_K=5
GENERATION_MODE="group-beam"

for DATA in ${DATA_NAME}; do
  python -m memory_profiler workflow/predict_paths_and_answers_2.py --data_path ${DATA_PATH} --data_name ${DATA_NAME} --split ${SPLIT} --model_name ${MODEL_NAME} --model_path ${MODEL_PATH} --generation_k ${GENERATION_K} --generation_mode ${GENERATION_MODE}
done
