#!/bin/bash
PID_TO_MONITOR=2708829 # 您想要监控的进程ID

echo "开始监控进程 PID: ${PID_TO_MONITOR}..."

# 循环直到找不到该进程
# 'kill -0 PID' 会检查进程是否存在，如果存在则返回0，否则返回非0
# 我们将标准输出和标准错误重定向到 /dev/null 以避免不必要的输出
while kill -0 ${PID_TO_MONITOR} >/dev/null 2>&1; do
  echo "进程 ${PID_TO_MONITOR} 仍在运行。等待10秒后再次检查..."
  sleep 10 # 每10秒检查一次，您可以根据需要调整这个间隔
done

echo "进程 ${PID_TO_MONITOR} 已结束。现在开始运行主脚本..."
echo "-----------------------------------------------------"

# 环境设置
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
# DeepSeek API密钥设置，请替换为您的密钥
export DEEPSEEK_API_KEY="sk-9ebd54eae9f749ee825ed1c331270eb6"
# export DEEPSEEK_API_KEY="e058d4fa-4202-4669-b543-b7350289731c"

# 数据参数
DATA_PATH=rmanluo
DATA_NAME="RoG-webqsp"
SPLIT="test"

# 输出路径
PREDICT_PATH="results/IterativeReasoning_v13"

# 探索模型参数 - 探索阶段仍使用原模型
EXPLORE_MODEL_NAME="lora_naive_instruction_dataset_webqsp_pn_kg_supplement_shortest_paths"
EXPLORE_MODEL_PATH="/mnt/wangjingxiong/think_on_graph/sft_models_v4/lora_naive_instruction_dataset_webqsp_pn_kg_supplement_shortest_paths/0e9e39f249a16976918f6564b8830bc894c89659_epoch5_lora_r8/merged_model"

# EXPLORE_MODEL_NAME="GCR-lora-sft_v3_with_label-Llama-3.1-8B-Instruct"
# EXPLORE_MODEL_PATH="/mnt/wangjingxiong/think_on_graph/sft_models_v3/GCR-lora-sft_v3_with_label-Llama-3.1-8B-Instruct/merged_model"

# EXPLORE_MODEL_NAME="Llama-3.1-8B-Instruct"
# EXPLORE_MODEL_PATH="/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# 预测模型参数 - 使用DeepSeek Chat API
PREDICT_MODEL_NAME="deepseek-chat"
PREDICT_MODEL_PATH=None

# 生成参数
GENERATION_K=4
GENERATION_MODE="beam"

# 知识图谱参数
MAX_ROUNDS=2
MAX_SELECTION_COUNT=5
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang"
EMBEDDING_MODEL="msmarco-distilbert-base-tas-b"

# 探索历史大小限制
MAX_EXPLORATION_HISTORY_SIZE=200000

# 模型加载和生成参数
MAX_NEW_TOKENS=4096  # 降低token限制，适合DeepSeek模型
DTYPE="bf16"
ATTN_IMPLEMENTATION="flash_attention_2"
QUANT="8bit"
USE_FLASH_ATTN="true"

# 并行处理参数
MAX_WORKERS=8

# 重试次数
RETRY=5

echo "开始使用DeepSeek Chat API进行知识图谱推理..."
echo "API密钥: ${DEEPSEEK_API_KEY:0:4}...${DEEPSEEK_API_KEY: -4}"

for DATA in ${DATA_NAME}; do
  echo "处理数据集: ${DATA}"
  python workflow/predict_paths_and_answers_2.py \
    --data_path ${DATA_PATH} \
    --data_name ${DATA} \
    --split ${SPLIT} \
    --predict_path ${PREDICT_PATH} \
    --explore_model_name ${EXPLORE_MODEL_NAME} \
    --explore_model_path ${EXPLORE_MODEL_PATH} \
    --predict_model_name ${PREDICT_MODEL_NAME} \
    --predict_model_path ${PREDICT_MODEL_PATH} \
    --max_rounds ${MAX_ROUNDS} \
    --max_selection_count ${MAX_SELECTION_COUNT} \
    --neo4j_uri ${NEO4J_URI} \
    --neo4j_user ${NEO4J_USER} \
    --neo4j_password ${NEO4J_PASSWORD} \
    --embedding_model ${EMBEDDING_MODEL} \
    --generation_mode ${GENERATION_MODE} \
    --generation_k ${GENERATION_K} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --dtype ${DTYPE} \
    --attn_implementation ${ATTN_IMPLEMENTATION} \
    --max_workers ${MAX_WORKERS} \
    --retry ${RETRY} \
    --deepseek_api_key "${DEEPSEEK_API_KEY}" \
    --debug
done

echo "所有任务完成！结果保存在 ${PREDICT_PATH}"
