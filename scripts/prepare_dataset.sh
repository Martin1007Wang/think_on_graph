#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
DATA_PATH="rmanluo/RoG-cwq"
DATASET_NAME="RoG-cwq" # 通常与DATA_PATH的核心部分相同，用于缓存等
SPLIT="test"
OUTPUT_PATH="data/processed" # 基础输出路径

CLEANED_DATA_PATH=$(echo "${DATA_PATH}" | tr '/' '_') 
OUTPUT_NAME="${CLEANED_DATA_PATH}_${SPLIT}"

NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang"

MAX_PATH_LENGTH=3
TOP_K_RELATIONS=5
MODEL_NAME="all-mpnet-base-v2"
MAX_PAIRS=5
MAX_NEGATIVES_PER_PAIR=5
NUM_SAMPLES=-1
PYTHON_SCRIPT_PATH="workflow/prepare_paths.py"

# 检查 Python 脚本是否存在
if [ ! -f "${PYTHON_SCRIPT_PATH}" ]; then
    echo "错误: Python 脚本未找到于 ${PYTHON_SCRIPT_PATH}"
    echo "请确保您在正确的项目根目录下运行此脚本。"
    exit 1
fi

echo "输出目录名 (output_name): ${OUTPUT_NAME}"
echo "完整输出路径将是: ${OUTPUT_PATH}/${OUTPUT_NAME}"

python ${PYTHON_SCRIPT_PATH} \
    --data_path "${DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --output_path "${OUTPUT_PATH}" \
    --output_name "${OUTPUT_NAME}" \
    --neo4j_uri "${NEO4J_URI}" \
    --neo4j_user "${NEO4J_USER}" \
    --neo4j_password "${NEO4J_PASSWORD}" \
    --max_path_length ${MAX_PATH_LENGTH} \
    --top_k_relations ${TOP_K_RELATIONS} \
    --max_pairs ${MAX_PAIRS} \
    --max_negatives_per_pair ${MAX_NEGATIVES_PER_PAIR} \
    --model_name "${MODEL_NAME}" \
    --num_samples ${NUM_SAMPLES} # 如果您添加了NUM_SAMPLES变量

echo "脚本执行完毕。"