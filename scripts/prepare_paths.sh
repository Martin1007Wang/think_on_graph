#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export BASE_PROJECT_DIR="/mnt/wangjingxiong/think_on_graph" 
export PYTHONPATH="${BASE_PROJECT_DIR}"
# --- 数据集和路径配置 ---
DATASET_INPUTS=("rmanluo/RoG-webqsp" "rmanluo/RoG-cwq")
SPLITS=("test" "test")
# 如果您的 Hugging Face 数据集需要特定的配置名 (例如 "wikitext-2-raw-v1" for wikitext)
# 如果不需要，可以留空 DATASET_CONFIG_NAME=""
# 注意：当前 Python 脚本假设对所有 --dataset_inputs 使用相同的 config_name (如果提供)
DATASET_CONFIG_NAME="" # 例如 "sub ديالكt" or ""
OUTPUT_PATH="data/paths" # 建议使用新的输出路径以避免与旧数据混淆

# --- Neo4j 配置 ---
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang" # 请确保密码安全

# --- 模型和嵌入配置 ---
MODEL_NAME="all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE=1024

# ---路径生成配置 ---
MAX_PATH_LENGTH=3
TOP_K_RELATIONS=3 # 请与Python脚本中的默认值或您的需求匹配

# --- 样本处理配置 ---
MAX_PAIRS=5
MAX_NEGATIVES_PER_PAIR=5
NUM_SAMPLES=-1 # -1 表示处理所有样本
NUM_THREADS=16

# --- 新增：恢复和临时文件配置 ---
RESUME_PROCESSING=true # 设置为 true 以启用恢复功能，false 则不启用
KEEP_TEMP_FILES=false # 设置为 true 以保留临时 .jsonl 文件，false 则删除

# --- Python 脚本路径 ---
PYTHON_SCRIPT_PATH="workflow/prepare_paths.py"

# 检查 Python 脚本是否存在
if [ ! -f "${PYTHON_SCRIPT_PATH}" ]; then
    echo "错误: Python 脚本未找到于 ${PYTHON_SCRIPT_PATH}"
    echo "请确保您在正确的项目根目录下运行此脚本。"
    exit 1
fi

# 构建 Python 命令参数
CMD_ARGS=(
    --dataset_inputs "${DATASET_INPUTS[@]}"
    --splits "${SPLITS[@]}"
    --output_path "${OUTPUT_PATH}"
    --neo4j_uri "${NEO4J_URI}"
    --neo4j_user "${NEO4J_USER}"
    --neo4j_password "${NEO4J_PASSWORD}"
    --model_name "${MODEL_NAME}"
    --embedding_encode_batch_size ${EMBEDDING_BATCH_SIZE}
    --max_path_length ${MAX_PATH_LENGTH}
    --top_k_relations ${TOP_K_RELATIONS}
    --max_pairs ${MAX_PAIRS}
    --max_negatives_per_pair ${MAX_NEGATIVES_PER_PAIR}
    --num_samples ${NUM_SAMPLES}
    --num_threads ${NUM_THREADS}
)

# 添加 dataset_config_name (如果已设置)
if [ -n "${DATASET_CONFIG_NAME}" ]; then
    CMD_ARGS+=(--dataset_config_name "${DATASET_CONFIG_NAME}")
fi

# 添加 --resume_processing 标志 (如果 RESUME_PROCESSING 为 true)
if [ "${RESUME_PROCESSING}" = true ]; then
    CMD_ARGS+=(--resume_processing)
fi

# 添加 --keep_temp_files 标志 (如果 KEEP_TEMP_FILES 为 true)
if [ "${KEEP_TEMP_FILES}" = true ]; then
    CMD_ARGS+=(--keep_temp_files)
fi

# 移除了 --force_recompute_embeddings，如果您的 Python 脚本仍需要它，请取消注释下一行
# CMD_ARGS+=(--force_recompute_embeddings)


echo "正在执行 Python 脚本: ${PYTHON_SCRIPT_PATH}"
echo "参数: ${CMD_ARGS[@]}"
echo "----------------------------------------------------"

# 执行 Python 脚本
python ${PYTHON_SCRIPT_PATH} "${CMD_ARGS[@]}"

echo "----------------------------------------------------"
echo "脚本执行完毕。"