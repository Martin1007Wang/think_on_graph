#!/bin/bash

# 设置环境变量
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="Martin1007Wang"  # 生产环境中应考虑使用环境变量或配置文件

# 设置工作目录
WORKDIR="/mnt/wangjingxiong/think_on_graph"
cd $WORKDIR || { echo "工作目录不存在!"; exit 1; }

# 创建日志目录
LOGDIR="$WORKDIR/logs"
mkdir -p $LOGDIR

# 当前时间戳(用于日志文件名)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 设置常用参数
DATASET_NAME="webqsp"  
SPLIT="train"
INPUT_DATASET="/mnt/wangjingxiong/think_on_graph/data/raw/${DATASET_NAME}_${SPLIT}.json"
OUTPUT_BASE="/mnt/wangjingxiong/think_on_graph/data/processed"
PATH_OUTPUT_NAME="rmanluo_RoG-${DATASET_NAME}_${SPLIT}"
PREFERENCE_OUTPUT_BASE="/mnt/wangjingxiong/think_on_graph/data/finetune"
PREFERENCE_OUTPUT_NAME="${DATASET_NAME}_${SPLIT}_preference_with_label"

# 日志文件
PATH_LOG="${LOGDIR}/prepare_paths_${TIMESTAMP}.log"
PREFERENCE_LOG="${LOGDIR}/create_preference_${TIMESTAMP}.log"

echo "===== 开始数据处理流程 - $(date) ====="

# 第一步: 生成路径数据
echo "1. 运行 prepare_paths.py 生成路径数据..."
python workflow/prepare_paths.py \
    --data_path "$INPUT_DATASET" \
    --dataset_name "$DATASET_NAME" \
    --split "$SPLIT" \
    --output_path "$OUTPUT_BASE" \
    --output_name "$PATH_OUTPUT_NAME" \
    --neo4j_uri "$NEO4J_URI" \
    --neo4j_user "$NEO4J_USER" \
    --neo4j_password "$NEO4J_PASSWORD" \
    --max_path_length 3 \
    --top_k_relations 5 \
    --max_pairs 5 \
    --max_negatives_per_pair 5 \
    --num_threads 16 \
    --num_samples -1 \
    2>&1 | tee "$PATH_LOG"

# 检查上一步是否成功
if [ $? -ne 0 ]; then
    echo "路径数据生成失败，请检查日志: $PATH_LOG"
    exit 1
fi

# 第二步: 创建偏好数据集
echo "2. 运行 create_preference_dataset_with_label.py 创建偏好数据集..."
python workflow/create_preference_dataset_with_label.py \
    --input_path "$OUTPUT_BASE/$PATH_OUTPUT_NAME/path_data.json" \
    --output_path "$PREFERENCE_OUTPUT_BASE" \
    --output_name "$PREFERENCE_OUTPUT_NAME" \
    --max_selection_count 5 \
    --neo4j_uri "$NEO4J_URI" \
    --neo4j_user "$NEO4J_USER" \
    --neo4j_password "$NEO4J_PASSWORD" \
    --num_samples -1 \
    2>&1 | tee "$PREFERENCE_LOG"

# 检查结果
if [ $? -eq 0 ]; then
    echo "===== 数据处理流程完成 - $(date) ====="
    echo "路径数据已生成: $OUTPUT_BASE/$PATH_OUTPUT_NAME/path_data.json"
    echo "偏好数据集已生成: $PREFERENCE_OUTPUT_BASE/$PREFERENCE_OUTPUT_NAME/"
else
    echo "偏好数据集创建失败，请检查日志: $PREFERENCE_LOG"
    exit 1
fi

# 打印摘要信息
echo "===== 处理摘要 ====="
echo "处理时间: $(date)"
echo "路径日志: $PATH_LOG"
echo "偏好日志: $PREFERENCE_LOG"

exit 0 