#!/bin/bash

# 设置工作目录
WORKDIR="/mnt/wangjingxiong/think_on_graph"
cd $WORKDIR || { echo "工作目录不存在!"; exit 1; }

# 创建输出目录
mkdir -p "$WORKDIR/results"

# 数据路径
PROCESSED_DATA="/mnt/wangjingxiong/think_on_graph/data/processed/rmanluo_RoG-webqsp_train/path_data.json"
OUTPUT_FILE="$WORKDIR/results/long_semantic_paths_$(date +"%Y%m%d").txt"

echo "===== 开始查找显著长于最短路径的语义路径 - $(date) ====="

# 运行find.py脚本查找特定示例
python find.py > "$OUTPUT_FILE"

# 检查结果
if [ $? -eq 0 ]; then
    COUNT=$(grep -c "ID:" "$OUTPUT_FILE" || echo 0)
    echo "找到 $COUNT 个符合条件的样本"
    echo "结果已保存到: $OUTPUT_FILE"
    
    # 显示部分结果预览
    if [ "$COUNT" -gt 0 ]; then
        echo -e "\n===== 结果预览 ====="
        head -n 20 "$OUTPUT_FILE"
        echo -e "\n...(更多结果请查看输出文件)..."
    fi
else
    echo "脚本执行失败"
    exit 1
fi

echo -e "\n===== 查找完成 - $(date) ====="
exit 0 