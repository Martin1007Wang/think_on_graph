#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List, Any, Set, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件内容"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.error(f"Error parsing line in {file_path}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} items to {file_path}")

def filter_perfect_hits(eval_path: str, pred_path: str, 
                      output_eval_path: Optional[str] = None,
                      output_pred_path: Optional[str] = None) -> None:
    """
    筛选ans_hit=1的案例，并保存到新文件。
    
    Args:
        eval_path: 评估结果文件路径 (detailed_eval_result.jsonl)
        pred_path: 预测文件路径 (predictions.jsonl)
        output_eval_path: 输出评估结果文件路径，默认为原路径加上_perfect_hits后缀
        output_pred_path: 输出预测文件路径，默认为原路径加上_perfect_hits后缀
    """
    # 设置默认输出路径
    if not output_eval_path:
        output_eval_path = os.path.join(
            os.path.dirname(eval_path),
            os.path.basename(eval_path).replace('.jsonl', '_perfect_hits.jsonl')
        )
    
    if not output_pred_path:
        output_pred_path = os.path.join(
            os.path.dirname(pred_path),
            os.path.basename(pred_path).replace('.jsonl', '_perfect_hits.jsonl')
        )
    
    # 加载数据
    logger.info(f"Loading evaluation data from {eval_path}")
    eval_data = load_jsonl(eval_path)
    logger.info(f"Loading prediction data from {pred_path}")
    pred_data = load_jsonl(pred_path)
    
    # 转换为ID索引的字典
    pred_dict = {item["id"]: item for item in pred_data}
    
    # 筛选ans_hit=1的案例
    perfect_ids = set()
    filtered_eval_data = []
    
    logger.info(f"Filtering cases with ans_hit=1")
    for item in eval_data:
        if item.get('ans_hit', 0) == 1:
            perfect_ids.add(item["id"])
            filtered_eval_data.append(item)
    
    # 根据筛选出的ID过滤预测数据
    filtered_pred_data = [pred_dict[id] for id in perfect_ids if id in pred_dict]
    
    # 保存筛选后的数据
    logger.info(f"Found {len(filtered_eval_data)} cases with perfect hits out of {len(eval_data)} total cases")
    logger.info(f"Saving filtered evaluation data to {output_eval_path}")
    save_jsonl(filtered_eval_data, output_eval_path)
    
    logger.info(f"Saving filtered prediction data to {output_pred_path}")
    save_jsonl(filtered_pred_data, output_pred_path)
    
    # 输出统计信息
    perfect_percentage = (len(filtered_eval_data) / len(eval_data)) * 100 if eval_data else 0
    logger.info(f"Perfect hits percentage: {perfect_percentage:.2f}%")
    logger.info(f"Perfect hits count: {len(filtered_eval_data)} / {len(eval_data)}")

def main():
    parser = argparse.ArgumentParser(description="Filter cases with perfect hit (ans_hit=1)")
    parser.add_argument("--eval_path", default="/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v6/RoG-webqsp/GCR-lora-sft_with_label-Llama-3.1-8B-Instruct/test/iterative-rounds3-topk5/detailed_eval_result.jsonl", help="Path to detailed_eval_result.jsonl")
    parser.add_argument("--pred_path", default="/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v6/RoG-webqsp/GCR-lora-sft_with_label-Llama-3.1-8B-Instruct/test/iterative-rounds3-topk5/predictions.jsonl", help="Path to predictions.jsonl")
    parser.add_argument("--output_eval_path", default="/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v6/RoG-webqsp/GCR-lora-sft_with_label-Llama-3.1-8B-Instruct/test/iterative-rounds3-topk5/detailed_eval_result.jsonl", help="Output path for filtered evaluation data")
    parser.add_argument("--output_pred_path", default="/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v6/RoG-webqsp/GCR-lora-sft_with_label-Llama-3.1-8B-Instruct/test/iterative-rounds3-topk5/predictions.jsonl", help="Output path for filtered prediction data")
    
    args = parser.parse_args()
    
    filter_perfect_hits(
        args.eval_path,
        args.pred_path,
        args.output_eval_path,
        args.output_pred_path
    )

if __name__ == "__main__":
    main()