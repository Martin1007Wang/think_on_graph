import os
import json
import re
import string
import logging
import argparse
from collections import defaultdict
from typing import List, Set, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def normalize(s: str) -> str:
    """文本标准化函数"""
    s = str(s).lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def calculate_all_metrics(predicted_list: List[Any], gold_list: List[Any]) -> Dict[str, float]:
    """
    一个函数计算所有指标，包括两种定义的 Accuracy。
    """
    predicted_set = {normalize(p) for p in predicted_list if str(p).strip()}
    gold_set = {normalize(g) for g in gold_list if str(g).strip()}

    # --- 边缘情况处理 ---
    if not gold_set:
        is_perfect = 1.0 if not predicted_set else 0.0
        return {
            'f1': is_perfect, 'precision': is_perfect, 'recall': is_perfect,
            'hit_rate': is_perfect, 'accuracy_partial': is_perfect
        }
    if not predicted_set:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'hit_rate': 0.0, 'accuracy_partial': 0.0}

    # --- 核心计算 ---
    true_positives = len(predicted_set.intersection(gold_set))
    
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 定义1: Hit Rate (命中即正确)
    hit_rate = 1.0 if true_positives > 0 else 0.0
    
    # 定义2: Accuracy (Partial Credit - 部分得分)
    # 注意：这里我们用集合交集大小来代替原始的字符串包含匹配，更为精确
    accuracy_partial = true_positives / len(gold_set) # 这其实等同于 Recall，在集合评估中是常见的

    return {
        'f1': f1, 'precision': precision, 'recall': recall,
        'hit_rate': hit_rate,
        'accuracy_partial': accuracy_partial
    }

def run_evaluation(predict_file_path: str):
    """主评估流程"""
    output_dir = os.path.dirname(predict_file_path)
    prefix_name = os.path.splitext(os.path.basename(predict_file_path))[0]
    detailed_eval_file = os.path.join(output_dir, f"{prefix_name}_detailed_eval.jsonl")
    summary_eval_file = os.path.join(output_dir, f"{prefix_name}_summary_eval.txt")

    total_metrics = defaultdict(float)
    valid_samples_count = 0

    logging.info(f"开始评估: {predict_file_path}")
    
    with open(predict_file_path, "r", encoding="utf-8") as f_in, \
         open(detailed_eval_file, "w", encoding="utf-8") as f_out_detailed:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line)
                predicted_entities = data.get("answer_entities", [])
                ground_truth = data.get("ground_truth", [])
                
                current_metrics = calculate_all_metrics(predicted_entities, ground_truth)
                
                for key, value in current_metrics.items():
                    total_metrics[key] += value
                valid_samples_count += 1
                
                output_line_data = {"id": data.get("id"), "metrics": current_metrics, **data}
                f_out_detailed.write(json.dumps(output_line_data) + "\n")
            except json.JSONDecodeError:
                logging.warning(f"跳过无效JSON行 (行号 {line_num})")
                continue

    if valid_samples_count == 0:
        logging.warning("无有效样本进行评估。")
        return

    avg_metrics = {key: val / valid_samples_count for key, val in total_metrics.items()}
    
    logging.info("--- 评估结果汇总 ---")
    summary_content = f"Evaluated File: {predict_file_path}\nValid Samples: {valid_samples_count}\n\n--- Average Metrics ---\n"
    for key, value in avg_metrics.items():
        logging.info(f"{key.replace('_', ' ').capitalize():<20}: {value:.4f}")
        summary_content += f"{key.replace('_', ' ').capitalize():<20}: {value:.4f}\n"
    
    with open(summary_eval_file, "w", encoding="utf-8") as f_summary:
        f_summary.write(summary_content)
    logging.info(f"汇总报告已保存至: {summary_eval_file}")


def main():
    parser = argparse.ArgumentParser(description="最终版评估脚本，清晰区分不同指标定义。")
    parser.add_argument("--predict_file_path", type=str, default="/mnt/wangjingxiong/think_on_graph/results/all_result/RoG-cwq/deepseek-chat/deepseek-chat/iterative-rounds2-topk3/predictions.jsonl", help="JSONL 格式的预测文件路径。")
    args = parser.parse_args()
    run_evaluation(args.predict_file_path)

if __name__ == "__main__":
    main()