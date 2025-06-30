import os
import json
import argparse
import logging
import shutil
from typing import Dict, Any, Set, List, Tuple

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_id(data: Dict[str, Any]) -> str or None:
    """从数据行中提取ID，兼容 'id' 和 'q_id' 两种常见键名。"""
    if 'id' in data:
        return data['id']
    if 'q_id' in data:
        return data['q_id']
    logging.warning(f"无法在数据行中找到 'id' 或 'q_id': {str(data)[:100]}...")
    return None

def is_hit(data: Dict[str, Any]) -> bool:
    """
    通用函数，用于检查给定数据行（字典格式）的“hit”指标是否为1。
    """
    metrics = data.get("metrics", {})
    if isinstance(metrics, dict):
        hit_rate = metrics.get("hit_rate")
        if hit_rate is not None and float(hit_rate) == 1.0:
            return True

    ans_hit = data.get("ans_hit")
    if ans_hit is not None and int(ans_hit) == 1:
        return True
        
    return False

def find_successful_ids(eval_filepath: str) -> Tuple[Set[str], List[str]]:
    """
    读取 eval 文件，返回一个包含所有 hit=1 样本ID的集合，以及这些成功的原始数据行。
    """
    successful_ids = set()
    successful_lines = []
    
    if not os.path.exists(eval_filepath):
        return successful_ids, successful_lines

    with open(eval_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if is_hit(data):
                    sample_id = get_id(data)
                    if sample_id:
                        successful_ids.add(sample_id)
                        successful_lines.append(line)
            except json.JSONDecodeError:
                # 跳过格式错误的行
                continue
    return successful_ids, successful_lines

def filter_file_by_ids(filepath: str, ids_to_keep: Set[str]) -> List[str]:
    """
    根据给定的ID集合，过滤一个JSONL文件，返回需要保留的行列表。
    """
    lines_to_keep = []
    if not os.path.exists(filepath):
        return lines_to_keep
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                sample_id = get_id(data)
                if sample_id and sample_id in ids_to_keep:
                    lines_to_keep.append(line)
            except json.JSONDecodeError:
                continue
    return lines_to_keep

def safe_overwrite(filepath: str, lines: List[str]):
    """
    安全地覆写文件：先备份原文件，再写入新内容。
    """
    backup_path = filepath + ".bak"
    logging.info(f"    -> 创建备份文件: {os.path.basename(backup_path)}")
    shutil.copy(filepath, backup_path) # 使用copy而不是move，更安全
    
    logging.info(f"    -> 写入 {len(lines)} 行到: {os.path.basename(filepath)}")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def process_directory(directory: str, dry_run: bool = False):
    """
    对包含目标文件对的单个目录进行联动清理。
    """
    eval_file = os.path.join(directory, "predictions_detailed_eval.jsonl")
    pred_file = os.path.join(directory, "predictions.jsonl")
    
    logging.info("-" * 60)
    logging.info(f"正在处理目录: {directory}")

    # 步骤 1: 从 eval 文件中获取 hit=1 的 ID 和数据行
    successful_ids, eval_lines_to_keep = find_successful_ids(eval_file)
    original_eval_count = sum(1 for line in open(eval_file, 'r'))

    if not successful_ids:
        logging.warning("未能从 'predictions_detailed_eval.jsonl' 中找到任何 hit=1 的样本，此目录将被跳过。")
        return

    logging.info(f"  -> 从 '{os.path.basename(eval_file)}' 中识别出 {len(successful_ids)} 个成功样本ID。")

    # 步骤 2: 根据ID列表，筛选 'predictions.jsonl'
    pred_lines_to_keep = filter_file_by_ids(pred_file, successful_ids)
    original_pred_count = sum(1 for line in open(pred_file, 'r'))

    # 步骤 3: 报告或执行
    if dry_run:
        logging.info(f"[演习模式] 针对 '{os.path.basename(eval_file)}':")
        logging.info(f"    -> 将保留 {len(eval_lines_to_keep)} 行，移除 {original_eval_count - len(eval_lines_to_keep)} 行。")
        logging.info(f"[演习模式] 针对 '{os.path.basename(pred_file)}':")
        logging.info(f"    -> 将保留 {len(pred_lines_to_keep)} 行，移除 {original_pred_count - len(pred_lines_to_keep)} 行。")
        return

    # --- 实际执行 ---
    logging.info(f"正在清理文件对...")
    safe_overwrite(eval_file, eval_lines_to_keep)
    safe_overwrite(pred_file, pred_lines_to_keep)
    logging.info("清理完成。")


def main():
    parser = argparse.ArgumentParser(
        description="联动清理 'predictions.jsonl' 和 'predictions_detailed_eval.jsonl' 文件，仅保留 hit=1 的记录。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default="/mnt/wangjingxiong/think_on_graph/results/all_result/RoG-cwq/mpo/deepseek-chat/iterative-rounds2-topk3",
        help="需要清理的目标根目录。"
    )
    parser.add_argument(
        "--dry-run",
        action="store_false",
        help="演习模式：只打印将要进行的更改，不实际修改任何文件。"
    )
    args = parser.parse_args()

    if args.dry_run:
        logging.info("="*20 + " 运行在演习模式 (Dry Run) " + "="*20)
        logging.info("将仅显示操作计划，不会对文件进行任何实际修改。")

    # 查找所有包含目标文件对的目录
    target_dirs = []
    for root, _, files in os.walk(args.directory):
        if "predictions.jsonl" in files and "predictions_detailed_eval.jsonl" in files:
            target_dirs.append(root)

    if not target_dirs:
        logging.warning(f"在 '{args.directory}' 中未找到任何同时包含 'predictions.jsonl' 和 'predictions_detailed_eval.jsonl' 的目录。")
        return
        
    logging.info(f"共找到 {len(target_dirs)} 个包含目标文件对的目录，准备开始处理...")

    for directory in target_dirs:
        process_directory(directory, args.dry_run)

    logging.info("="*20 + " 所有任务完成 " + "="*20)


if __name__ == "__main__":
    main()