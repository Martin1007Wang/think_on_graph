import json
import argparse
import logging
from typing import Dict, List, Any

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_to_dict(filepath: str) -> Dict[str, Any]:
    """将JSONL文件加载到以'id'为键的字典中，方便快速查找。"""
    data_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        data_dict[data['id']] = data
                    else:
                        # 兼容部分文件id在别处的格式
                        if 'q_id' in data:
                           data_dict[data['q_id']] = data
                        else:
                           logging.warning(f"文件 {filepath} 的第 {line_num} 行缺少 'id'/'q_id' 字段，已跳过。")
                except json.JSONDecodeError:
                    logging.warning(f"文件 {filepath} 的第 {line_num} 行JSON格式无效，已跳过。")
    except FileNotFoundError:
        logging.error(f"错误: 文件未找到 -> {filepath}")
    return data_dict

def format_case_output(sample_id: str, question: str, ground_truth: List[str], 
                       ipo_sample: Dict[str, Any], 
                       baseline1_sample: Dict[str, Any], 
                       baseline2_sample: Dict[str, Any]) -> Dict[str, Any]:
    """格式化用于最终输出的案例信息，包含三个模型的对比。"""
    return {
        "id": sample_id,
        "question": question,
        "ground_truth": ground_truth,
        "ipo_prediction": ipo_sample.get("answer_entities", "N/A"),
        "baseline1_prediction": baseline1_sample.get("answer_entities", "N/A"),
        "baseline2_prediction": baseline2_sample.get("prediction_to_eval_for_f1", "N/A"),
        "ipo_hit": int(ipo_sample.get("metrics", {}).get("hit_rate", 0.0)),
        "baseline1_hit": int(baseline1_sample.get("metrics", {}).get("hit_rate", 0.0)),
        "baseline2_hit": int(baseline2_sample.get("ans_hit", 0))
    }

def compare_results(ipo_file: str, baseline1_file: str, baseline2_file: str):
    """主比较函数，找出改进版模型优于两个基线模型的案例。"""
    logging.info("正在加载改进模型 (IPO) 的结果...")
    ipo_data = load_data_to_dict(ipo_file)
    
    logging.info("正在加载基线模型 #1 (deepseek-chat, old) 的结果...")
    baseline1_data = load_data_to_dict(baseline1_file)
    
    logging.info("正在加载基线模型 #2 (GCR-Qwen2-7B) 的结果...")
    baseline2_data = load_data_to_dict(baseline2_file)
    
    if not ipo_data or not baseline1_data or not baseline2_data:
        logging.error("至少一个文件未能成功加载或内容为空，无法进行比较。")
        return

    # 找出三个结果文件共有的样本ID
    common_ids = set(ipo_data.keys()) & set(baseline1_data.keys()) & set(baseline2_data.keys())
    logging.info(f"发现 {len(common_ids)} 个共同样本，将在此范围内进行比较。")

    # 根据 'answer_found_during_exploration' 进行过滤
    # 注意：此字段仅存在于IPO和基线1中，基线2是生成式方法，不适用此过滤条件。
    filtered_ids = []
    for sample_id in common_ids:
        ipo_found = ipo_data[sample_id].get("answer_found_during_exploration", False)
        baseline1_found = baseline1_data[sample_id].get("answer_found_during_exploration", False)
        if ipo_found and baseline1_found:
            filtered_ids.append(sample_id)
            
    logging.info(f"对IPO和基线1进行 'answer_found_during_exploration': True 过滤后，剩余 {len(filtered_ids)} 个样本用于最终比较。")

    improvement_cases = []

    # 在过滤后的ID列表上进行迭代
    for sample_id in filtered_ids:
        ipo_sample = ipo_data[sample_id]
        b1_sample = baseline1_data[sample_id]
        b2_sample = baseline2_data[sample_id]

        # 关键：从每种不同的数据结构中提取Hit指标
        ipo_hit = int(ipo_sample.get("metrics", {}).get("hit_rate", 0.0))
        b1_hit = int(b1_sample.get("metrics", {}).get("hit_rate", 0.0))
        b2_hit = int(b2_sample.get("ans_hit", 0)) # 基线2的hit指标字段是 'ans_hit'

        question = ipo_sample.get("question", "N/A")
        ground_truth = ipo_sample.get("ground_truth", [])
        
        # 核心比较逻辑: 找出IPO成功，且两个基线都失败的案例
        if ipo_hit == 1 and b1_hit == 0 and b2_hit == 0:
            improvement_cases.append(
                format_case_output(sample_id, question, ground_truth, ipo_sample, b1_sample, b2_sample)
            )

    # --- 打印结果 ---
    print("\n" + "="*80)
    print("胜利分析 (Win-Analysis) 结果")
    print("="*80 + "\n")

    print(f"【分析】改进版(IPO)优于两个基线模型的样本 (共 {len(improvement_cases)} 例):")
    if improvement_cases:
        print(json.dumps(improvement_cases, indent=2, ensure_ascii=False))
    else:
        print("未发现此类样本。")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="胜利分析：比较IPO模型与两个基线模型，找出IPO成功而两个基线均失败的案例。")
    parser.add_argument(
        "--ipo_file",
        type=str,
        default="/mnt/wangjingxiong/think_on_graph/results/all_result/RoG-cwq/ipo/deepseek-chat/iterative-rounds2-topk3/predictions_detailed_eval.jsonl",
        help="改进版(IPO)模型的结果文件路径。"
    )
    parser.add_argument(
        "--baseline1_file",
        type=str,
        default="/mnt/wangjingxiong/think_on_graph/results/all_result/RoG-cwq/deepseek-chat/deepseek-chat/iterative-rounds2-topk3/predictions_detailed_eval.jsonl",
        help="基线模型#1 (旧版方法) 的结果文件路径。"
    )
    parser.add_argument(
        "--baseline2_file",
        type=str,
        default="/mnt/wangjingxiong/think_on_graph/results/KGQA/RoG-cwq/deepseek-chat/test/add_path_results_GenPaths_RoG-cwq_GCR-Qwen2-7B-Instruct_test_zero-shot-gr_581f65c9473a5160adf85530c0451637_no_dup/detailed_eval_result.jsonl",
        help="基线模型#2 (GCR) 的结果文件路径。"
    )
    args = parser.parse_args()
    
    compare_results(args.ipo_file, args.baseline1_file, args.baseline2_file)

if __name__ == "__main__":
    main()