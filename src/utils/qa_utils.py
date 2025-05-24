from collections import OrderedDict
import json
import re
import string
from statistics import mean # 在 eval_joint_result 中使用 (虽然此函数未在当前重构范围内，但保留导入)
import os
import logging # 添加 logging 模块

# 基本的日志配置 (与 main.py 脚本一致)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def normalize(s: str) -> str:
    """将文本小写，移除标点、冠词和多余空格。"""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s) # 移除 <pad> token
    s = " ".join(s.split())
    return s

def old_match(s1: str, s2: str) -> bool:
    """
    原始的 match 函数，检查 s2 是否是 s1 的子串 (标准化后)。
    保留此函数以明确 eval_acc 和 eval_hit 的原始行为。
    """
    s1_normalized = normalize(s1)
    s2_normalized = normalize(s2)
    return s2_normalized in s1_normalized

def eval_acc(prediction_concatenated: str, answer_list: list) -> float:
    """
    计算准确率 (原始定义): ground_truth 中有多少项在连接后的预测字符串中出现。
    'prediction_concatenated' 是所有预测项连接后的单个字符串。
    'answer_list' 是一个包含多个标准答案字符串的列表。
    """
    matched_count = 0.0
    if not answer_list:
        return 0.0
    for ans_item in answer_list:
        if old_match(prediction_concatenated, ans_item): # 使用原始的子串匹配
            matched_count += 1
    return matched_count / len(answer_list)

def eval_hit(prediction_concatenated: str, answer_list: list) -> int:
    """
    计算命中率 (原始定义): ground_truth 中是否至少有一项在连接后的预测字符串中出现。
    'prediction_concatenated' 是所有预测项连接后的单个字符串。
    'answer_list' 是一个包含多个标准答案字符串的列表。
    """
    if not answer_list:
        return 0
    for ans_item in answer_list:
        if old_match(prediction_concatenated, ans_item): # 使用原始的子串匹配
            return 1
    return 0

def eval_f1_set_based(prediction_list: list, answer_list: list) -> tuple:
    """
    计算基于集合的F1分数、精确率和召回率 (标准定义)。
    比较的是标准化后的预测项与标准答案项之间的精确匹配。
    """
    # 标准化预测项和答案项
    normalized_predictions = {normalize(str(p)) for p in prediction_list if str(p).strip()}
    normalized_answers = {normalize(str(a)) for a in answer_list if str(a).strip()}

    if not normalized_answers: # 标准答案集为空
        if not normalized_predictions: # 预测集也为空
            return 1.0, 1.0, 1.0  # F1, Precision, Recall (约定：正确地预测了“无答案”)
        else: # 标准答案集为空，但预测集不为空
            return 0.0, 0.0, 0.0  # F1, Precision, Recall (错误地预测了答案)
    
    # 从此处开始，normalized_answers 不为空
    if not normalized_predictions: # 预测集为空，但标准答案集不为空
        return 0.0, 0.0, 0.0  # F1, Precision, Recall (未能预测出任何答案)

    # 从此处开始，normalized_predictions 和 normalized_answers 均不为空
    true_positives = len(normalized_predictions.intersection(normalized_answers))
    
    precision = true_positives / len(normalized_predictions)
    recall = true_positives / len(normalized_answers)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        
    return f1, precision, recall


def extract_topk_prediction(prediction_input, k=-1):
    """
    从输入中提取top-k预测。
    输入可以是单个字符串（逗号或换行符分隔）或字符串列表（其中每个字符串也可能包含换行符）。
    返回一个经过处理的字符串列表（已去除首尾空格并过滤空字符串）。
    """
    raw_items_list = []
    if isinstance(prediction_input, str):
        # 首先按逗号分割，得到一个初步的列表
        comma_split_list = prediction_input.split(',')
        # 然后，对这个列表中的每一项，再按换行符分割
        for item_from_comma_split in comma_split_list:
            newline_split_items = item_from_comma_split.split('\n')
            for sub_item in newline_split_items:
                raw_items_list.append(sub_item)
    elif isinstance(prediction_input, list):
        # 如果输入是列表，对列表中的每一项按换行符分割
        for item_from_list_input in prediction_input:
            item_str = str(item_from_list_input) # 确保是字符串
            newline_split_items = item_str.split('\n')
            for sub_item in newline_split_items:
                raw_items_list.append(sub_item)
    else: # 其他类型，尝试转为字符串并按换行符分割
        try:
            item_str = str(prediction_input)
            newline_split_items = item_str.split('\n')
            for sub_item in newline_split_items:
                raw_items_list.append(sub_item)
        except:
            # 如果转换失败，返回空列表
            return []

    # 去除首尾空格并过滤掉处理后产生的空字符串
    processed_list = [s.strip() for s in raw_items_list if s.strip()]

    if not processed_list:
        return []

    # 后续的频率统计和 top-k 选择逻辑
    results_freq = {}
    for p_item in processed_list:
        results_freq[p_item] = results_freq.get(p_item, 0) + 1
    
    # 按频率排序（如果需要，虽然对于F1的集合操作，顺序和频率不重要，但保留以防其他用途）
    sorted_results_by_freq = sorted(results_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 应用 top-k
    actual_k = k
    if k < 0 or k > len(sorted_results_by_freq): # k=-1 表示取全部
        actual_k = len(sorted_results_by_freq)
        
    final_prediction_list = [r[0] for r in sorted_results_by_freq[:actual_k]]
    
    return final_prediction_list


def eval_path_result_w_ans(predict_file_path: str, cal_f1: bool = True, topk: int = -1, output_path_prefix: str = None):
    """
    评估预测文件中的每一行，计算指标。
    'predict_file_path': 预测文件的路径 (JSONL格式)。
    'cal_f1': 是否计算F1, Precision, Recall (使用基于集合的精确匹配)。
    'topk': 从每个样本的预测中考虑top-k项 (基于extract_topk_prediction的逻辑)。
             -1 表示全部。
    'output_path_prefix': 如果提供，则详细评估结果和总结结果将保存在以此为前缀的目录/文件名中。
                         例如，如果 output_path_prefix = "results/my_experiment",
                         则文件会是 "results/my_experiment_detailed_eval_top_k.jsonl"
                         和 "results/my_experiment_eval_result_top_k.txt"。
                         如果为 None，则在 predict_file_path 相同目录下生成。
    """
    
    base_dir = os.path.dirname(predict_file_path)
    base_filename = os.path.basename(predict_file_path)

    if output_path_prefix:
        # 如果 output_path_prefix 是一个目录
        if os.path.isdir(output_path_prefix) or (not os.path.splitext(output_path_prefix)[1] and output_path_prefix.endswith(os.sep)):
            output_dir = output_path_prefix
            # 使用原始文件名（不含扩展名）作为前缀的一部分，去除 "predictions"
            file_prefix_from_input = os.path.splitext(base_filename)[0].replace("predictions", "")
            # 确保前缀不以斜杠结尾，除非它是根目录
            if file_prefix_from_input.endswith(os.sep):
                 file_prefix_from_input = file_prefix_from_input[:-1]
            if file_prefix_from_input.startswith(os.sep): # 避免双斜杠
                 file_prefix_from_input = file_prefix_from_input[1:]

            prefix_name = file_prefix_from_input

        # 如果 output_path_prefix 是一个文件前缀 (可能包含路径)
        else:
            output_dir = os.path.dirname(output_path_prefix)
            prefix_name = os.path.basename(output_path_prefix)
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_dir
        prefix_name = base_filename.replace("predictions.jsonl", "")


    detailed_eval_filename = f"{prefix_name}detailed_eval_result"
    summary_filename = f"{prefix_name}eval_result"
    if topk > 0: # 只有当 topk 是一个正数时才添加到文件名中
        detailed_eval_filename += f"_top_{topk}"
        summary_filename += f"_top_{topk}"
    detailed_eval_filename += ".jsonl"
    summary_filename += ".txt"

    detailed_eval_file = os.path.join(output_dir, detailed_eval_filename)
    summary_eval_file = os.path.join(output_dir, summary_filename)

    # 指标列表
    all_acc_scores = []
    all_hit_scores = []
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []
    
    valid_samples_count = 0

    with open(predict_file_path, "r", encoding="utf-8") as f_in, \
         open(detailed_eval_file, "w", encoding="utf-8") as f_out_detailed:
        
        for line_num, line in enumerate(f_in):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"跳过无效的JSON行 (行号 {line_num + 1}): {line.strip()}")
                continue
            
            sample_id = data.get("id", f"sample_{line_num+1}")
            raw_prediction_input = data.get("prediction") # 可能是字符串或列表
            ground_truth_answers = list(set(data.get("ground_truth", []))) # 去重标准答案

            # 初始化当前样本的指标
            current_acc = 0.0
            current_hit = 0
            current_f1, current_precision, current_recall = 0.0, 0.0, 0.0
            processed_predictions_for_f1 = []
            prediction_str_for_acc_hit = ""


            if not ground_truth_answers:
                logging.info(f"跳过样本 {sample_id} 因为其标准答案列表为空。")
                # 写入一个空结果或标记，以便知道此样本被跳过
                # 对于F1等指标，如果GT为空，预测也为空，则P/R/F1为1，否则为0
                if raw_prediction_input is None or not extract_topk_prediction(raw_prediction_input, topk):
                    current_f1, current_precision, current_recall = 1.0, 1.0, 1.0
                else:
                    current_f1, current_precision, current_recall = 0.0, 0.0, 0.0
                
                error_output = {
                    "id": sample_id, 
                    "prediction_to_eval_for_f1": extract_topk_prediction(raw_prediction_input, topk) if raw_prediction_input is not None else [], 
                    "ground_truth": [],
                    "ans_acc": 0.0, # Acc/Hit 依赖于答案列表长度，答案为空则为0
                    "ans_hit": 0, 
                    "info": "Empty ground_truth"
                }
                if cal_f1:
                    error_output.update({"ans_f1": current_f1, "ans_precision": current_precision, "ans_recall": current_recall})
                f_out_detailed.write(json.dumps(error_output) + "\n")
                # 即使跳过，也需要将这些（可能为0或1的）指标计入总数，如果这是期望的行为
                # 或者，完全不计入 valid_samples_count，这里选择不计入，因为主要评估逻辑未执行
                continue # 跳过此样本的后续累积

            # 从此处开始，ground_truth_answers 不为空

            if raw_prediction_input is None:
                logging.info(f"样本 {sample_id} 的预测为空 (None)。F1/P/R 将为0。")
                # processed_predictions_for_f1 保持为空列表 []
                # prediction_str_for_acc_hit 保持为空字符串 ""
                # current_acc, current_hit, current_f1, current_precision, current_recall 保持 0.0
                pass # 指标已初始化为0

            else:
                # 步骤 1: 使用 extract_topk_prediction 处理原始预测
                list_after_topk = extract_topk_prediction(raw_prediction_input, topk)
                prediction_str_for_acc_hit = " ".join(str(p) for p in list_after_topk)

                # 步骤 2: 从 list_after_topk 中进一步提取用于F1评估的答案
                # (例如，如果预测项中包含特定标记如 "# Answer:\n")
                # processed_predictions_for_f1 用于 eval_f1_set_based
                predicted_ans_extracted_for_f1 = []
                if cal_f1: # 仅当需要计算F1时才进行此提取
                    for p_item_str in list_after_topk:
                        p_item_str_safe = str(p_item_str) # 确保是字符串
                        if "# Answer:\n" in p_item_str_safe: # 假设这是答案标记
                            ans_segment = p_item_str_safe.split("# Answer:\n", 1)[-1]
                            for single_ans_line in ans_segment.splitlines():
                                stripped_ans = single_ans_line.strip()
                                if stripped_ans:
                                    predicted_ans_extracted_for_f1.append(stripped_ans)
                        else: # 如果没有特定标记，则将整个处理过的预测项视为一个答案候选项
                            stripped_p = p_item_str_safe.strip()
                            if stripped_p:
                                predicted_ans_extracted_for_f1.append(stripped_p)
                    processed_predictions_for_f1 = predicted_ans_extracted_for_f1
                else: 
                    # 如果不计算F1，processed_predictions_for_f1 可以是 list_after_topk 或空
                    # 为了输出一致性，可以设为 list_after_topk
                    processed_predictions_for_f1 = list_after_topk


                # 计算指标
                current_acc = eval_acc(prediction_str_for_acc_hit, ground_truth_answers)
                current_hit = eval_hit(prediction_str_for_acc_hit, ground_truth_answers)
                
                if cal_f1:
                    current_f1, current_precision, current_recall = eval_f1_set_based(
                        processed_predictions_for_f1, ground_truth_answers
                    )
                # else: current_f1, current_precision, current_recall 保持 0.0

            # 累积指标
            all_acc_scores.append(current_acc)
            all_hit_scores.append(current_hit)
            if cal_f1:
                all_f1_scores.append(current_f1)
                all_precision_scores.append(current_precision)
                all_recall_scores.append(current_recall)
            
            valid_samples_count += 1
            
            # 准备详细输出的JSON行
            output_line_data = {
                "id": sample_id,
                "prediction_to_eval_for_f1": processed_predictions_for_f1, 
                "ground_truth": ground_truth_answers,
                "ans_acc": current_acc,
                "ans_hit": current_hit,
            }
            if cal_f1:
                output_line_data["ans_f1"] = current_f1
                output_line_data["ans_precision"] = current_precision
                output_line_data["ans_recall"] = current_recall
            
            f_out_detailed.write(json.dumps(output_line_data) + "\n")

    # 计算并打印/保存最终的平均指标
    summary_metrics = {}
    result_str_parts = []

    if valid_samples_count == 0:
        logging.warning("没有有效的样本进行评估。")
        summary_metrics = {"accuracy": 0.0, "hit_rate": 0.0}
        result_str_parts.extend(["Accuracy: 0.00%", "Hit Rate: 0.00%"])
        if cal_f1:
            summary_metrics.update({"f1": 0.0, "precision": 0.0, "recall": 0.0})
            result_str_parts.extend(["F1: 0.00%", "Precision: 0.00%", "Recall: 0.00%"])
    else:
        avg_acc = sum(all_acc_scores) / valid_samples_count
        avg_hit = sum(all_hit_scores) / valid_samples_count 
        summary_metrics.update({"accuracy": avg_acc, "hit_rate": avg_hit})
        result_str_parts.extend([
            f"Accuracy: {avg_acc * 100:.2f}%",
            f"Hit Rate: {avg_hit * 100:.2f}%"
        ])
        
        if cal_f1 : # 即使 all_f1_scores 为空 (如果所有有效样本的F1都为0)，也应计算平均值
            avg_f1 = sum(all_f1_scores) / valid_samples_count if all_f1_scores else 0.0
            avg_precision = sum(all_precision_scores) / valid_samples_count if all_precision_scores else 0.0
            avg_recall = sum(all_recall_scores) / valid_samples_count if all_recall_scores else 0.0
            summary_metrics.update({
                "f1": avg_f1, "precision": avg_precision, "recall": avg_recall
            })
            result_str_parts.extend([
                f"F1: {avg_f1 * 100:.2f}%",
                f"Precision: {avg_precision * 100:.2f}%",
                f"Recall: {avg_recall * 100:.2f}%"
            ])

    final_result_str = " | ".join(result_str_parts)
    logging.info(f"评估文件: {predict_file_path}")
    logging.info(f"最终平均指标 ({valid_samples_count} 个有效样本): {final_result_str}")
    
    with open(summary_eval_file, "w", encoding="utf-8") as f_summary:
        f_summary.write(f"Evaluated File: {predict_file_path}\n")
        f_summary.write(f"Number of Valid Samples: {valid_samples_count}\n")
        f_summary.write(f"Metrics: {final_result_str}\n")
        f_summary.write(f"Detailed results saved to: {detailed_eval_file}\n")

    logging.info(f"详细评估结果已保存到: {detailed_eval_file}")
    logging.info(f"总结评估结果已保存到: {summary_eval_file}")
            
    return summary_metrics


# --- 其他评估函数 (eval_result, eval_rank_results, eval_joint_result, eval_path_result) ---
# 这些函数在您的原始代码中存在，但未在 main.py 中直接调用 eval_path_result_w_ans。
# 如果需要重构它们，请告知。目前仅重构了 eval_path_result_w_ans 及其依赖的核心指标函数。
# 为保持完整性，这里可以复制粘贴它们，或者根据需要进行调整。
# 为了聚焦于 eval_path_result_w_ans 的重构，暂时省略其他未直接调用的 eval_* 函数。
# 如果您希望它们也被包含和审查/重构，请明确指出。

