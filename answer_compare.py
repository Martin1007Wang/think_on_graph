import os
import json
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set, Optional

# 定义目录路径
dir1 = "/mnt/wangjingxiong/think_on_graph/results/KGQA/RoG-cwq/deepseek-chat/test/add_path_results_GenPaths_RoG-cwq_GCR-Qwen2-7B-Instruct_test_zero-shot-gr_581f65c9473a5160adf85530c0451637_no_dup"
dir2 = "/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v13/RoG-cwq/lora_naive_preference_dataset_combined_webqsp_pn_only_shortest_paths_epoch_1/deepseek-chat/iterative-rounds2-topk3"
reference_id_file = "/mnt/wangjingxiong/think_on_graph/data/processed/rmanluo_RoG-cwq_test/path_data_temp.jsonl"


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件内容"""
    data = []
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在。")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"错误: 解析文件 {file_path} 第 {i+1} 行时出错: {line.strip()}")
    return data

def load_reference_ids(file_path: str) -> Set[str]:
    """从指定的JSONL文件中加载ID列表。"""
    ids = set()
    if not os.path.exists(file_path):
        print(f"警告: 参考ID文件 {file_path} 不存在。将不按此文件进行ID过滤。")
        return ids # 返回空集合，后续交集操作不会改变common_ids

    print(f"从 {file_path} 加载参考ID...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data_item = json.loads(line)
                if "id" in data_item:
                    ids.add(str(data_item["id"])) # 确保ID是字符串类型
                else:
                    print(f"警告: 参考文件 {file_path} 第 {i+1} 行缺少 'id' 字段。")
            except json.JSONDecodeError:
                print(f"错误: 解析参考文件 {file_path} 第 {i+1} 行时出错: {line.strip()}")
    print(f"加载了 {len(ids)} 个唯一的参考ID。")
    return ids

def analyze_method_differences(dir1: str, dir2: str, reference_ids: Optional[Set[str]] = None) -> None:
    """分析方法1正确但方法2错误的案例 (Regression Cases for Method 2)"""
    print("\n=== 分析方法1正确，但方法2错误的案例 (方法2的退化点) ===")
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    pred2_path = os.path.join(dir2, "predictions.jsonl")

    if not os.path.exists(pred1_path) or not os.path.exists(pred2_path):
        print(f"错误: 预测文件 {pred1_path} 或 {pred2_path} 不存在。")
        return

    pred1 = load_jsonl(pred1_path)
    pred2 = load_jsonl(pred2_path)

    pred1_dict = {item["id"]: item for item in pred1}
    pred2_dict = {item["id"]: item for item in pred2}

    common_ids = set(pred1_dict.keys()) & set(pred2_dict.keys())
    
    if reference_ids is not None and len(reference_ids) > 0: #仅当参考ID非空时才进行过滤
        initial_common_count = len(common_ids)
        common_ids &= reference_ids
        print(f"与参考ID列表取交集后，用于此分析的共同ID数量从 {initial_common_count} 变为 {len(common_ids)}。")
    elif reference_ids is not None and len(reference_ids) == 0:
        print(f"警告: 提供的参考ID列表为空，此分析将基于0个ID。")
        common_ids = set()


    if not common_ids:
        print("在两个预测文件与参考ID列表的交集中找不到共同的问题ID。")
        return
    
    total_failures_method2 = 0
    failure_categories = defaultdict(int)
    failure_examples = defaultdict(list)
    regression_case_ids_method2 = []
    regression_case_details_method2 = []
    total_common_cases_analyzed = len(common_ids)

    for id_val in common_ids:
        item1 = pred1_dict[id_val]
        item2 = pred2_dict[id_val]
        
        gt_raw = item1.get("ground_truth", item2.get("ground_truth", [])) 
        gt_set = set()
        if isinstance(gt_raw, list):
            gt_set = {str(g).strip().lower() for g in gt_raw if str(g).strip()}
        elif isinstance(gt_raw, str):
            gt_set = {gt_raw.strip().lower()} if gt_raw.strip() else set()

        pred1_text_raw = item1.get("prediction", "")
        pred1_answers_processed = []
        if isinstance(pred1_text_raw, list):
            for p in pred1_text_raw:
                if isinstance(p, str) and "# Answer:" in p:
                    answer = p.split("# Answer:")[-1].strip()
                    pred1_answers_processed.append(answer)
            pred1_final_text_for_eval = ", ".join(pred1_answers_processed) if pred1_answers_processed else str(pred1_text_raw)
        elif isinstance(pred1_text_raw, str) and "# Answer:" in pred1_text_raw:
             pred1_final_text_for_eval = pred1_text_raw.split("# Answer:")[-1].strip()
        else:
            pred1_final_text_for_eval = str(pred1_text_raw)

        pred2_text_raw = item2.get("prediction", "")
        if isinstance(pred2_text_raw, list):
            pred2_final_text_for_eval = ", ".join(map(str, pred2_text_raw)) if pred2_text_raw else ""
        else:
            pred2_final_text_for_eval = str(pred2_text_raw)
        
        m1_correct = any(g_ans in pred1_final_text_for_eval.lower() for g_ans in gt_set) if gt_set and pred1_final_text_for_eval else False
        m2_correct = any(g_ans in pred2_final_text_for_eval.lower() for g_ans in gt_set) if gt_set and pred2_final_text_for_eval else False

        if m1_correct and not m2_correct:
            total_failures_method2 += 1
            regression_case_ids_method2.append(id_val)
            # ... (rest of your failure analysis logic for m1_correct and not m2_correct) ...
            failure_reason = "未知原因" 
            if "m." in pred2_final_text_for_eval: 
                m_codes = re.findall(r'm\.[0-9a-z_]+', pred2_final_text_for_eval)
                failure_reason = "m.编码问题"
                has_m_code_in_pred1 = "m." in pred1_final_text_for_eval
                if not has_m_code_in_pred1:
                    failure_reason = "m.编码问题(仅方法2)"
            elif not pred2_final_text_for_eval.strip(): 
                failure_reason = "空预测"
            elif "exploration_history" in item2 and not item2.get("answer_found_during_exploration", True):
                failure_reason = "探索未找到答案"
            elif isinstance(pred2_text_raw, list) and not pred2_text_raw: 
                failure_reason = "空列表预测"
            elif not isinstance(pred2_text_raw, (str, list)): 
                failure_reason = "预测类型异常"
            
            failure_categories[failure_reason] += 1
            
            exploration_issue_info = ""
            if "exploration_history" in item2: 
                has_m_code_in_exploration = False
                for round_data in item2.get("exploration_history", []):
                    for expansion in round_data.get("expansions", []):
                        for relation_item in expansion.get("relations", []):
                            if any("m." in str(target) for target in relation_item.get("targets", [])):
                                has_m_code_in_exploration = True; break
                        if has_m_code_in_exploration: break
                    if has_m_code_in_exploration: break
                if has_m_code_in_exploration:
                    exploration_issue_info = "探索历史包含m.编码"

            if len(failure_examples[failure_reason]) < 10: 
                 failure_examples[failure_reason].append(
                     (id_val, gt_set, pred1_final_text_for_eval, pred2_final_text_for_eval, exploration_issue_info)
                 )
            
            regression_case_details_method2.append({
                "id": id_val,
                "question": item1.get("question", item2.get("question", "N/A")),
                "ground_truth": list(gt_set) if gt_set else (gt_raw if isinstance(gt_raw, list) else [str(gt_raw)]),
                "method1_prediction": pred1_final_text_for_eval,
                "method2_prediction": pred2_final_text_for_eval,
                "failure_reason_for_method2": failure_reason,
                "method2_exploration_issue": exploration_issue_info
            })

    print(f"方法1正确但方法2错误的总案例数 (筛选后): {total_failures_method2} / {total_common_cases_analyzed} ({(total_failures_method2/total_common_cases_analyzed*100 if total_common_cases_analyzed else 0):.2f}%)")
    # ... (rest of your printing and saving logic) ...
    print("\n方法2主要失败原因统计:")
    for reason, count in sorted(failure_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_failures_method2) * 100 if total_failures_method2 > 0 else 0
        print(f"- {reason}: {count} ({percentage:.2f}%)")

    print("\n=== 方法2典型失败案例 ===")
    for reason, examples in failure_examples.items():
        print(f"\n【方法2失败原因: {reason}】")
        for i, (ex_id, ex_gt, ex_pred1, ex_pred2, ex_exp_issue) in enumerate(examples[:5], 1): # Show top 5
            print(f"案例 {i}: ID: {ex_id}")
            print(f"  Ground Truth: {ex_gt}")
            print(f"  方法1 预测 (正确): {ex_pred1}")
            print(f"  方法2 预测 (错误): {ex_pred2}")
            if ex_exp_issue:
                print(f"  方法2 探索历史问题: {ex_exp_issue}")
            print()

    if regression_case_ids_method2:
        id_path = os.path.join(dir2, "method2_regression_filtered_case_ids.txt") 
        with open(id_path, "w", encoding='utf-8') as f:
            for case_id in regression_case_ids_method2:
                f.write(str(case_id) + "\n")
        print(f"\n方法1正确但方法2错误的筛选后case id已保存到: {id_path}")

    if regression_case_details_method2:
        detail_path = os.path.join(dir2, "method2_regression_filtered_case_details.jsonl") 
        with open(detail_path, "w", encoding="utf-8") as f:
            for case_detail in regression_case_details_method2:
                json.dump(case_detail, f, ensure_ascii=False)
                f.write("\n")
        print(f"方法1正确但方法2错误的筛选后详细case对比信息已保存到: {detail_path}")


def analyze_ground_truth_and_evaluation_logic(dir1: str, dir2: str, reference_ids: Optional[Set[str]] = None) -> None:
    """验证ground_truth中m.编码的比例和评分逻辑, 仅针对参考ID列表中的ID"""
    print("\n=== Ground Truth 分析 (基于方法1数据, 筛选后) ===")
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    det1_path = os.path.join(dir1, "detailed_eval_result.jsonl") 
    
    if not os.path.exists(pred1_path):
        print(f"预测文件 {pred1_path} 不存在。")
        return
    
    pred1_all = load_jsonl(pred1_path)
    det1_all = []
    if os.path.exists(det1_path):
        det1_all = load_jsonl(det1_path)
    else:
        print(f"警告: 详细评估文件 {det1_path} 不存在，部分评分逻辑分析将受限。")

    # Filter by reference_ids if provided
    if reference_ids is not None and len(reference_ids) > 0:
        pred1 = [item for item in pred1_all if item["id"] in reference_ids]
        det1 = [item for item in det1_all if item["id"] in reference_ids]
        print(f"Ground Truth分析将基于 {len(pred1)} 个筛选后的案例。")
        if not pred1:
            print("筛选后没有案例可用于Ground Truth分析。")
            return
    elif reference_ids is not None and len(reference_ids) == 0: # Empty reference list given
        print("警告: 提供的参考ID列表为空，此Ground Truth分析将基于0个ID。")
        return
    else: # reference_ids is None (no filtering)
        pred1 = pred1_all
        det1 = det1_all


    pred1_dict = {item["id"]: item for item in pred1}
    det1_dict = {item["id"]: item for item in det1}
    
    # ... (rest of your ground truth analysis logic) ...
    total_questions = 0
    questions_with_m_code_gt = 0
    m_code_gt_examples = []
    
    for item in pred1: # Use filtered pred1
        if "ground_truth" not in item:
            continue
        total_questions += 1
        gt_raw = item["ground_truth"]
        
        has_m_code_in_current_gt = False
        current_m_codes = []
        
        gt_list_for_m_code_check = []
        if isinstance(gt_raw, str):
            gt_list_for_m_code_check = [gt_raw]
        elif isinstance(gt_raw, list):
            gt_list_for_m_code_check = gt_raw
        
        for g_item in gt_list_for_m_code_check:
            if "m." in str(g_item):
                has_m_code_in_current_gt = True
                current_m_codes.extend(re.findall(r'm\.[0-9a-z_]+', str(g_item)))
        
        if has_m_code_in_current_gt:
            questions_with_m_code_gt += 1
            if len(m_code_gt_examples) < 5:
                m_code_gt_examples.append((item["id"], gt_raw, list(set(current_m_codes)))) 
    
    m_code_percentage = (questions_with_m_code_gt / total_questions) * 100 if total_questions > 0 else 0
    print(f"总问题数 (筛选后方法1 GT): {total_questions}")
    print(f"Ground Truth中包含m.编码的问题数: {questions_with_m_code_gt} ({m_code_percentage:.2f}%)")
    # ... (rest of this function)


def analyze_method2_perfect_method1_failure(dir1: str, dir2: str, reference_ids: Optional[Set[str]] = None) -> None:
    """找出方法2完美(F1=1, Acc=1)而方法1完全失败(F1=0, Acc=0)的案例, 筛选后"""
    print("\n=== 分析方法2完美 & 方法1失败 的案例 (筛选后) ===")
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    eval1_path = os.path.join(dir1, "detailed_eval_result.jsonl")
    pred2_path = os.path.join(dir2, "predictions.jsonl")
    eval2_path = os.path.join(dir2, "detailed_eval_result.jsonl")

    required_files = [pred1_path, eval1_path, pred2_path, eval2_path]
    if not all(os.path.exists(p) for p in required_files):
        print("错误: 缺少必要的预测或评估文件。")
        missing = [p for p in required_files if not os.path.exists(p)]
        print(f"缺失的文件: {', '.join(missing)}")
        return

    pred1_data = load_jsonl(pred1_path)
    eval1_data = load_jsonl(eval1_path)
    pred2_data = load_jsonl(pred2_path)
    eval2_data = load_jsonl(eval2_path)

    pred1_dict = {item["id"]: item for item in pred1_data}
    eval1_dict = {item["id"]: item for item in eval1_data}
    pred2_dict = {item["id"]: item for item in pred2_data}
    eval2_dict = {item["id"]: item for item in eval2_data}

    common_ids = set(pred1_dict.keys()) & set(eval1_dict.keys()) & set(pred2_dict.keys()) & set(eval2_dict.keys())
    
    if reference_ids is not None and len(reference_ids) > 0:
        initial_common_count = len(common_ids)
        common_ids &= reference_ids
        print(f"与参考ID列表取交集后，用于此分析的共同ID数量从 {initial_common_count} 变为 {len(common_ids)}。")
    elif reference_ids is not None and len(reference_ids) == 0:
        print(f"警告: 提供的参考ID列表为空，此分析将基于0个ID。")
        common_ids = set()

    if not common_ids:
        print("在所有文件与参考ID列表的交集中找不到共同的问题ID。")
        return
    
    found_cases_m2_perfect_m1_fail = []
    # ... (rest of your logic for this function, using the filtered common_ids) ...
    for id_val in common_ids:
        item1_pred_content = pred1_dict[id_val]
        item1_eval_scores = eval1_dict[id_val]
        item2_pred_content = pred2_dict[id_val]
        item2_eval_scores = eval2_dict[id_val]

        m2_is_perfect = item2_eval_scores.get('ans_f1', 0.0) == 1.0 and item2_eval_scores.get('ans_acc', 0.0) == 1.0
        m1_is_failure = item1_eval_scores.get('ans_f1', 1.0) == 0.0 and item1_eval_scores.get('ans_acc', 1.0) == 0.0

        if m2_is_perfect and m1_is_failure:
            # ... (data extraction and appending to found_cases_m2_perfect_m1_fail)
            question_text = item1_pred_content.get("question", item2_pred_content.get("question", "N/A"))
            gt_text = item1_pred_content.get("ground_truth", item2_pred_content.get("ground_truth", "N/A"))
            
            pred1_text_raw = item1_pred_content.get("prediction", "N/A")
            if isinstance(pred1_text_raw, list):
                answers = [p.split("# Answer:")[-1].strip() for p in pred1_text_raw if isinstance(p, str) and "# Answer:" in p]
                pred1_cleaned = ", ".join(answers) if answers else str(pred1_text_raw)
            elif isinstance(pred1_text_raw, str) and "# Answer:" in pred1_text_raw:
                pred1_cleaned = pred1_text_raw.split("# Answer:")[-1].strip()
            else:
                pred1_cleaned = str(pred1_text_raw)

            pred2_text_raw = item2_pred_content.get("prediction", "N/A")
            if isinstance(pred2_text_raw, list):
                pred2_cleaned = ", ".join(map(str, pred2_text_raw)) if pred2_text_raw else "N/A"
            else:
                pred2_cleaned = str(pred2_text_raw)

            found_cases_m2_perfect_m1_fail.append({
                "id": id_val, "question": question_text, "ground_truth": gt_text,
                "method1_prediction": pred1_cleaned, 
                "method1_f1": item1_eval_scores.get('ans_f1', 0.0), "method1_acc": item1_eval_scores.get('ans_acc', 0.0),
                "method2_prediction": pred2_cleaned,
                "method2_f1": item2_eval_scores.get('ans_f1', 1.0), "method2_acc": item2_eval_scores.get('ans_acc', 1.0),
            })


    if found_cases_m2_perfect_m1_fail:
        print(f"找到 {len(found_cases_m2_perfect_m1_fail)} 个方法2完美而方法1失败的案例 (筛选后):")
        # ... (printing logic)
    else:
        print("未找到满足条件 (方法2完美 & 方法1失败) 的案例 (筛选后)。")


def analyze_method2_outperforms_method1(dir1: str, dir2: str, 
                                        reference_ids: Optional[Set[str]] = None, 
                                        f1_threshold_diff: float = 0.0, 
                                        acc_threshold_diff: float = 0.0) -> None:
    """找出方法2在F1和Accuracy上均优于方法1的案例, 筛选后。"""
    print(f"\n=== 分析方法2优于方法1 (F1 Diff > {f1_threshold_diff}, Acc Diff > {acc_threshold_diff}) 的案例 (筛选后) ===")
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    eval1_path = os.path.join(dir1, "detailed_eval_result.jsonl")
    pred2_path = os.path.join(dir2, "predictions.jsonl")
    eval2_path = os.path.join(dir2, "detailed_eval_result.jsonl")

    required_files = [pred1_path, eval1_path, pred2_path, eval2_path]
    if not all(os.path.exists(p) for p in required_files):
        print("错误: 缺少必要的预测或评估文件。")
        missing = [p for p in required_files if not os.path.exists(p)]
        print(f"缺失的文件: {', '.join(missing)}")
        return

    pred1_data = load_jsonl(pred1_path)
    eval1_data = load_jsonl(eval1_path)
    pred2_data = load_jsonl(pred2_path)
    eval2_data = load_jsonl(eval2_path)

    pred1_dict = {item["id"]: item for item in pred1_data}
    eval1_dict = {item["id"]: item for item in eval1_data}
    pred2_dict = {item["id"]: item for item in pred2_data}
    eval2_dict = {item["id"]: item for item in eval2_data}

    common_ids = set(pred1_dict.keys()) & set(eval1_dict.keys()) & set(pred2_dict.keys()) & set(eval2_dict.keys())
    
    if reference_ids is not None and len(reference_ids) > 0:
        initial_common_count = len(common_ids)
        common_ids &= reference_ids
        print(f"与参考ID列表取交集后，用于此分析的共同ID数量从 {initial_common_count} 变为 {len(common_ids)}。")
    elif reference_ids is not None and len(reference_ids) == 0:
        print(f"警告: 提供的参考ID列表为空，此分析将基于0个ID。")
        common_ids = set()


    if not common_ids:
        print("在所有文件与参考ID列表的交集中找不到共同的问题ID。")
        return
    
    outperforming_cases = []
    outperforming_case_ids = []
    # ... (rest of your logic using filtered common_ids) ...
    for id_val in common_ids:
        item1_pred = pred1_dict.get(id_val) 
        item1_eval = eval1_dict.get(id_val)
        item2_pred = pred2_dict.get(id_val)
        item2_eval = eval2_dict.get(id_val)

        if not all([item1_pred, item1_eval, item2_pred, item2_eval]):
            continue

        m1_f1 = item1_eval.get('ans_f1', 0.0)
        m1_acc = item1_eval.get('ans_acc', 0.0)
        m2_f1 = item2_eval.get('ans_f1', 0.0)
        m2_acc = item2_eval.get('ans_acc', 0.0)

        is_outperforming = (m2_f1 > m1_f1 + f1_threshold_diff) and \
                           (m2_acc > m1_acc + acc_threshold_diff)
        
        if is_outperforming:
            # ... (data extraction and appending to outperforming_cases)
            question = item1_pred.get("question", item2_pred.get("question", "N/A"))
            gt_raw = item1_pred.get("ground_truth", item2_pred.get("ground_truth", "N/A"))
            
            pred1_text_raw = item1_pred.get("prediction", "N/A")
            if isinstance(pred1_text_raw, list):
                answers = [p.split("# Answer:")[-1].strip() for p in pred1_text_raw if isinstance(p, str) and "# Answer:" in p]
                pred1_text_cleaned = ", ".join(answers) if answers else str(pred1_text_raw)
            elif isinstance(pred1_text_raw, str) and "# Answer:" in pred1_text_raw:
                pred1_text_cleaned = pred1_text_raw.split("# Answer:")[-1].strip()
            else:
                pred1_text_cleaned = str(pred1_text_raw)

            pred2_text_raw = item2_pred.get("prediction", "N/A")
            if isinstance(pred2_text_raw, list):
                pred2_text_cleaned = ", ".join(map(str, pred2_text_raw)) if pred2_text_raw else "N/A"
            else:
                pred2_text_cleaned = str(pred2_text_raw)

            outperforming_cases.append({
                "id": id_val,
                "question": question,
                "ground_truth": gt_raw, 
                "method1_prediction": pred1_text_cleaned,
                "method1_f1": m1_f1,
                "method1_acc": m1_acc,
                "method2_prediction": pred2_text_cleaned,
                "method2_f1": m2_f1,
                "method2_acc": m2_acc
            })
            outperforming_case_ids.append(id_val)

    if outperforming_cases:
        print(f"找到 {len(outperforming_cases)} 个方法2优于方法1的案例 (筛选后):")
        outperforming_cases.sort(key=lambda x: (x["method2_f1"] - x["method1_f1"]), reverse=True)

        for i, case in enumerate(outperforming_cases[:10], 1): 
            print(f"\n案例 {i}: ID: {case['id']}")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  方法1 预测: {str(case['method1_prediction'])[:200]}...")
            print(f"  方法1 F1: {case['method1_f1']:.4f}, Acc: {case['method1_acc']:.4f}")
            print(f"  方法2 预测: {str(case['method2_prediction'])[:200]}...")
            print(f"  方法2 F1: {case['method2_f1']:.4f}, Acc: {case['method2_acc']:.4f}")
        
        ids_file_path = os.path.join(dir2, "method2_outperforms_method1_filtered_ids.txt")
        with open(ids_file_path, "w", encoding='utf-8') as f:
            for case_id in outperforming_case_ids:
                f.write(str(case_id) + "\n")
        print(f"\n所有方法2优于方法1的筛选后案例ID已保存到: {ids_file_path}")

        details_file_path = os.path.join(dir2, "method2_outperforms_method1_filtered_details.jsonl")
        with open(details_file_path, "w", encoding="utf-8") as f:
            for case in outperforming_cases:
                json.dump(case, f, ensure_ascii=False)
                f.write("\n")
        print(f"所有方法2优于方法1的筛选后详细案例信息已保存到: {details_file_path}")
    else:
        print(f"未找到满足条件 (方法2 F1 > 方法1 F1 + {f1_threshold_diff} AND 方法2 Acc > 方法1 Acc + {acc_threshold_diff}) 的案例 (筛选后)。")


if __name__ == "__main__":
    print(f"分析目录1 (方法1): {dir1}")
    print(f"分析目录2 (方法2): {dir2}")
    print(f"参考ID文件: {reference_id_file}\n")

    # 加载参考ID列表
    ref_ids = load_reference_ids(reference_id_file)

    # 如果ref_ids为空集合但文件确实存在且就是空的，那么后续交集会正确处理
    # 如果ref_ids为空是因为文件不存在，load_reference_ids会打印警告并返回空集合，后续交集也会正确处理
    
    analyze_method_differences(dir1, dir2, reference_ids=ref_ids)
    
    # 注意: analyze_ground_truth_and_evaluation_logic 当前主要分析dir1的数据。
    # 如果需要它也基于reference_ids过滤，需要对其内部逻辑进行相似的修改。
    # 为保持原样，暂时不传递ref_ids，或你可以按需修改它。
    analyze_ground_truth_and_evaluation_logic(dir1, dir2, reference_ids=ref_ids) # 修改了这里，使其也使用ref_ids

    analyze_method2_perfect_method1_failure(dir1, dir2, reference_ids=ref_ids)
    analyze_method2_outperforms_method1(dir1, dir2, reference_ids=ref_ids)
    # 你可以调整阈值:
    # analyze_method2_outperforms_method1(dir1, dir2, reference_ids=ref_ids, f1_threshold_diff=0.1)