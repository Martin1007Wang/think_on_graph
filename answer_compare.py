import os
import json
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set

# 定义目录路径
dir1 = "/mnt/wangjingxiong/think_on_graph/results/GCR_original/RoG-webqsp/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10-index_len2"
dir2 = "/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v4/RoG-webqsp/GCR-Llama-3.1-8B-Instruct/test/iterative-rounds3-topk5"

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件内容"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error parsing line in {file_path}")
    return data

def analyze_method_differences(dir1: str, dir2: str) -> None:
    """专注分析方法1正确而方法2错误的原因"""
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    pred2_path = os.path.join(dir2, "predictions.jsonl")
    
    if not os.path.exists(pred1_path) or not os.path.exists(pred2_path):
        print("预测文件不存在")
        return
    
    pred1 = load_jsonl(pred1_path)
    pred2 = load_jsonl(pred2_path)
    
    # 转换为ID索引的字典
    pred1_dict = {item["id"]: item for item in pred1}
    pred2_dict = {item["id"]: item for item in pred2}
    
    # 找出共有的问题ID
    common_ids = set(pred1_dict.keys()) & set(pred2_dict.keys())
    
    print(f"=== 方法结构分析 ===")
    # 分析两个方法的JSON结构差异
    method1_sample = pred1[0] if pred1 else {}
    method2_sample = pred2[0] if pred2 else {}
    
    print(f"方法1 JSON结构: {list(method1_sample.keys())}")
    print(f"方法2 JSON结构: {list(method2_sample.keys())}")
    print(f"方法2特有字段: {set(method2_sample.keys()) - set(method1_sample.keys())}")
    
    # 分析预测格式差异
    print("\n=== 预测内容格式分析 ===")
    if pred1 and "prediction" in pred1[0]:
        print(f"方法1预测格式: {type(pred1[0]['prediction'])}")
        if isinstance(pred1[0]['prediction'], str):
            print(f"方法1预测样例: {pred1[0]['prediction'][:200]}...")
        elif isinstance(pred1[0]['prediction'], list):
            print(f"方法1预测样例: {pred1[0]['prediction'][0][:200]}...")
            if len(pred1[0]['prediction']) > 1:
                print(f"方法1有多个预测答案，数量: {len(pred1[0]['prediction'])}")
    
    if pred2 and "prediction" in pred2[0]:
        print(f"方法2预测格式: {type(pred2[0]['prediction'])}")
        print(f"方法2预测样例: {pred2[0]['prediction'][:200]}...")
    
    # 核心分析：方法1正确而方法2错误的案例
    print("\n=== 方法1正确而方法2错误的案例分析 ===")
    
    # 统计变量
    total_failures = 0
    failure_categories = defaultdict(int)
    failure_examples = defaultdict(list)
    
    # 分析策略
    for id in common_ids:
        item1 = pred1_dict[id]
        item2 = pred2_dict[id]
        
        # 获取ground truth
        gt = set(item1.get("ground_truth", []))
        if not gt and "ground_truth" in item1:
            if isinstance(item1["ground_truth"], str):
                gt = {item1["ground_truth"]}
        
        # 提取方法1的答案
        pred1_answers = []
        pred1_text = item1.get("prediction", "")
        
        if isinstance(pred1_text, list):
            # 从每个预测中提取Answer部分
            for p in pred1_text:
                if "# Answer:" in p:
                    answer = p.split("# Answer:")[-1].strip()
                    pred1_answers.append(answer)
            
            if pred1_answers:
                pred1_text = ", ".join(pred1_answers)
        
        # 获取方法2的预测
        pred2_text = item2.get("prediction", "")
        
        # 判断正确性
        m1_correct = any(g.lower() in pred1_text.lower() for g in gt) if gt else False
        m2_correct = any(g.lower() in pred2_text.lower() for g in gt) if gt else False
        
        # 分析方法1正确但方法2错误的情况
        if m1_correct and not m2_correct:
            total_failures += 1
            
            # 分析失败原因
            failure_reason = "未知原因"
            
            # 检查方法2是否输出了m.编码
            if "m." in pred2_text:
                m_codes = re.findall(r'm\.[0-9a-z_]+', pred2_text)
                failure_reason = "m.编码问题"
                # 看方法1是否有m.编码
                has_m_code_in_pred1 = any("m." in p for p in pred1_answers) if pred1_answers else "m." in pred1_text
                
                if not has_m_code_in_pred1:
                    failure_reason = "m.编码问题(仅方法2)"
            
            # 检查方法2是否输出空或无效内容
            elif not pred2_text.strip():
                failure_reason = "空预测"
            
            # 检查是否遇到格式错误
            elif "exploration_history" in item2 and not item2.get("answer_found_during_exploration", True):
                failure_reason = "探索未找到答案"
            
            # 统计失败原因
            failure_categories[failure_reason] += 1
            
            # 保存样例(每种类型最多5个)
            if len(failure_examples[failure_reason]) < 5:
                # 尝试查找方法2探索历史中的问题
                exploration_issue = ""
                if "exploration_history" in item2:
                    # 分析探索历史中是否有m.编码
                    has_m_code_in_exploration = False
                    for round_data in item2.get("exploration_history", []):
                        for expansion in round_data.get("expansions", []):
                            for relation in expansion.get("relations", []):
                                if any("m." in str(target) for target in relation.get("targets", [])):
                                    has_m_code_in_exploration = True
                                    break
                    
                    if has_m_code_in_exploration:
                        exploration_issue = "探索历史包含m.编码"
                
                failure_examples[failure_reason].append((id, gt, pred1_text, pred2_text, exploration_issue))
    
    # 输出统计结果
    print(f"方法1正确但方法2错误的总案例数: {total_failures}")
    print("\n主要失败原因统计:")
    for reason, count in sorted(failure_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_failures) * 100 if total_failures > 0 else 0
        print(f"- {reason}: {count} ({percentage:.2f}%)")
    
    # 输出具体样例
    print("\n=== 典型失败案例 ===")
    for reason, examples in failure_examples.items():
        print(f"\n【失败原因: {reason}】")
        for i, (id, gt, pred1, pred2, exploration_issue) in enumerate(examples, 1):
            print(f"案例 {i}:")
            print(f"ID: {id}")
            print(f"正确答案: {gt}")
            print(f"方法1预测: {pred1}")
            print(f"方法2预测: {pred2}")
            if exploration_issue:
                print(f"探索历史问题: {exploration_issue}")
            print()
    
    # 提供针对性解决方案
    print("\n=== 解决方案 ===")
    if failure_categories.get("m.编码问题", 0) + failure_categories.get("m.编码问题(仅方法2)", 0) > 0:
        print("1. 添加实体解析: 在方法2中添加一个后处理步骤，将m.编码转换为自然语言实体名称")
        print("2. 改进提示模板: 明确指示模型返回自然语言答案而非内部编码")
        print("3. 探索过程优化: 在知识图谱探索过程中就将m.编码转换为实体名称")
    
    if failure_categories.get("探索未找到答案", 0) > 0:
        print("4. 增加探索轮数: 方法2可能需要更多轮次的探索才能找到正确答案")
        print("5. 优化关系选择策略: 改进LLM选择关系的方式，确保选择更相关的路径")
    
    if failure_categories.get("空预测", 0) > 0:
        print("6. 改进答案生成逻辑: 确保方法2在探索过程中能够提取出有效答案")
        print("7. 添加备选答案机制: 当无法确定答案时，提供备选项而非空返回")

def analyze_ground_truth_and_evaluation_logic(dir1: str, dir2: str) -> None:
    """验证ground_truth中m.编码的比例和评分逻辑"""
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    det1_path = os.path.join(dir1, "detailed_eval_result.jsonl")
    
    if not os.path.exists(pred1_path) or not os.path.exists(det1_path):
        print("预测文件或评估结果文件不存在")
        return
    
    pred1 = load_jsonl(pred1_path)
    det1 = load_jsonl(det1_path)
    
    # 转换为ID索引的字典
    pred1_dict = {item["id"]: item for item in pred1}
    det1_dict = {item["id"]: item for item in det1}
    
    print("\n=== Ground Truth分析 ===")
    
    # 统计变量
    total_questions = 0
    questions_with_m_code_gt = 0
    m_code_gt_examples = []
    
    # 分析ground_truth中的m.编码
    for item in pred1:
        if "ground_truth" not in item:
            continue
            
        total_questions += 1
        gt = item["ground_truth"]
        
        # 检查ground_truth中是否有m.编码
        has_m_code = False
        m_codes = []
        
        if isinstance(gt, str):
            if "m." in gt:
                has_m_code = True
                m_codes = re.findall(r'm\.[0-9a-z_]+', gt)
        elif isinstance(gt, list):
            for g in gt:
                if "m." in str(g):
                    has_m_code = True
                    m_codes.extend(re.findall(r'm\.[0-9a-z_]+', str(g)))
        
        if has_m_code:
            questions_with_m_code_gt += 1
            if len(m_code_gt_examples) < 5:
                m_code_gt_examples.append((item["id"], gt, m_codes))
    
    # 计算比例
    m_code_percentage = (questions_with_m_code_gt / total_questions) * 100 if total_questions > 0 else 0
    print(f"总问题数: {total_questions}")
    print(f"Ground Truth中包含m.编码的问题数: {questions_with_m_code_gt} ({m_code_percentage:.2f}%)")
    
    # 输出样例
    print("\n包含m.编码的Ground Truth样例:")
    for i, (id, gt, m_codes) in enumerate(m_code_gt_examples, 1):
        print(f"样例 {i}:")
        print(f"ID: {id}")
        print(f"Ground Truth: {gt}")
        print(f"m.编码: {m_codes}")
        print()
    
    # 分析评分逻辑
    print("\n=== 评分逻辑分析 ===")
    
    # 找出具有多个答案的ground truth
    multi_answer_examples = []
    partial_match_correct = 0
    partial_match_incorrect = 0
    
    for id, item in pred1_dict.items():
        if id not in det1_dict:
            continue
            
        gt = item.get("ground_truth", [])
        if not isinstance(gt, list) or len(gt) <= 1:
            continue
            
        # 多答案ground truth
        det_item = det1_dict[id]
        pred = item.get("prediction", "")
        
        # 检查预测是否部分匹配ground truth
        any_match = False
        all_match = True
        matches = []
        
        for g in gt:
            # 规范化比较
            g_lower = str(g).lower()
            pred_lower = str(pred).lower()
            is_match = g_lower in pred_lower
            
            if is_match:
                any_match = True
                matches.append(g)
            else:
                all_match = False
        
        # 如果部分匹配但不是全部匹配
        if any_match and not all_match:
            # 检查评估结果
            is_correct = False
            if "ans_acc" in det_item:
                is_correct = det_item["ans_acc"] > 0
            elif "ans_f1" in det_item:
                is_correct = det_item["ans_f1"] > 0
            
            if is_correct:
                partial_match_correct += 1
            else:
                partial_match_incorrect += 1
                
            if len(multi_answer_examples) < 5:
                multi_answer_examples.append((id, gt, pred, matches, is_correct))
    
    # 输出部分匹配的评分情况
    total_partial = partial_match_correct + partial_match_incorrect
    print(f"多答案且部分匹配的问题数: {total_partial}")
    
    if total_partial > 0:
        correct_percentage = (partial_match_correct / total_partial) * 100
        print(f"部分匹配被评为正确的数量: {partial_match_correct} ({correct_percentage:.2f}%)")
        print(f"部分匹配被评为错误的数量: {partial_match_incorrect} ({100-correct_percentage:.2f}%)")
        
        if partial_match_correct > partial_match_incorrect:
            print("\n结论: 评分逻辑接受部分匹配作为正确答案")
        else:
            print("\n结论: 评分逻辑要求完全匹配所有答案")
    
    # 输出部分匹配样例
    print("\n部分匹配样例:")
    for i, (id, gt, pred, matches, is_correct) in enumerate(multi_answer_examples, 1):
        print(f"样例 {i}:")
        print(f"ID: {id}")
        print(f"Ground Truth: {gt}")
        print(f"预测: {pred[:200]}..." if len(str(pred)) > 200 else f"预测: {pred}")
        print(f"成功匹配: {matches}")
        print(f"评分结果: {'正确' if is_correct else '错误'}")
        print()

def analyze_method2_perfect_method1_failure(dir1: str, dir2: str) -> None:
    """找出方法2完美(F1=1, Acc=1)而方法1完全失败(F1=0, Acc=0)的案例"""
    pred1_path = os.path.join(dir1, "predictions.jsonl")
    eval1_path = os.path.join(dir1, "detailed_eval_result.jsonl")
    pred2_path = os.path.join(dir2, "predictions.jsonl")
    eval2_path = os.path.join(dir2, "detailed_eval_result.jsonl")

    # Check if all required files exist
    required_files = [pred1_path, eval1_path, pred2_path, eval2_path]
    if not all(os.path.exists(p) for p in required_files):
        print("错误: 缺少必要的预测或评估文件。")
        missing = [p for p in required_files if not os.path.exists(p)]
        print(f"缺失的文件: {', '.join(missing)}")
        return

    # Load data
    pred1 = load_jsonl(pred1_path)
    eval1 = load_jsonl(eval1_path)
    pred2 = load_jsonl(pred2_path)
    eval2 = load_jsonl(eval2_path)

    # Create dictionaries indexed by ID
    pred1_dict = {item["id"]: item for item in pred1}
    eval1_dict = {item["id"]: item for item in eval1}
    pred2_dict = {item["id"]: item for item in pred2}
    eval2_dict = {item["id"]: item for item in eval2}

    # Find common IDs across all files
    common_ids = set(pred1_dict.keys()) & set(eval1_dict.keys()) & set(pred2_dict.keys()) & set(eval2_dict.keys())
    
    if not common_ids:
        print("在所有文件中找不到共同的问题ID。")
        return

    print("\n=== 方法2完美 & 方法1失败 的案例分析 ===")
    
    found_cases = []

    for id in common_ids:
        item1_pred = pred1_dict[id]
        item1_eval = eval1_dict[id]
        item2_pred = pred2_dict[id]
        item2_eval = eval2_dict[id]

        # Check metrics - Assuming 'ans_f1' and 'ans_acc' exist and are floats
        # Method 2 Perfect: F1 = 1.0 AND Acc = 1.0
        m2_perfect = item2_eval.get('ans_f1', 0.0) == 1.0 and item2_eval.get('ans_acc', 0.0) == 1.0
        
        # Method 1 Failure: F1 = 0.0 AND Acc = 0.0
        m1_failure = item1_eval.get('ans_f1', 1.0) == 0.0 and item1_eval.get('ans_acc', 1.0) == 0.0

        if m2_perfect and m1_failure:
            # Extract relevant info, handling potential missing keys
            question = item1_pred.get("question", item2_pred.get("question", "N/A"))
            gt = item1_pred.get("ground_truth", item2_pred.get("ground_truth", "N/A"))
            pred1_text = item1_pred.get("prediction", "N/A")
            pred2_text = item2_pred.get("prediction", "N/A")

            # Clean up method 1's prediction if it's a list of strings containing '# Answer:'
            if isinstance(pred1_text, list):
                answers = []
                for p in pred1_text:
                    if isinstance(p, str) and "# Answer:" in p:
                         answers.append(p.split("# Answer:")[-1].strip())
                pred1_text = ", ".join(answers) if answers else str(pred1_text) # Fallback to string representation
            
            found_cases.append({
                "id": id,
                "question": question,
                "ground_truth": gt,
                "method1_prediction": pred1_text,
                "method2_prediction": pred2_text
            })

    # Output results
    if found_cases:
        print(f"找到 {len(found_cases)} 个方法2完美而方法1失败的案例:")
        for i, case in enumerate(found_cases, 1):
            print(f"\n案例 {i}:")
            print(f"  ID: {case['id']}")
            print(f"  Question: {case['question']}")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  方法1 预测 (失败): {case['method1_prediction']}")
            print(f"  方法2 预测 (完美): {case['method2_prediction']}")
    else:
        print("未找到满足条件 (方法2完美 & 方法1失败) 的案例。")

if __name__ == "__main__":
    print(f"分析目录1: {dir1}")
    print(f"分析目录2: {dir2}\n")
    analyze_method_differences(dir1, dir2)
    # analyze_ground_truth_and_evaluation_logic(dir1, dir2)
    analyze_method2_perfect_method1_failure(dir1, dir2)