from collections import OrderedDict
import json
import re
import string
# from sklearn.metrics import precision_score # 这行看起来没有被使用，可以移除或保留
from statistics import mean # 在 eval_joint_result 中使用
import os

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction: str, answer: list) -> float: # 明确 prediction 类型为 str
    matched = 0.0
    if not answer: # 处理 answer 为空列表的情况
        return 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction: str, answer: list) -> int: # 明确 prediction 类型为 str
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction: list, answer: list) -> tuple: # 明确 prediction 类型为 list
    if len(prediction) == 0 or len(answer) == 0:
        return 0.0, 0.0, 0.0 # 返回浮点数以保持一致性
    ans_recalled = 0
    prediction_correct = 0
    # 对于 recall, 将所有预测合并为一个字符串进行匹配
    prediction_str = " ".join(str(p) for p in prediction) # 确保所有元素为字符串
    for a in answer:
        if match(prediction_str, a):
            ans_recalled += 1
    recall = ans_recalled / len(answer)
    
    # 对于 precision, 检查每个单独的预测
    for p_item in prediction:
        # p_item 应该是字符串，如果 prediction 列表中的元素不是字符串，需要处理
        # 但通常 extract_topk_prediction 返回的是字符串列表
        current_p_str = str(p_item) # 确保是字符串
        for a in answer:
            if match(current_p_str, a): # p_item 是单个预测字符串
                prediction_correct += 1
                break # 每个预测只算一次正确
    precision = prediction_correct / len(prediction)
    
    if precision + recall == 0:
        return 0.0, precision, recall
    else:
        return (2 * precision * recall) / (precision + recall), precision, recall


def extract_topk_prediction(prediction, k=-1):
    if isinstance(prediction, str):
        # 如果原始预测是单个字符串（可能包含逗号分隔的答案）
        # 或者本身就是一个答案，这里按逗号分割可能不总是适用
        # 需要根据 data["prediction"] 的实际格式来决定这里的处理
        # 假设如果它是字符串，它就是单个答案，或者调用者已确保其格式正确
        # 如果它本身就是单个答案字符串，则下面的 split 和 frequency count 可能不适用
        # 为了安全，如果它是单个答案字符串，我们直接返回它（作为列表）
        if "," not in prediction and k == 1 or k == -1 : # 假设单个字符串是单个答案
             return [prediction.strip()] if prediction.strip() else []
        prediction = prediction.split(",")

    results = {}
    for p in prediction: # prediction 现在应该是一个列表
        p_stripped = p.strip()
        if p_stripped == "":
            continue
        if p_stripped in results:
            results[p_stripped] += 1
        else:
            results[p_stripped] = 1
    
    if not results: # 如果没有有效的预测
        return []

    if k > len(results) or k < 0:
        k = len(results)
    
    # sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    # 如果只是为了提取topk而不关心原始频率（因为后续eval_f1等不在乎频率），可以简化
    # 但如果原始prediction列表中的顺序或频率有意义（例如来自模型的多个输出），则保留
    # 当前代码是基于频率排序，这对于“去重并取最常见”的top-k是有意义的
    # 但如果prediction已经是经过排序的列表，则直接取[:k]即可
    # 鉴于后续代码并未说明prediction的来源细节，保留现有频率统计逻辑
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in sorted_results[:k]]

def eval_rank_results(predict_file, topk=[1, 3, 5, 10]):
    eval_name = (
        f"detailed_eval_result_top_{topk}.jsonl"
        if topk # 修正: topk 本身是列表，直接判断其真值性
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    all_acc_list = OrderedDict({k_val: [] for k_val in topk})
    all_hit_list = OrderedDict({k_val: [] for k_val in topk})
    all_f1_list = OrderedDict({k_val: [] for k_val in topk})
    all_precision_list = OrderedDict({k_val: [] for k_val in topk}) # 修正拼写
    all_recall_list = OrderedDict({k_val: [] for k_val in topk})
    
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError: # 更具体的异常
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue
            
            id_val = data.get("id") # 使用 .get() 避免 KeyError
            answer = list(set(data.get("answer", []))) # 使用 .get() 并提供默认空列表
            
            if not answer: # 如果没有标准答案，跳过此样本的评估
                print(f"Skipping sample {id_val} due to empty ground truth answers.")
                continue

            acc_list_sample = OrderedDict() # 避免与外部 acc_list 混淆
            hit_list_sample = OrderedDict()
            f1_list_sample = OrderedDict()
            precision_list_sample = OrderedDict() # 修正拼写
            recall_list_sample = OrderedDict()
            
            ranks = data.get('ranks', []) # 使用 .get()
            for k_val in topk:
                top_k_pred_count = min(k_val, len(ranks))
                topk_rank_responses = [r.get('response', "") for r in ranks[:top_k_pred_count]] # 使用 .get()
                
                # extract_topk_prediction 可能不需要在这里调用，因为 ranks 已经是排序的
                # 但如果 response 内部可能包含逗号分隔的多个答案，则需要
                # 假设每个 r['response'] 是一个独立的预测项
                current_predictions = [str(p).strip() for p in topk_rank_responses if str(p).strip()]


                f1_score, precision_score, recall_score = eval_f1(current_predictions, answer)
                
                # prediction_str 用于 eval_acc 和 eval_hit
                prediction_str = " ".join(current_predictions)
                
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                
                acc_list_sample[k_val] = acc
                hit_list_sample[k_val] = hit
                f1_list_sample[k_val]= f1_score
                precision_list_sample[k_val] = precision_score # 修正拼写
                recall_list_sample[k_val] = recall_score
            
            f2.write(
                json.dumps(
                    {
                        "id": id_val,
                        # "prediction" 在这里指的是针对最大k值的current_predictions，或者不写入也可
                        # "ground_truth": answer, # ground_truth 是 answer
                        "answer": answer, # 更符合字段名
                        "acc@k": acc_list_sample,
                        "hit@k": hit_list_sample,
                        "f1@k": f1_list_sample,
                        "precision@k": precision_list_sample, # 修正拼写
                        "recall@k": recall_list_sample,
                    }
                )
                + "\n"
            )
            for k_val in topk:
                all_acc_list[k_val].append(acc_list_sample[k_val])
                all_hit_list[k_val].append(hit_list_sample[k_val])
                all_f1_list[k_val].append(f1_list_sample[k_val])
                all_precision_list[k_val].append(precision_list_sample[k_val]) # 修正拼写
                all_recall_list[k_val].append(recall_list_sample[k_val])
                
    result_str = ""
    for k_val in topk:
        num_samples = len(all_acc_list[k_val])
        if num_samples == 0:
            result_str += f"Top-{k_val}: No valid samples found for evaluation.\n"
            continue
            
        avg_acc = sum(all_acc_list[k_val]) * 100 / num_samples
        avg_hit = sum(all_hit_list[k_val]) * 100 / num_samples
        avg_f1 = sum(all_f1_list[k_val]) * 100 / num_samples
        avg_precision = sum(all_precision_list[k_val]) * 100 / num_samples # 修正拼写
        avg_recall = sum(all_recall_list[k_val]) * 100 / num_samples
        
        result_str += f"Top-{k_val}:\n"
        result_str += (
            f"Accuracy: {avg_acc:.2f}% " # 添加格式化
            f"Hit: {avg_hit:.2f}% "
            f"F1: {avg_f1:.2f}% "
            f"Precision: {avg_precision:.2f}% " # 修正拼写
            f"Recall: {avg_recall:.2f}%\n"
        )
        
    print(result_str)
    # 修正 result_name 中 topk 的使用，它是一个列表
    topk_str = "_".join(map(str, topk)) if topk else ""
    result_name = f"eval_result_top_{topk_str}.txt" if topk_str else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)

def eval_result(predict_file, cal_f1=True, topk=-1):
    eval_name = (
        f"detailed_eval_result_top_{topk}.jsonl" # topk在这里是整数
        if topk > 0
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = [] # 修正拼写
    recall_list = []
    
    count_valid_samples = 0

    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue
            
            id_val = data.get("id")
            raw_prediction = data.get("prediction") # 可以是字符串或列表
            answer = list(set(data.get("ground_truth", [])))

            if answer is None or len(answer) == 0 : # 确保 answer 有效
                 print(f"Skipping sample {id_val} due to empty or missing ground_truth.")
                 continue
            if raw_prediction is None: # 如果 prediction 不存在，跳过
                 print(f"Skipping sample {id_val} due to missing prediction.")
                 continue


            current_prediction_list = []
            prediction_for_acc_hit_str = ""

            if cal_f1:
                # extract_topk_prediction 期望一个列表或逗号分隔的字符串
                # 确保 raw_prediction 传递给它的格式正确
                current_prediction_list = extract_topk_prediction(raw_prediction, topk)
                
                f1_score, local_precision_score, local_recall_score = eval_f1(current_prediction_list, answer)
                f1_list.append(f1_score)
                precision_list.append(local_precision_score) # 修正拼写
                recall_list.append(local_recall_score)
                
                # prediction_str 用于 eval_acc 和 eval_hit
                prediction_for_acc_hit_str = " ".join(str(p) for p in current_prediction_list)
                
            else: # not cal_f1
                # 如果 raw_prediction 是列表，合并它；如果是字符串，直接使用
                if isinstance(raw_prediction, list):
                    prediction_for_acc_hit_str = " ".join(str(p) for p in raw_prediction)
                    current_prediction_list = [str(p) for p in raw_prediction] # 用于写入JSON
                elif isinstance(raw_prediction, str):
                    prediction_for_acc_hit_str = raw_prediction
                    # 尝试按常见分隔符分割，如果它是代表多个答案的字符串
                    # 但为了简单，这里我们假设如果不是列表，它就是单个答案字符串
                    # 或者 extract_topk_prediction 应该在 cal_f1=False 时也适用
                    # 为了保持一致性，eval_acc/eval_hit 的输入应该与 cal_f1=True 时类似
                    # 但原始逻辑是直接用 raw_prediction, 现在我们确保它是字符串
                    current_prediction_list = [raw_prediction] if raw_prediction else []

            acc = eval_acc(prediction_for_acc_hit_str, answer)
            hit = eval_hit(prediction_for_acc_hit_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)
            
            count_valid_samples += 1
            
            # 写入 JSON 的内容
            output_data = {
                "id": id_val,
                "prediction": current_prediction_list, # 写入处理后的预测列表
                "ground_truth": answer,
                "acc": acc,
                "hit": hit,
            }
            if cal_f1:
                output_data["f1"] = f1_score
                output_data["precision"] = local_precision_score # 修正拼写
                output_data["recall"] = local_recall_score
            
            f2.write(json.dumps(output_data) + "\n")

    if count_valid_samples == 0:
        print("No valid samples found for evaluation.")
        result_str = "Accuracy: 0.00% Hit: 0.00%"
        if cal_f1:
            result_str += " F1: 0.00% Precision: 0.00% Recall: 0.00%"
    elif cal_f1 and len(f1_list) > 0: # 确保 f1_list 不为空
        avg_acc = sum(acc_list) * 100 / count_valid_samples
        avg_hit = sum(hit_list) * 100 / count_valid_samples
        avg_f1 = sum(f1_list) * 100 / count_valid_samples
        avg_precision = sum(precision_list) * 100 / count_valid_samples # 修正拼写
        avg_recall = sum(recall_list) * 100 / count_valid_samples
        result_str = (
            f"Accuracy: {avg_acc:.2f}% "
            f"Hit: {avg_hit:.2f}% "
            f"F1: {avg_f1:.2f}% "
            f"Precision: {avg_precision:.2f}% " # 修正拼写
            f"Recall: {avg_recall:.2f}%"
        )
    else: # not cal_f1 or f1_list is empty but acc_list might not be
        avg_acc = sum(acc_list) * 100 / count_valid_samples
        avg_hit = sum(hit_list) * 100 / count_valid_samples
        result_str = (
            f"Accuracy: {avg_acc:.2f}% "
            f"Hit: {avg_hit:.2f}%"
        )
        
    print(result_str)
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)

def eval_joint_result(predict_file):
    eval_name = "detailed_eval_result.jsonl"
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = [] # 修正拼写
    recall_list = []
    path_f1_list = []
    path_precision_list = [] # 修正拼写
    path_recall_list = []
    path_ans_f1_list = []
    path_ans_recall_list = []
    path_ans_precision_list = [] # 修正拼写

    count_valid_samples = 0

    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue
            
            id_val = data.get("id")
            raw_predictions_list = data.get("prediction", []) # 原始预测列表
            answer = list(set(data.get("ground_truth", [])))
            ground_truth_paths = data.get("ground_truth_paths", [])


            if not answer:
                print(f"Skipping sample {id_val} due to empty ground_truth answers.")
                continue
            # if not ground_truth_paths: # Path evaluation might be optional
            #     print(f"Skipping path evaluation for {id_val} due to empty ground_truth_paths.")


            predicted_reasoning_paths = set()
            predicted_answers_set = set() # 使用 set 去重

            for pre_item_str in raw_predictions_list:
                try:
                    ans_in_pred = False
                    # 确保 pre_item_str 是字符串
                    if not isinstance(pre_item_str, str):
                        pre_item_str = str(pre_item_str)

                    if "the answer is: " in pre_item_str:
                        ans_in_pred = True
                        ans_pred_segment = pre_item_str.split("the answer is: ", 1)[1] # 分割一次
                        for ans_line in ans_pred_segment.split("\n"): # 答案可能多行
                            if ans_line.strip(): # 确保不是空行
                                predicted_answers_set.add(ans_line.strip())
                    
                    if "Reasoning path:\n" in pre_item_str:
                        path_segment_split = pre_item_str.split("Reasoning path:\n", 1)
                        if len(path_segment_split) > 1:
                            path_pred_segment = path_segment_split[1]
                            if ans_in_pred and "\nthe answer is: " in path_pred_segment:
                                path_pred_segment = path_pred_segment.split("\nthe answer is: ",1)[0]
                            
                            # 路径通常是多行的，最后一个元素可能是空字符串或答案的一部分
                            # splitlines() 比 split('\n') 更安全
                            paths_from_segment = path_pred_segment.splitlines()
                            for path_line in paths_from_segment:
                                stripped_path = path_line.strip()
                                # 过滤掉可能是答案残留的行，或空行
                                if stripped_path and "the answer is:" not in stripped_path.lower():
                                     predicted_reasoning_paths.add(stripped_path)
                except Exception as e:
                    print(f"Error processing prediction item for ID {id_val}: '{pre_item_str[:100]}...'")
                    print(e)
                    continue
            
            final_predicted_answers_list = sorted(list(predicted_answers_set)) # 转为列表并排序，便于复现
            final_predicted_paths_list = sorted(list(predicted_reasoning_paths))

            if not final_predicted_answers_list and not answer: # 如果没预测也没答案，跳过
                 pass # 或者直接continue，但上面已经有 if not answer: continue
            elif not final_predicted_answers_list and answer : # 有答案但没预测出来
                f1_score, local_precision_score, local_recall_score = 0.0,0.0,0.0
            else:
                f1_score, local_precision_score, local_recall_score = eval_f1(final_predicted_answers_list, answer)

            f1_list.append(f1_score)
            precision_list.append(local_precision_score) # 修正拼写
            recall_list.append(local_recall_score)
            
            prediction_str_for_acc_hit = " ".join(final_predicted_answers_list)
            acc = eval_acc(prediction_str_for_acc_hit, answer)
            hit = eval_hit(prediction_str_for_acc_hit, answer)
            acc_list.append(acc)
            hit_list.append(hit)
            
            # Path F1 (Path vs Ground Truth Paths)
            if ground_truth_paths: # 只在有GT路径时评估路径
                path_f1_score, path_local_precision_score, path_local_recall_score = eval_f1(
                    final_predicted_paths_list, ground_truth_paths
                )
            else: # 没有GT路径，则路径指标为0
                path_f1_score, path_local_precision_score, path_local_recall_score = 0.0, 0.0, 0.0

            path_f1_list.append(path_f1_score)
            path_precision_list.append(path_local_precision_score) # 修正拼写
            path_recall_list.append(path_local_recall_score)
            
            # Path Ans F1 (Path vs Ground Truth Answers) - 评估路径是否包含答案
            path_ans_f1_score, path_ans_local_precision_score, path_ans_local_recall_score = eval_f1(final_predicted_paths_list, answer)
            path_ans_f1_list.append(path_ans_f1_score)
            path_ans_precision_list.append(path_ans_local_precision_score) # 修正拼写
            path_ans_recall_list.append(path_ans_local_recall_score)

            count_valid_samples +=1
            
            f2.write(
                json.dumps(
                    {
                        "id": id_val,
                        "prediction_raw": raw_predictions_list, # 记录原始的，因为 "prediction" 通常指处理后的
                        "predicted_answers_extracted": final_predicted_answers_list,
                        "predicted_paths_extracted": final_predicted_paths_list,
                        "ground_truth_answers": answer,
                        "ground_truth_paths": ground_truth_paths,
                        "ans_acc": acc,
                        "ans_hit": hit,
                        "ans_f1": f1_score,
                        "ans_precision": local_precision_score, # 修正拼写
                        "ans_recall": local_recall_score,
                        "path_f1": path_f1_score,
                        "path_precision": path_local_precision_score, # 修正拼写
                        "path_recall": path_local_recall_score,
                        "path_ans_f1": path_ans_f1_score,
                        "path_ans_precision": path_ans_local_precision_score, # 修正拼写
                        "path_ans_recall": path_ans_local_recall_score
                    }
                )
                + "\n"
            )

    if count_valid_samples == 0:
        print("No valid samples processed for joint results.")
        result_str = "Accuracy: 0.00% Hit: 0.00% F1: 0.00% Precision: 0.00% Recall: 0.00% Path F1: 0.00% Path Precision: 0.00% Path Recall: 0.00% Path Ans F1: 0.00 Path Ans Precision: 0.00 Path Ans recall: 0.00"
    else:
        avg_acc = sum(acc_list) * 100 / count_valid_samples
        avg_hit = sum(hit_list) * 100 / count_valid_samples
        avg_f1 = sum(f1_list) * 100 / count_valid_samples
        avg_precision = sum(precision_list) * 100 / count_valid_samples # 修正拼写
        avg_recall = sum(recall_list) * 100 / count_valid_samples
        avg_path_f1 = sum(path_f1_list) * 100 / count_valid_samples
        avg_path_precision = sum(path_precision_list) * 100 / count_valid_samples # 修正拼写
        avg_path_recall = sum(path_recall_list) * 100 / count_valid_samples
        # mean from statistics.mean for path_ans metrics
        avg_path_ans_f1 = mean(path_ans_f1_list) * 100 if path_ans_f1_list else 0.0
        avg_path_ans_precision = mean(path_ans_precision_list) * 100 if path_ans_precision_list else 0.0 # 修正拼写
        avg_path_ans_recall = mean(path_ans_recall_list) * 100 if path_ans_recall_list else 0.0

        result_str = (
            f"Accuracy: {avg_acc:.2f}% Hit: {avg_hit:.2f}% F1: {avg_f1:.2f}% Precision: {avg_precision:.2f}% Recall: {avg_recall:.2f}% "
            f"Path F1: {avg_path_f1:.2f}% Path Precision: {avg_path_precision:.2f}% Path Recall: {avg_path_recall:.2f}% "
            f"Path Ans F1: {avg_path_ans_f1:.2f}% Path Ans Precision: {avg_path_ans_precision:.2f}% Path Ans recall: {avg_path_ans_recall:.2f}%" # 修正 recall 拼写
        )
        
    print(result_str)
    result_name = "eval_result.txt" # eval_joint_result 通常不带 topk
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)


def eval_path_result(predict_file, cal_f1=True, topk=-1):
    eval_name = (
        f"detailed_eval_result_top_{topk}.jsonl"
        if topk > 0
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    
    # Initialize counters
    sum_acc, sum_hit, sum_f1, sum_precision, sum_recall = 0.0, 0.0, 0.0, 0.0, 0.0 # 使用浮点数
    sum_path_f1, sum_path_precision, sum_path_recall = 0.0, 0.0, 0.0
    count = 0

    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue
            
            id_val = data.get("id")
            # data.get("prediction", []) 确保如果 "prediction" 不存在，我们得到一个空列表
            raw_prediction_list = data.get("prediction", [])
            # 移除 start_entities (如果存在)
            start_entities = set(data.get("start_entities", []))
            # current_prediction_list 将是用于评估的预测列表
            current_prediction_list = [p for p in raw_prediction_list if p not in start_entities]
            
            answer = list(set(data.get("ground_truth", [])))
            ground_truth_paths = data.get("ground_truth_paths", []) # 确保存在

            # 根据您的原始逻辑，如果这些关键数据不完整，则跳过
            # 修改：如果 ground_truth_paths 为空，可能只想跳过路径评估，而不是整个样本
            if not answer: # 必须有答案才能评估
                 print(f"Skipping sample {id_val} due to empty ground_truth answers.")
                 continue
            # 如果 current_prediction_list 为空，则 acc, hit, f1, precision, recall 都会是0
            # 这通常是可接受的，表示模型没有做出有效预测

            prediction_for_acc_hit_str = ""
            acc, hit = 0.0, 0 # 默认值

            if cal_f1:
                # extract_topk_prediction 现在处理的是 current_prediction_list
                processed_prediction_list_for_f1 = extract_topk_prediction(current_prediction_list, topk)
                
                f1_score, local_precision_score, local_recall_score = eval_f1(processed_prediction_list_for_f1, answer)
                
                # Path F1 (Processed Predictions vs Ground Truth Paths)
                # 只有在 ground_truth_paths 存在时才进行评估
                if ground_truth_paths:
                    path_f1_score, path_local_precision_score, path_local_recall_score = eval_f1(
                        processed_prediction_list_for_f1, ground_truth_paths
                    )
                else:
                    path_f1_score, path_local_precision_score, path_local_recall_score = 0.0, 0.0, 0.0

                # Compute accuracy and hit using the same processed list for consistency
                prediction_for_acc_hit_str = " ".join(str(p) for p in processed_prediction_list_for_f1)
                acc = eval_acc(prediction_for_acc_hit_str, answer)
                hit = eval_hit(prediction_for_acc_hit_str, answer)

                # Update sums for F1 related metrics
                sum_f1 += f1_score
                sum_precision += local_precision_score # 修正拼写
                sum_recall += local_recall_score
                sum_path_f1 += path_f1_score
                sum_path_precision += path_local_precision_score # 修正拼写
                sum_path_recall += path_local_recall_score
            else: # not cal_f1
                # current_prediction_list 是原始（已过滤start_entities）的预测列表
                prediction_for_acc_hit_str = " ".join(str(p) for p in current_prediction_list)
                acc = eval_acc(prediction_for_acc_hit_str, answer)
                hit = eval_hit(prediction_for_acc_hit_str, answer)

            # Update sums for acc and hit
            sum_acc += acc
            sum_hit += hit
            count += 1

            # Write detailed results
            output_data_detailed = {
                "id": id_val,
                "prediction_processed": processed_prediction_list_for_f1 if cal_f1 else current_prediction_list,
                "ground_truth": answer,
                "ans_acc": acc, # 使用 acc
                "ans_hit": hit, # 使用 hit
            }
            if cal_f1:
                output_data_detailed["ans_f1"] = f1_score
                output_data_detailed["ans_precision"] = local_precision_score # 修正拼写
                output_data_detailed["ans_recall"] = local_recall_score
                output_data_detailed["path_f1"] = path_f1_score
                output_data_detailed["path_precision"] = path_local_precision_score # 修正拼写
                output_data_detailed["path_recall"] = path_local_recall_score
            
            f2.write(json.dumps(output_data_detailed) + "\n")

    # Compute and print summary
    if count == 0:
        print("No valid samples found for path results evaluation.")
        # 根据 cal_f1 决定输出哪些指标
        result_str = "Accuracy: 0.00% Hit: 0.00%"
        if cal_f1:
            result_str += " F1: 0.00% Precision: 0.00% Recall: 0.00% Path F1: 0.00% Path Precision: 0.00% Path Recall: 0.00%"
    else:
        avg_mult = 100.0 / count
        result_str = (
            f"Accuracy: {sum_acc * avg_mult:.2f}% "
            f"Hit: {sum_hit * avg_mult:.2f}% "
        )
        if cal_f1:
            result_str += (
                f"F1: {sum_f1 * avg_mult:.2f}% "
                f"Precision: {sum_precision * avg_mult:.2f}% " # 修正拼写
                f"Recall: {sum_recall * avg_mult:.2f}% "
                f"Path F1: {sum_path_f1 * avg_mult:.2f}% "
                f"Path Precision: {sum_path_precision * avg_mult:.2f}% " # 修正拼写
                f"Path Recall: {sum_path_recall * avg_mult:.2f}%"
            )

    print(result_str)
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)


def eval_path_result_w_ans(predict_file, cal_f1=True, topk=-1, output_path=None):
    if output_path is None:
        eval_name = (
            f"detailed_eval_result_top_{topk}.jsonl"
            if topk > 0
            else "detailed_eval_result.jsonl"
        )
        detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
        result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
        eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    else:
        output_dir = os.path.dirname(output_path)
        # output_base = os.path.splitext(os.path.basename(output_path))[0] # output_base không được sử dụng
        if not os.path.exists(output_dir) and output_dir != "": # Kiểm tra output_dir không phải là chuỗi rỗng
             os.makedirs(output_dir)

        eval_name = (
            f"detailed_eval_result_top_{topk}.jsonl"
            if topk > 0
            else "detailed_eval_result.jsonl"
        )
        detailed_eval_file = os.path.join(output_dir, eval_name)
        
        result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
        eval_result_path = os.path.join(output_dir, result_name)
    
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = [] # 修正拼写
    recall_list = []
    count_valid_samples = 0
    
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue
                
            id_val = data.get("id")
            raw_prediction_from_file = data.get("prediction") # 这是原始的，可能是字符串或列表
            answer = list(set(data.get("ground_truth", [])))
            
            if not answer:
                print(f"Skipping sample {id_val} due to empty ground_truth answers.")
                continue
            if raw_prediction_from_file is None:
                print(f"Skipping sample {id_val} due to missing prediction.")
                continue

            # `prediction_for_eval_acc_hit` 用于 eval_acc 和 eval_hit
            # `processed_prediction_list_for_f1` 用于 eval_f1
            prediction_for_eval_acc_hit_str = ""
            processed_prediction_list_for_f1 = []


            if cal_f1:
                # 1. 处理原始预测 (可能是字符串或列表) -> 得到用于F1的预测列表
                # extract_topk_prediction 处理字符串或列表，返回列表
                list_after_topk = extract_topk_prediction(raw_prediction_from_file, topk)
                
                # 2. 从 list_after_topk 提取特定答案部分 (predicted_ans)
                predicted_ans_extracted = []
                for p_item_str in list_after_topk:
                    # 确保 p_item_str 是字符串
                    if not isinstance(p_item_str, str):
                        p_item_str = str(p_item_str)

                    if "# Answer:\n" in p_item_str:
                        ans_segment = p_item_str.split("# Answer:\n", 1)[-1]
                        # 答案可能也是多行的，或者包含多个以换行符分隔的答案
                        for single_ans_line in ans_segment.splitlines(): # 使用 splitlines
                            stripped_ans = single_ans_line.strip()
                            if stripped_ans: # 避免添加空字符串
                                predicted_ans_extracted.append(stripped_ans)
                    else:
                        stripped_p = p_item_str.strip()
                        if stripped_p: # 避免添加空字符串
                            predicted_ans_extracted.append(stripped_p)
                
                processed_prediction_list_for_f1 = predicted_ans_extracted
                
                f1_score, local_precision_score, local_recall_score = eval_f1(processed_prediction_list_for_f1, answer)
                f1_list.append(f1_score)
                precision_list.append(local_precision_score) # 修正拼写
                recall_list.append(local_recall_score)
                
                # acc/hit 使用 extract_topk_prediction 的直接输出 (list_after_topk)，而不是 predicted_ans_extracted
                # 这是原始逻辑，如果希望 acc/hit 基于提取的答案，则应使用 predicted_ans_extracted
                # 为了保持与原始逻辑相似（acc/hit 基于 " ".join(prediction)），我们用 list_after_topk
                prediction_for_eval_acc_hit_str = " ".join(str(p) for p in list_after_topk)

            else: # not cal_f1
                # 确保 prediction_for_eval_acc_hit_str 是字符串
                if isinstance(raw_prediction_from_file, list):
                    prediction_for_eval_acc_hit_str = " ".join(str(p) for p in raw_prediction_from_file)
                    # 为了写入JSON，我们也需要一个列表形式的预测
                    # processed_prediction_list_for_f1 在这种情况下是原始预测（如果它是列表）
                    processed_prediction_list_for_f1 = [str(p) for p in raw_prediction_from_file]
                elif isinstance(raw_prediction_from_file, str):
                    prediction_for_eval_acc_hit_str = raw_prediction_from_file
                    processed_prediction_list_for_f1 = [raw_prediction_from_file] if raw_prediction_from_file else []
                else: # 其他类型，尝试转为字符串
                    prediction_for_eval_acc_hit_str = str(raw_prediction_from_file)
                    processed_prediction_list_for_f1 = [str(raw_prediction_from_file)] if raw_prediction_from_file is not None else []


            acc = eval_acc(prediction_for_eval_acc_hit_str, answer)
            hit = eval_hit(prediction_for_eval_acc_hit_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)
            
            count_valid_samples +=1
            
            output_data = {
                "id": id_val,
                "prediction_to_eval": processed_prediction_list_for_f1, # 这是用于F1或代表性预测的列表
                "ground_truth": answer,
                "ans_acc": acc,
                "ans_hit": hit,
            }
            if cal_f1:
                output_data["ans_f1"] = f1_score
                output_data["ans_precision"] = local_precision_score # 修正拼写
                output_data["ans_recall"] = local_recall_score
            
            f2.write(json.dumps(output_data) + "\n")

    metrics = {}
    if count_valid_samples == 0:
        print("No valid samples found for path_w_ans evaluation.")
        metrics = {"acc": 0.0, "hit": 0.0}
        if cal_f1:
            metrics.update({"f1": 0.0, "precision": 0.0, "recall": 0.0})
        result_str = "Accuracy: 0.00% Hit: 0.00%"
        if cal_f1:
             result_str += " F1: 0.00% Precision: 0.00% Recall: 0.00%"

    elif cal_f1 and len(f1_list) > 0 : # 确保 f1_list 有内容 (cal_f1=True 时应该有)
        metrics.update({
            "acc": sum(acc_list) / count_valid_samples,
            "hit": sum(hit_list) / count_valid_samples,
            "f1": sum(f1_list) / count_valid_samples,
            "precision": sum(precision_list) / count_valid_samples, # 修正拼写
            "recall": sum(recall_list) / count_valid_samples
        })
        result_str = (
            f"Accuracy: {metrics['acc'] * 100:.2f}% "
            f"Hit: {metrics['hit'] * 100:.2f}% "
            f"F1: {metrics['f1'] * 100:.2f}% "
            f"Precision: {metrics['precision'] * 100:.2f}% " # 修正拼写
            f"Recall: {metrics['recall'] * 100:.2f}%"
        )
    else: # not cal_f1 or f1_list is empty
        metrics.update({
            "acc": sum(acc_list) / count_valid_samples,
            "hit": sum(hit_list) / count_valid_samples
        })
        result_str = (
            f"Accuracy: {metrics['acc'] * 100:.2f}% "
            f"Hit: {metrics['hit'] * 100:.2f}%"
        )
    
    print(result_str)
    
    with open(eval_result_path, "w") as f:
        f.write(result_str)
        
    return metrics