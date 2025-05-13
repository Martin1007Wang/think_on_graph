from collections import OrderedDict
import json
import re
import string
from sklearn.metrics import precision_score
from statistics import mean
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


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction, answer):
    if len(prediction) == 0 or len(answer) == 0:
        return 0, 0, 0
    ans_recalled = 0
    prediction_correct = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            ans_recalled += 1
    recall = ans_recalled / len(answer)
    for p in prediction:
        for a in answer:
            if match(p, a):
                prediction_correct += 1
                break
    precision = prediction_correct / len(prediction)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return (2 * precision * recall) / (precision + recall), precision, recall


def extract_topk_prediction(prediction, k=-1):
    if isinstance(prediction, str):
        prediction = prediction.split(",")
    results = {}
    for p in prediction:
        if p.strip() == "":
            continue
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_rank_results(predict_file, topk=[1, 3, 5, 10]):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = (
        f"detailed_eval_result_top_{topk}.jsonl"
        if topk
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    all_acc_list = OrderedDict({k: [] for k in topk})
    all_hit_list = OrderedDict({k: [] for k in topk})
    all_f1_list = OrderedDict({k: [] for k in topk})
    all_precission_list = OrderedDict({k: [] for k in topk})
    all_recall_list = OrderedDict({k: [] for k in topk})
    
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            answer = list(set(data["answer"]))
            acc_list = OrderedDict()
            hit_list = OrderedDict()
            f1_list = OrderedDict()
            precission_list = OrderedDict()
            recall_list = OrderedDict()
            for k in topk:
                top_k_pred = min(k, len(data['ranks']))
                topk_rank = data['ranks'][:top_k_pred]
                prediction = [r['response'] for r in topk_rank]
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list[k] = acc
                hit_list[k] = hit
                f1_list[k]= f1_score
                precission_list[k] = precision_score
                recall_list[k] = recall_score
            f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc@k": acc_list,
                            "hit@k": hit_list,
                            "f1@k": f1_list,
                            "precission@k": precission_list,
                            "recall@k": recall_list,
                        }
                    )
                    + "\n"
                )
            for k in topk:
                all_acc_list[k].append(acc_list[k])
                all_hit_list[k].append(hit_list[k])
                all_f1_list[k].append(f1_list[k])
                all_precission_list[k].append(precission_list[k])
                all_recall_list[k].append(recall_list[k])
    result_str = ""
    for k in topk:
        result_str += f"Top-{k}:\n"
        result_str += (
            "Accuracy: "
            + str(sum(all_acc_list[k]) * 100 / len(all_acc_list[k]))
            + " Hit: "
            + str(sum(all_hit_list[k]) * 100 / len(all_hit_list[k]))
            + " F1: "
            + str(sum(all_f1_list[k]) * 100 / len(all_f1_list[k]))
            + " Precision: "
            + str(sum(all_precission_list[k]) * 100 / len(all_precission_list[k]))
            + " Recall: "
            + str(sum(all_recall_list[k]) * 100 / len(all_recall_list[k]))
            + "\n"
        )
    print(result_str)
    result_name = f"eval_result_top_{topk}.txt" if topk else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)
                
    
def eval_result(predict_file, cal_f1=True, topk=-1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = (
        "detailed_eval_result_top_{topk}.jsonl"
        if topk > 0
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                            "f1": f1_score,
                            "precission": precision_score,
                            "recall": recall_score,
                        }
                    )
                    + "\n"
                )
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    if len(f1_list) > 0:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
            + " F1: "
            + str(sum(f1_list) * 100 / len(f1_list))
            + " Precision: "
            + str(sum(precission_list) * 100 / len(precission_list))
            + " Recall: "
            + str(sum(recall_list) * 100 / len(recall_list))
        )
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)

def eval_joint_result(predict_file):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result.jsonl"
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    path_f1_list = []
    path_precission_list = []
    path_recall_list = []
    path_ans_f1_list = []
    path_ans_recall_list = []
    path_ans_precision_list = []
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            # Extract reasoning paths and answers
            predicted_reasoning_paths = set()
            predicted_answers = set()

            for pre in prediction:
                try:
                    ans_in_pred = False
                    if "the answer is: " in pre:
                        ans_in_pred = True
                        ans_pred = pre.split("the answer is: ")[1]
                        for ans in ans_pred.split("\n"):
                            predicted_answers.add(ans.strip())
                    if "Reasoning path:\n" in pre:
                        if ans_in_pred:
                            path_pred = pre.split("Reasoning path:\n")[1].split("\nthe answer is: ")[0]
                        else:
                            path_pred = pre.split("Reasoning path:\n")[1]
                        for path in path_pred.split("\n")[:-1]:
                            predicted_reasoning_paths.add(path.strip())
                except Exception as e:
                    print("Error in line: ", pre)
                    print(e)
                    continue
            predicted_reasoning_paths = list(predicted_reasoning_paths)
            predicted_answers = list(predicted_answers)
            
            f1_score, precision_score, recall_score = eval_f1(predicted_answers, answer)
            f1_list.append(f1_score)
            precission_list.append(precision_score)
            recall_list.append(recall_score)
            prediction_str = " ".join(predicted_answers)
            acc = eval_acc(prediction_str, answer)
            hit = eval_hit(prediction_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)
            path_f1_score, path_precision_score, path_recall_score = eval_f1(
                predicted_reasoning_paths, data["ground_truth_paths"]
            )
            path_f1_list.append(path_f1_score)
            path_precission_list.append(path_precision_score)
            path_recall_list.append(path_recall_score)
            path_ans_f1_score, path_ans_precision_score, path_ans_recall_score = eval_f1(predicted_reasoning_paths, answer)
            path_ans_f1_list.append(path_ans_f1_score)
            path_ans_precision_list.append(path_ans_precision_score)
            path_ans_recall_list.append(path_ans_recall_score)
            f2.write(
                json.dumps(
                    {
                        "id": id,
                        "prediction": prediction,
                        "ground_truth": answer,
                        "ans_acc": acc,
                        "ans_hit": hit,
                        "ans_f1": f1_score,
                        "ans_precission": precision_score,
                        "ans_recall": recall_score,
                        "path_f1": path_f1_score,
                        "path_precision": path_precision_score,
                        "path_recall": path_recall_score,
                        "path_ans_f1": path_ans_f1_score,
                        "path_ans_precision": path_ans_precision_score,
                        "path_ans_recall": path_ans_recall_score
                    }
                )
                + "\n"
            )


    if len(f1_list) > 0:
        result_str = f"Accuracy: {sum(acc_list) * 100 / len(acc_list)} Hit: {sum(hit_list) * 100 / len(hit_list)} F1: {sum(f1_list) * 100 / len(f1_list)} Precision: {sum(precission_list) * 100 / len(precission_list)} Recall: {sum(recall_list) * 100 / len(recall_list)} Path F1: {sum(path_f1_list) * 100 / len(path_f1_list)} Path Precision: {sum(path_precission_list) * 100 / len(path_precission_list)} Path Recall: {sum(path_recall_list) * 100 / len(path_recall_list)} Path Ans F1: {mean(path_ans_f1_list)} Path Ans Precision: {mean(path_ans_precision_list)} Path Ans recall: {mean(path_ans_recall_list)}"
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result.txt"
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
    if cal_f1:
        sum_acc = sum_hit = sum_f1 = sum_precision = sum_recall = 0
        sum_path_f1 = sum_path_precision = sum_path_recall = 0
        count = 0
    else:
        sum_acc = sum_hit = count = 0

    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue
            id = data.get("id")
            prediction = data.get("prediction", [])
            # Remove start_entities from predictions
            start_entities = set(data.get("start_entities", []))
            prediction = [p for p in prediction if p not in start_entities]
            answer = list(set(data.get("ground_truth", [])))

            if not prediction or len(data.get("ground_truth_paths", [])) == 0 or not answer:
                continue

            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                path_f1_score, path_precision_score, path_recall_score = eval_f1(
                    prediction, data.get("ground_truth_paths", [])
                )

            # Compute accuracy and hit
            if cal_f1:
                pred_str = " ".join(prediction)
                acc = eval_acc(pred_str, answer)
                hit = eval_hit(pred_str, answer)
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)

            # Update sums
            sum_acc += acc
            sum_hit += hit
            if cal_f1:
                sum_f1 += f1_score
                sum_precision += precision_score
                sum_recall += recall_score
                sum_path_f1 += path_f1_score
                sum_path_precision += path_precision_score
                sum_path_recall += path_recall_score

            count += 1

            # Write detailed results
            if cal_f1:
                f2.write(
                    json.dumps({
                        "id": id,
                        "prediction": prediction,
                        "ground_truth": answer,
                        "ans_acc": acc,
                        "ans_hit": hit,
                        "ans_f1": f1_score,
                        "ans_precission": precision_score,
                        "ans_recall": recall_score,
                        "path_f1": path_f1_score,
                        "path_precision": path_precision_score,
                        "path_recall": path_recall_score,
                    }) + "\n"
                )
            else:
                f2.write(
                    json.dumps({
                        "id": id,
                        "prediction": prediction,
                        "ground_truth": answer,
                        "acc": acc,
                        "hit": hit,
                    }) + "\n"
                )

    # Compute and print summary
    if cal_f1 and count > 0:
        avg_mult = 100.0 / count
        result_str = (
            f"Accuracy: {sum_acc * avg_mult} "
            f"Hit: {sum_hit * avg_mult} "
            f"F1: {sum_f1 * avg_mult} "
            f"Precision: {sum_precision * avg_mult} "
            f"Recall: {sum_recall * avg_mult} "
            f"Path F1: {sum_path_f1 * avg_mult} "
            f"Path Precision: {sum_path_precision * avg_mult} "
            f"Path Recall: {sum_path_recall * avg_mult}"
        )
    else:
        mult = 100.0 / count if count > 0 else 0.0
        result_str = f"Accuracy: {sum_acc * mult} Hit: {sum_hit * mult}"

    print(result_str)
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)

def eval_path_result_w_ans(predict_file, cal_f1=True, topk=-1, output_path=None):
    """
    Evaluate path results with answer extraction.
    
    Args:
        predict_file: Path to the prediction JSONL file.
        cal_f1: Whether to calculate F1 score.
        topk: Top-k predictions to consider.
        output_path: Optional path to save evaluation results.
            If None, results will be saved in the same directory as predict_file.
    
    Returns:
        A dictionary with evaluation metrics.
    """
    # Generate evaluation filenames based on input file
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
        # If output_path is provided, use it as base for output files
        output_dir = os.path.dirname(output_path)
        output_base = os.path.splitext(os.path.basename(output_path))[0]
        
        eval_name = (
            f"detailed_eval_result_top_{topk}.jsonl"
            if topk > 0
            else "detailed_eval_result.jsonl"
        )
        detailed_eval_file = os.path.join(output_dir, eval_name)
        
        result_name = f"eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
        eval_result_path = os.path.join(output_dir, result_name)
    
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
                
            id = data["id"]
            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            
            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                
                predicted_ans = []
                for p in prediction:
                    # 简化处理，只提取答案部分
                    if "# Answer:\n" in p:
                        ans = p.split("# Answer:\n")[-1]
                        predicted_ans.append(ans.strip())
                    else:
                        # 如果没有特定格式，则将整个预测作为答案
                        predicted_ans.append(p.strip())
                
                f1_score, precision_score, recall_score = eval_f1(predicted_ans, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "ans_acc": acc,
                            "ans_hit": hit,
                            "ans_f1": f1_score,
                            "ans_precision": precision_score,
                            "ans_recall": recall_score,
                        }
                    )
                    + "\n"
                )
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    # Calculate and format results
    metrics = {}
    if len(f1_list) > 0:
        metrics.update({
            "acc": sum(acc_list) / len(acc_list),
            "hit": sum(hit_list) / len(hit_list),
            "f1": sum(f1_list) / len(f1_list),
            "precision": sum(precission_list) / len(precission_list),
            "recall": sum(recall_list) / len(recall_list)
        })
        
        result_str = (
            f"Accuracy: {metrics['acc'] * 100:.2f}% "
            f"Hit: {metrics['hit'] * 100:.2f}% "
            f"F1: {metrics['f1'] * 100:.2f}% "
            f"Precision: {metrics['precision'] * 100:.2f}% "
            f"Recall: {metrics['recall'] * 100:.2f}%"
        )
    else:
        metrics.update({
            "acc": sum(acc_list) / len(acc_list),
            "hit": sum(hit_list) / len(hit_list)
        })
        
        result_str = (
            f"Accuracy: {metrics['acc'] * 100:.2f}% "
            f"Hit: {metrics['hit'] * 100:.2f}%"
        )
    
    print(result_str)
    
    with open(eval_result_path, "w") as f:
        f.write(result_str)
        
    return metrics