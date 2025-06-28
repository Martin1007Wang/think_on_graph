import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any
from collections import Counter, defaultdict
import logging

# ... (顶部的函数 parse_path_to_segments, build_history_key, print_distribution 保持不变) ...
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pre-compiled Regex for Performance ---
PATH_STEP_REGEX = re.compile(r"(.+?)\s*-\[\s*(.+?)\s*\]->\s*(.+)")

def parse_path_to_segments(path_str: str) -> List[Tuple[str, str, str]]:
    """将路径字符串解析为 (源实体, 关系, 目标实体) 的段列表。"""
    if not path_str:
        return []
    segments: List[Tuple[str, str, str]] = []
    path_steps = path_str.split(" ; ")
    for step in path_steps:
        match = PATH_STEP_REGEX.match(step)
        if match:
            src, rel, tgt = match.groups()
            segments.append((src.strip(), rel.strip(), tgt.strip()))
    return segments

def build_history_key(history_segments: List[Tuple[str, str, str]]) -> str:
    """根据历史段落构建一个唯一的字符串键。"""
    if not history_segments:
        return ""
    return " ; ".join([f"{s}-[{r}]->{t}" for s, r, t in history_segments])

def print_distribution(title: str, counter: Counter, top_n: int = 20):
    """以可读的格式打印分布统计信息。"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    if not counter:
        print("  (No data to display)")
        return

    total_count = sum(counter.values())
    print(f"{'Value':<20} | {'Count':>10} | {'Percentage':>12} | {'Cumulative %':>15}")
    print("-" * 80)
    
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    cumulative_percentage = 0.0
    for i, (value, count) in enumerate(sorted_items):
        percentage = (count / total_count) * 100
        cumulative_percentage += percentage
        
        # 对于关系频率，value可能很长，需要截断
        display_value = str(value)
        if len(display_value) > 18:
            display_value = display_value[:15] + "..."

        print(f"{display_value:<20} | {count:>10} | {percentage:>11.2f}% | {cumulative_percentage:>14.2f}%")
        
        if i >= top_n - 1 and len(sorted_items) > top_n:
            remaining_items = len(sorted_items) - top_n
            print(f"... and {remaining_items} more ...")
            break


def analyze_path_data(input_paths: List[str]):
    """主分析函数，处理所有输入文件并聚合统计数据。"""
    
    # 统计数据结构保持不变
    positive_path_lengths = Counter()
    negative_path_lengths = Counter()
    total_relation_counts = Counter()
    positive_relation_counts = Counter()
    negative_relation_counts = Counter()
    choices_per_state = defaultdict(lambda: {"positive": set(), "negative": set()})
    total_items_processed = 0
    
    logger.info(f"Starting analysis on {len(input_paths)} file(s)...")

    all_data = []
    for file_path in input_paths:
        logger.info(f"Processing file: {file_path}")
        try:
            # ===============================================================
            #  核心修改在这里：使用 json.load() 来读取整个文件
            # ===============================================================
            with open(file_path, 'r', encoding='utf-8') as f:
                # json.load(f) 会将整个文件内容作为一个单一的JSON对象/数组进行解析
                loaded_data = json.load(f)
                
                # 确保加载的数据是一个列表
                if isinstance(loaded_data, list):
                    all_data.extend(loaded_data)
                else:
                    logger.warning(f"File {file_path} does not contain a JSON array at the top level. Skipping.")
                    continue
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {file_path} as a single, valid JSON file. Error: {e}")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
            continue

    logger.info(f"Successfully loaded a total of {len(all_data)} items from all files.")
    
    for item in all_data:
        total_items_processed += 1
        
        # --- 后续的统计逻辑完全不需要改变 ---
        item_states = defaultdict(lambda: {"positive": set(), "negative": set()})

        for path_str in item.get('positive_paths', []) + item.get('shortest_paths', []):
            segments = parse_path_to_segments(path_str)
            positive_path_lengths[len(segments)] += 1
            history = []
            for src, rel, tgt in segments:
                total_relation_counts[rel] += 1
                positive_relation_counts[rel] += 1
                state_key = (build_history_key(history), src)
                item_states[state_key]["positive"].add(rel)
                history.append((src, rel, tgt))
        
        for path_str in item.get('negative_paths', []):
            segments = parse_path_to_segments(path_str)
            negative_path_lengths[len(segments)] += 1
            history = []
            for src, rel, tgt in segments:
                total_relation_counts[rel] += 1
                negative_relation_counts[rel] += 1
                state_key = (build_history_key(history), src)
                item_states[state_key]["negative"].add(rel)
                history.append((src, rel, tgt))
        
        for state_key, relations in item_states.items():
            choices_per_state[state_key]["positive"].update(relations["positive"])
            choices_per_state[state_key]["negative"].update(relations["negative"])

    logger.info("Analysis complete. Aggregating final statistics...")

    # ... (后续的打印报告和保存JSON的逻辑保持完全不变) ...
    num_pos_choices_dist = Counter(len(state["positive"]) for state in choices_per_state.values())
    num_neg_choices_dist = Counter(len(state["negative"]) for state in choices_per_state.values())
    total_choices_dist = Counter(len(state["positive"]) + len(state["negative"]) for state in choices_per_state.values())
    
    # --- 打印报告 ---
    print("\n" + "#"*80)
    print("###               Paths Dataset Statistical Analysis Report               ###")
    print("#"*80)
    print(f"\n- Total Questions/Items Processed: {total_items_processed}")
    print(f"- Total Unique Decision Points (Entity + History): {len(choices_per_state)}")

    print_distribution("Distribution of POSITIVE Choices per Decision Point", num_pos_choices_dist)
    print_distribution("Distribution of NEGATIVE Choices per Decision Point", num_neg_choices_dist)
    print_distribution("Distribution of TOTAL Choices per Decision Point", total_choices_dist)
    
    print_distribution("Distribution of Positive Path Lengths (hops)", positive_path_lengths)
    print_distribution("Distribution of Negative Path Lengths (hops)", negative_path_lengths)
    
    print_distribution("Top 20 Most Frequent Relations (Overall)", total_relation_counts)
    print_distribution("Top 20 Most Frequent Relations in POSITIVE Paths", positive_relation_counts)
    print_distribution("Top 20 Most Frequent Relations in NEGATIVE Paths", negative_relation_counts)

    # --- 保存详细统计数据到JSON ---
    output_stats = {
        "summary": {
            "total_items_processed": total_items_processed,
            "total_unique_decision_points": len(choices_per_state)
        },
        "distributions": {
            "positive_choices_per_state": dict(num_pos_choices_dist),
            "negative_choices_per_state": dict(num_neg_choices_dist),
            "total_choices_per_state": dict(total_choices_dist),
            "positive_path_lengths": dict(positive_path_lengths),
            "negative_path_lengths": dict(negative_path_lengths)
        },
        "frequencies": {
            "top_50_total_relations": dict(total_relation_counts.most_common(50)),
            "top_50_positive_relations": dict(positive_relation_counts.most_common(50)),
            "top_50_negative_relations": dict(negative_relation_counts.most_common(50))
        }
    }
    
    output_filename = "path_data_statistics.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_stats, f, indent=4)
        
    logger.info(f"Detailed statistics saved to '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze path-based datasets to gather statistics for preference data creation.")
    parser.add_argument('input_paths', nargs='+', help='One or more paths to the input standard JSON or JSONL data files.')
    args = parser.parse_args()
    
    # 注意：这个脚本现在主要设计为处理标准JSON数组文件。
    # 如果您确定文件是JSON Lines，请使用我们之前的脚本版本。
    analyze_path_data(args.input_paths)