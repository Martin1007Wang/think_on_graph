import os
import json
import argparse
import logging
from typing import Dict, Any

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transform_line(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单行旧格式的JSON数据转换为新格式。
    """
    prediction_str = old_data.get("prediction", "")
    if isinstance(prediction_str, str):
        # 1. 按逗号分割
        # 2. 对分割后的每一项去除首尾空格 (strip)
        # 3. 过滤掉处理后可能产生的空字符串
        answer_entities = [item.strip() for item in prediction_str.split(',') if item.strip()]
    else:
        # 如果 prediction 字段已经是列表或其他格式，则直接使用，增加兼容性
        answer_entities = old_data.get("prediction", [])
    # 1. 直接映射或重命名的字段
    new_data = {
        "id": old_data.get("id"),
        "question": old_data.get("question"),
        "ground_truth": old_data.get("ground_truth"),
        "answer_entities": answer_entities,
        "reasoning_paths": old_data.get("reasoning"),
        "reasoning_summary": old_data.get("analysis"),
        "answer_found_during_exploration": old_data.get("answer_found"),
        "fallback_used": old_data.get("fallback_used"),
        "start_entities": old_data.get("start_entities"),
        "exploration_history": old_data.get("exploration_history"),
    }
    
    return new_data

def convert_prediction_file(input_path: str, output_path: str):
    """
    读取整个旧格式文件，逐行转换，并写入新格式文件。
    """
    logging.info(f"开始转换文件: {input_path}")
    line_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                try:
                    old_data = json.loads(line)
                    new_data = transform_line(old_data)
                    f_out.write(json.dumps(new_data) + '\n')
                    line_count += 1
                except json.JSONDecodeError:
                    logging.warning(f"跳过无效的JSON行 (行号 {line_count + 1}): {line.strip()}")
                    continue

    except FileNotFoundError:
        logging.error(f"错误: 输入文件未找到 -> {input_path}")
        return

    logging.info(f"文件转换完成。共处理 {line_count} 行。")
    logging.info(f"转换后的文件已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将旧格式的 prediction.jsonl 文件转换为新格式。")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="旧格式的 JSONL 文件路径。"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="（可选）指定输出文件的路径。默认为在原文件名后添加 '_converted'。"
    )
    args = parser.parse_args()
    
    # 如果未指定输出文件，则自动生成一个
    output_file_path = args.output_file
    if not output_file_path:
        base, ext = os.path.splitext(args.input_file)
        output_file_path = f"{base}_converted{ext}"
        
    convert_prediction_file(args.input_file, output_file_path)

if __name__ == "__main__":
    main()