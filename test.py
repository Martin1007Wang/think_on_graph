import os
import re
from datasets import load_dataset
from collections import Counter
from typing import Dict, List, Tuple

def analyze_answers(dataset_path: str, split: str = 'test') -> Tuple[Dict, float]:
    """
    分析数据集中answer字段的格式，统计以m.开头等非自然语言形式的占比。
    
    Args:
        dataset_path: 数据集路径
        split: 数据集分割
        
    Returns:
        统计结果字典和非自然语言占比
    """
    # 加载数据集
    dataset = load_dataset(dataset_path, split=split)
    
    # 统计计数器
    answer_formats = Counter()
    total_answers = 0
    m_code_answers = 0
    
    # 分析所有答案
    for item in dataset:
        answer = item.get('answer', '')
        if not answer:
            continue
            
        total_answers += 1
        
        # 检查是否以m.开头的编码
        if isinstance(answer, str):
            if re.match(r'^m\.', answer.strip()):
                answer_formats['m_code_exact'] += 1
                m_code_answers += 1
            elif 'm.' in answer:
                answer_formats['contains_m_code'] += 1
                m_code_answers += 1
            else:
                # 分析其他可能的非自然语言格式
                if re.match(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+', answer.strip()):
                    answer_formats['other_code_format'] += 1
                    m_code_answers += 1
                else:
                    answer_formats['natural_language'] += 1
        elif isinstance(answer, list):
            # 处理列表类型的答案
            has_m_code = False
            for ans in answer:
                if isinstance(ans, str) and ('m.' in ans or re.match(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+', ans.strip())):
                    has_m_code = True
                    break
            
            if has_m_code:
                answer_formats['list_with_m_code'] += 1
                m_code_answers += 1
            else:
                answer_formats['list_natural_language'] += 1
    
    # 计算占比
    percentage = (m_code_answers / total_answers * 100) if total_answers > 0 else 0
    
    # 在analyze_answers函数中添加
    examples = []
    for i, item in enumerate(dataset):
        answer = item.get('answer', '')
        if isinstance(answer, str) and 'm.' in answer:
            examples.append({
                'index': i,
                'question': item.get('question', ''),
                'answer': answer
            })
        if len(examples) >= 5:  # 只保存前5个例子
            break

    # 打印例子
    print("\n示例:")
    for ex in examples:
        print(f"问题 {ex['index']}: {ex['question']}")
        print(f"答案: {ex['answer']}")
        print("---")
    
    return answer_formats, percentage

if __name__ == "__main__":
    dataset_path = os.path.join('rmanluo', 'RoG-webqsp')
    formats, percentage = analyze_answers(dataset_path)
    
    print(f"分析结果:")
    for format_type, count in formats.items():
        print(f"- {format_type}: {count}")
    
    print(f"\n非自然语言占比: {percentage:.2f}%")
    print(f"总样本数: {sum(formats.values())}")

