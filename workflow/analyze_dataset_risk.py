import json
import argparse
import logging
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

# =============================================================================
#  使用说明 (INSTRUCTIONS)
# =============================================================================
# 1. 将此文件保存为 `analyze_dataset_risk.py`，并放在您的项目根目录下。
# 2. 确保您的环境中已安装 `transformers` 和 `numpy`。
# 3. 在终端中运行此脚本，并传入您的数据集文件路径和模型名称。
#    示例:
#    python analyze_dataset_risk.py ./path/to/your/mpo_preference_data.jsonl --model_name unsloth/llama-3-8b
# 4. 脚本将输出风险最高的 Top K 个样本，这些就是最可能导致 OOM 的“元凶”。
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset(file_path: str, model_name: str, top_k: int = 20):
    """
    分析偏好数据集，找出具有高显存占用风险的样本。
    """
    logger.info(f"Loading tokenizer: {model_name}...")
    # 推荐使用与您训练时完全相同的分词器
    try:
        # 尝试使用 unsloth 的快速分词器（如果可用）
        from unsloth import FastLanguageModel
        _, tokenizer = FastLanguageModel.from_pretrained(model_name)
        logger.info("Using Unsloth's FastTokenizer.")
    except (ImportError, ValueError):
        logger.warning("Unsloth not found or model not compatible, falling back to standard AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Tokenizer loaded.")

    samples_with_risk = []

    logger.info(f"Analyzing dataset file: {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Analyzing samples")):
            try:
                data = json.loads(line)
                
                prompt = data.get("prompt", "")
                chosen_list = data.get("chosen", [])
                rejected_list = data.get("rejected", [])
                
                # 1. 计算各部分的 token 长度
                prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
                prompt_len = len(prompt_tokens)
                
                completions = chosen_list + rejected_list
                if not completions:
                    avg_completion_len = 0
                else:
                    completion_lens = [len(tokenizer(c, add_special_tokens=False)['input_ids']) for c in completions]
                    avg_completion_len = np.mean(completion_lens)
                
                # 2. 计算有效批次大小的“放大系数”
                effective_batch_multiplier = len(completions)
                if effective_batch_multiplier == 0:
                    continue

                # 3. 计算平均序列总长度 L
                sequence_len = prompt_len + avg_completion_len

                # 4. 计算风险分数 (Risk Score)，正比于 B_eff * L^2
                risk_score = effective_batch_multiplier * (sequence_len ** 2)

                samples_with_risk.append({
                    "line_num": i + 1,
                    "prompt_len": prompt_len,
                    "num_completions": effective_batch_multiplier,
                    "avg_completion_len": int(avg_completion_len),
                    "total_seq_len": int(sequence_len),
                    "risk_score": int(risk_score)
                })

            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON on line {i+1}. Skipping.")
                continue

    # 按风险分数从高到低排序
    sorted_samples = sorted(samples_with_risk, key=lambda x: x['risk_score'], reverse=True)

    print("\n\n===== 高风险样本分析报告 (TOP DANGEROUS SAMPLES REPORT) =====")
    print("以下样本具有最高的“显存风险分数” (B_eff * L^2)。")
    print("它们是导致 CUDA Out-of-Memory 错误的最可能原因。\n")
    print(f"{'Line #':<8} | {'Risk Score':<15} | {'Num Completions':<18} | {'Prompt Len':<12} | {'Avg Compl. Len':<15} | {'Total Seq. Len':<15}")
    print("-" * 100)
    
    for sample in sorted_samples[:top_k]:
        print(
            f"{sample['line_num']:<8} | "
            f"{sample['risk_score']::<15} | "
            f"{sample['num_completions']:<18} | "
            f"{sample['prompt_len']:<12} | "
            f"{sample['avg_completion_len']:<15} | "
            f"{sample['total_seq_len']:<15}"
        )

    print("\n===== 建议操作 (RECOMMENDATION) =====")
    print("1. 根据上面报告中的行号，检查您的数据集文件中的高风险样本。")
    print("2. 考虑从训练集中【过滤】掉这些样本，或者【手动编辑】它们以降低其复杂度。")
    print("   - 例如，对于这些特定的样本，进一步减少其 'rejected' 回应的数量。")
    print("3. 在清理数据后，重新运行您的训练脚本。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析偏好数据集，找出具有高显存占用风险的样本。")
    parser.add_argument("--file_path", type=str, default="/mnt/wangjingxiong/think_on_graph/data/preference_dataset_v2/cand_pn_only_pos_shortest_paths/mpo_preference_data.jsonl",help="您的 .jsonl 数据集文件路径。")
    parser.add_argument("--model_name", type=str, default="/mnt/data/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit/snapshots/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec", help="用于分词的模型名称或路径。")
    parser.add_argument("--top_k", type=int, default=20, help="要显示的最高风险样本数量。")
    
    args = parser.parse_args()
    analyze_dataset(args.file_path, args.model_name, args.top_k)
