# -*- coding: utf-8 -*-
import torch
import logging
from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer
from typing import List

# --- 配置区域 ---

# 请将此路径替换为您微调并合并后的模型所在的目录
# 这个目录应该包含 model.safetensors, config.json, tokenizer.json 等文件
MODEL_PATH = "/mnt/wangjingxiong/think_on_graph/mpo_models/cand_pn_only_pos_shortest_paths/hf_dataset/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec_ep1_loss-mpo_b0.1_sft0.05_lr1e-5_lora-r8-a16_bf16/merged_model" 

# 加载模型时使用的精度。应与您训练时使用的精度一致。
# 您的训练脚本中 BF16=True, 所以这里也使用 "bf16"
DTYPE = "bf16" 

# 加载模型时使用的量化方式。您的训练脚本中 LOAD_IN_8BIT=True。
# 如果是4-bit，请设置为 "4bit"。如果是8-bit，则为 "8bit"。如果无量化，则为 None。
LOAD_IN_QUANTIZATION = "4bit" # "8bit", "4bit", or None

# -----------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_path: str, dtype: str, load_in_quant: str):
    """
    使用 unsloth 高效加载模型和分词器。
    """
    logging.info(f"[*] 正在从 '{model_path}' 加载模型...")
    
    # 根据量化设置确定加载参数
    load_in_4bit = (load_in_quant == "4bit")
    load_in_8bit = (load_in_quant == "8bit")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024, # 应与训练时设置的 max_length 保持一致
        dtype=getattr(torch, dtype, None),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    
    # 为模型启用 PEFT (如果模型是 LoRA 合并的，unsloth 会自动处理)
    # 如果您加载的是基础模型和LoRA适配器，则需要使用 PeftModel.from_pretrained
    # 但您的路径指向一个 merged_model，所以这种直接加载的方式是正确的。
    logging.info("[*] 模型和分词器加载成功。")

    # 配置分词器的填充(padding)功能，这对于批处理至关重要
    if tokenizer.pad_token is None:
        # 如果模型没有预设的 pad_token，通常使用 eos_token 作为替代
        tokenizer.pad_token = tokenizer.eos_token 
        logging.info("分词器的 pad_token 未设置，已将其设置为 eos_token。")
    
    return model, tokenizer

def test_batch_generation_with_formatted_prompts(model, tokenizer):
    """
    测试并演示模型处理预格式化prompt的批处理能力。
    """
    print("\n--- 开始批量生成测试 (使用预格式化的指令) ---")
    
    # 1. 定义一个包含多个预格式化指令的输入批次
    #    这模拟了您在关系选择任务中会遇到的情况
    pre_formatted_prompts = [
f"""*Role:** KG Strategist
**Objective:** Identify paths to answer: "what is the name of justin bieber brother"
**Current Entity:** "Justin Bieber"
**Task:** From the 'Available Relations' listed below, select **up to 5** distinct relations.
You MUST choose relations that are MOST LIKELY to lead to relevant information for the Objective.
**Available Relations for "Justin Bieber":**
```
base.popstra.celebrity.breakup-base.popstra.breakup.participant
base.popstra.celebrity.friendship-base.popstra.friendship.participant
celebrities.celebrity.celebrity_friends-celebrities.friendship.friend
people.person.sibling_s-people.sibling_relationship.sibling
```
**Output Requirements:**
* Respond ONLY with the selected relations. NO other text, explanations, or comments.
* Each selected relation MUST be an **exact, verbatim copy** of a complete line from the 'Available Relations' list above.
* Output EACH selected relation on a **new line**.
**Example of Correct Output Format (if REL_A and REL_B were selected):**
[REL_A] chosen.relation.example_one
[REL_B] another.chosen.relation
**Your Selection:""",
f"""**Role:** KG Strategist
**Objective:** Identify paths to answer: "what character did natalie portman play in star wars"
**Current Entity:** "Natalie Portman"
**Task:** From the 'Available Relations' listed below, select **up to 5** distinct relations.
You MUST choose relations that are MOST LIKELY to lead to relevant information for the Objective.
**Available Relations for "Natalie Portman":**
```
film.actor.film-film.performance.character
film.actor.film-film.performance.special_performance_type
tv.tv_actor.guest_roles-tv.tv_guest_role.episodes_appeared_in
tv.tv_actor.guest_roles-tv.tv_guest_role.special_performance_type
```
**Output Requirements:**
* Respond ONLY with the selected relations. NO other text, explanations, or comments.
* Each selected relation MUST be an **exact, verbatim copy** of a complete line from the 'Available Relations' list above.
* Output EACH selected relation on a **new line**.
**Example of Correct Output Format (if REL_A and REL_B were selected):**
[REL_A] chosen.relation.example_one
[REL_B] another.chosen.relation
**Your Selection:""",
f"""**Role:** KG Strategist
**Objective:** Identify paths to answer: "what country is the grand bahama island in"
**Current Entity:** "Grand Bahama"
**Task:** From the 'Available Relations' listed below, select **up to 5** distinct relations.
You MUST choose relations that are MOST LIKELY to lead to relevant information for the Objective.
**Available Relations for "Grand Bahama":**
```
common.topic.webpage-common.webpage.in_index
location.location.containedby
location.location.nearby_airports
location.location.people_born_here
```
**Output Requirements:**
* Respond ONLY with the selected relations. NO other text, explanations, or comments.
* Each selected relation MUST be an **exact, verbatim copy** of a complete line from the 'Available Relations' list above.
* Output EACH selected relation on a **new line**.
**Example of Correct Output Format (if REL_A and REL_B were selected):**
[REL_A] chosen.relation.example_one
[REL_B] another.chosen.relation
**Your Selection:"""
    ]
    
    logging.info(f"[*] 准备处理 {len(pre_formatted_prompts)} 个预格式化prompt的批次...")

    # 2. 对整个批次进行分词
    #    padding=True 会将批次中的所有句子填充到最长句子的长度
    #    return_tensors="pt" 返回 PyTorch 张量
    inputs = tokenizer(pre_formatted_prompts, padding=True, return_tensors="pt").to("cuda")

    # 3. 定义生成参数
    generation_params = {
        "max_new_tokens": 128,  # 对于关系选择任务，回复通常较短
        "do_sample": False,     # 对于指令任务，通常使用确定性解码
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # 4. 使用单次调用处理整个批次
    logging.info("[*] 调用 model.generate 处理整个批次... 🤖 模型正在生成...")
    outputs = model.generate(**inputs, **generation_params)
    
    # 5. 对批处理结果进行解码
    #    skip_special_tokens=True 会在解码时移除特殊的token（如 padding, eos）
    batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 6. 打印批处理结果
    #    注意：生成的文本会包含原始的prompt部分，我们需要从中提取出模型真正新生成的部分
    print("\n--- 批量生成结果 ---")
    for i, (original_prompt, full_response) in enumerate(zip(pre_formatted_prompts, batch_responses)):
        # 提取模型新生成的部分
        generated_text = full_response[len(original_prompt):].strip()
        
        print("-" * 50)
        print(f"👤 格式化输入 {i+1}:\n{original_prompt}\n")
        print(f"🤖 模型回复 (仅新生成部分):\n{generated_text}")
    print("-" * 50)


def main():
    """主函数"""
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DTYPE, LOAD_IN_QUANTIZATION)
        test_batch_generation_with_formatted_prompts(model, tokenizer)
    except FileNotFoundError:
        logging.error(f"[!] 模型路径未找到: '{MODEL_PATH}'")
        logging.error("[!] 请确保 MODEL_PATH 变量指向了正确的位置。")
    except Exception as e:
        logging.error(f"\n[!] 推理过程中发生未知错误: {e}", exc_info=True)
        logging.error("[!] 请检查您的环境、模型路径和CUDA设置。")

if __name__ == "__main__":
    main()
