import torch
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, TrainingArguments, BitsAndBytesConfig
from tqdm import tqdm
import logging
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 从您的项目中导入相关类
# 假设您的代码文件名为 mpo_trainer.py
from src.mpo_trainer_v2 import MPOTrainer, DataCollatorForMPO, create_navigation_dataset 

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_stress_test(
    model_name: str,
    dummy_data: list,
    num_test_steps: int = 2000,
    batch_size: int = 4
):
    """
    对 MPOTrainer 进行长时间的显存稳定性压力测试。
    """
    logger.info("--- Starting MPO Trainer Stress Test ---")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model, tokenizer = FastLanguageModel.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 适配典型的Transformer模型
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 2. 准备数据加载器 (DataLoader)
    # 为了能长时间运行，我们可以将少量数据重复多次
    stress_test_dataset = create_navigation_dataset(dummy_data * 500) # 将数据重复500次
    
    processed_dataset = stress_test_dataset.map(
        MPOTrainer.tokenize_navigation_sample,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 2048},
        remove_columns=stress_test_dataset.column_names
    )
    
    data_collator = DataCollatorForMPO(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        processed_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    # 3. 实例化我们自定义的 Trainer (但只使用它的 `compute_loss` 方法)
    # 注意：我们不需要完整的 TrainingArguments，因为我们是手动循环
    dummy_args = TrainingArguments(output_dir="./stress_test_dummy_output")
    trainer = MPOTrainer(model=model, processing_class=tokenizer, args=dummy_args)
    
    # 4. 手动训练循环和显存监控
    initial_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info(f"Initial CUDA memory allocated: {initial_memory_mb:.2f} MB")
    
    model.train()
    progress_bar = tqdm(total=num_test_steps, desc="Stress Test Steps")
    
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            if step >= num_test_steps:
                done = True
                break

            # 将输入移动到GPU
            inputs = trainer._prepare_inputs(batch)
            
            # --- 核心训练步骤 ---
            loss = trainer.compute_loss(model, inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # ---------------------

            # --- 显存监控 ---
            if (step + 1) % 100 == 0: # 每100步检查一次
                current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                logger.info(f"Step {step + 1}: CUDA memory allocated: {current_memory_mb:.2f} MB")
                
            step += 1
            progress_bar.update(1)
            
    progress_bar.close()
    
    final_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info("--- Stress Test Finished ---")
    logger.info(f"Initial Memory: {initial_memory_mb:.2f} MB")
    logger.info(f"Final Memory:   {final_memory_mb:.2f} MB")
    logger.info(f"Memory Growth:  {final_memory_mb - initial_memory_mb:.2f} MB over {num_test_steps} steps.")


if __name__ == "__main__":
    # 准备一小批有代表性的数据
    dummy_stress_data = [
        {
            "prompt": "This is a stress test prompt. Navigate from A to B.",
            "chosen": ["relation_1", "relation_2"],
            "rejected": ["relation_3", "relation_4", "relation_5"]
        }
    ] * 8
    
    run_stress_test(
        model_name="/mnt/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        dummy_data=dummy_stress_data,
        num_test_steps=2000,
        batch_size=4
    )