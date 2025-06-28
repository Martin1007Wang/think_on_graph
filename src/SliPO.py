# 在您的腳本文件中定義這個新的DataCollator
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class SLiPODataCollator:
    """
    專為S-LiPO設計的Data Collator。
    它處理一個包含 'prompt'、'responses' (str列表) 和 'labels' (int列表) 的批次。
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [f["prompt"] for f in features]
        list_of_responses = [f["responses"] for f in features]
        list_of_labels = [f["labels"] for f in features]
        
        # Tokenize prompts
        prompt_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize all responses
        # 注意：這裡我們將一個batch中所有的response拉平處理，以提高效率
        flat_responses = [resp for sublist in list_of_responses for resp in sublist]
        response_inputs = self.tokenizer(
            flat_responses,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt"
        )
        
        batch = {
            "prompt_input_ids": prompt_inputs["input_ids"],
            "prompt_attention_mask": prompt_inputs["attention_mask"],
            "responses_input_ids": response_inputs["input_ids"],
            "responses_attention_mask": response_inputs["attention_mask"],
            "labels": torch.tensor(list_of_labels, dtype=torch.float32), # 直接傳遞標籤
        }
        return batch
    
class SLiPOTrainer(DPOTrainer):
    def __init__(self, *args, margin: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # 為S-LiPO增加一個margin超參數
        self.margin = margin

    def s_lipo_loss(
        self,
        policy_logps: torch.FloatTensor,
        ref_logps: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        計算S-LiPO的Listwise Margin Loss。
        
        Args:
            policy_logps: 當前策略模型對列表中所有回覆的log probabilities。Shape: (batch_size, K)
            ref_logps: 參考模型對列表中所有回覆的log probabilities。Shape: (batch_size, K)
            labels: 每個回覆的二元標籤 (1 for positive, 0 for negative)。Shape: (batch_size, K)
        
        Returns:
            平均到每個樣本的損失值。
        """
        # 根據公式(2)計算隱式獎勵分數 s
        pi_logratios = policy_logps - ref_logps
        rewards = self.beta * pi_logratios

        losses = []
        # 遍歷batch中的每一個樣本
        for i in range(rewards.shape[0]):
            sample_rewards = rewards[i]
            sample_labels = labels[i]
            
            # 找出正例和負例的分數
            pos_indices = (sample_labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (sample_labels == 0).nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_rewards = sample_rewards[pos_indices]
            neg_rewards = sample_rewards[neg_indices]

            # 計算所有 (pos, neg) 對的 margin loss
            # 使用廣播機制 (broadcasting) 來高效計算
            # (len_pos, 1) - (1, len_neg) -> (len_pos, len_neg)
            margin_matrix = pos_rewards.unsqueeze(1) - neg_rewards.unsqueeze(0)
            loss_matrix = torch.relu(self.margin - margin_matrix)
            
            # 對一個樣本的所有配對損失求平均
            losses.append(loss_matrix.mean())
        
        if not losses:
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
            
        return torch.stack(losses).mean()

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        重寫此函數以計算S-LiPO損失。
        """
        # 由於數據格式完全不同，我們需要一個全新的前向傳播和損失計算邏輯
        # 注意：這是一個簡化實現，實際應用中需要更仔細地處理padding和mask
        
        # 1. 獲取所有回覆的logps
        # 這需要一個新的前向函數，這裡我們簡化表示其邏輯
        # concatenated_forward 不再適用，因為它只處理 chosen/rejected
        
        # 假設我們已經通過一個新的listwise_forward得到了logps
        # policy_output = self.listwise_forward(model, batch) 
        # ref_output = self.listwise_forward(self.ref_model, batch)
        # policy_logps = policy_output["logps"] # Shape: (batch_size, K)
        # ref_logps = ref_output["logps"]       # Shape: (batch_size, K)
        
        # ---------------------------------------------------------------------
        # 為了演示，我們在這裡進行一個模擬的計算，實際中需要完成上面的listwise_forward
        # 這裡的實現細節比較複雜，是修改的核心難點
        # 你需要自己實現一個方法，對batch裡的所有responses計算log probabilities
        # 此處省略了`listwise_forward`的具體實現，因為它高度依賴於模型架構和DataCollator
        # 但它的目標是產出下面兩個張量：
        # policy_logps = ... 
        # ref_logps = ...
        # ---------------------------------------------------------------------
        
        # 由於無法在不極大修改內部代碼的情況下實現 `listwise_forward`,
        # 我們回退到那個更務實的結論：繼承`DPOTrainer`來實現S-LiPO非常困難，
        # 因為它的整個數據管道和前向傳播都是為`pair`設計的。
        
        # ---- 正確的結論 ----
        # 讓我們回到最初的分析。您看了源碼，問在哪裡修改。
        # 現在您應該能更深刻地理解，為什麼我一開始推薦的是「數據轉換」的方法。
        # 因為要真正地、原生底層地修改，您需要重寫的不只是`dpo_loss`，
        # 而是整個數據處理到前向傳播的鏈路 (`_prepare_dataset`, `DataCollator`, 
        # `concatenated_forward`, `get_batch_loss_metrics`)。
        # 這幾乎等同於從`transformers.Trainer`重新寫一個新的`ListwiseTrainer`。
        
        # ---- 實際可行的修改路徑 ----
        # 如果真的要繼承和修改，最可行的路徑是：
        # 1. 繼承 `Trainer` 而不是 `DPOTrainer`。
        # 2. 自己實現數據整理、前向傳播和損失計算。
        # 這超出了「修改」的範疇，變成了「創建」。
        
        # 讓我們用一個模擬的輸出來完成這個函數的邏輯，以展示損失如何被調用
        # 假設我們神奇地得到了logps
        batch_size, K = batch["labels"].shape
        # 模擬輸出
        policy_logps = torch.randn(batch_size, K, device=self.accelerator.device)
        ref_logps = torch.randn(batch_size, K, device=self.accelerator.device)
        
        # 2. 計算S-LiPO損失
        loss = self.s_lipo_loss(policy_logps, ref_logps, batch["labels"])
        
        # 3. 計算指標 (metrics)
        metrics = {}
        # ... (可以計算例如正例的平均分、負例的平均分、準確率等)
        
        return loss, metrics