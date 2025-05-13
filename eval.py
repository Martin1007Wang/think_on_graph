import logging
import sys
from typing import List

from src.utils.qa_utils import eval_path_result_w_ans

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(prediction_paths: List[str]) -> None:
    """
    Evaluate results from multiple prediction files.

    Args:
        prediction_paths: A list of paths to the prediction JSONL files.
    """
    if not prediction_paths:
        logging.warning("No prediction paths provided.")
        return

    logging.info(f"Starting evaluation for {len(prediction_paths)} files...")

    for path in prediction_paths:
        logging.info(f"Evaluating file: {path}")
        try:
            # Assuming eval_path_result_w_ans prints results or returns them
            eval_path_result_w_ans(path)
            logging.info(f"Successfully evaluated: {path}")
        except FileNotFoundError:
            logging.error(f"Evaluation failed: File not found at {path}")
        except Exception as e:
            logging.error(f"Evaluation failed for {path}: {e}", exc_info=True) # Log traceback

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    # List of prediction files to evaluate
    paths_to_evaluate: List[str] = [
        "/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v10/RoG-webqsp/GCR-lora-dpo_v3_with_label-Llama-3.1-8B-Instruct-all-available-relations/deepseek-chat/iterative-rounds2-topk5/predictions.jsonl",
        # Add more paths here if needed
    ]
    main(paths_to_evaluate)