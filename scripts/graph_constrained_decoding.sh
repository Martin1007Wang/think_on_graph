#!/bin/bash

set -e
set -u

export BASE_PROJECT_DIR="/mnt/wangjingxiong/think_on_graph"
export PYTHONPATH="${BASE_PROJECT_DIR}:${PYTHONPATH:-}" # Prepend to PYTHONPATH safely

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="0"

export NEO4J_PASSWORD="Martin1007Wang"
export SILICONFLOW_API_KEY="sk-sflamkrssqdvcndtqsvobqfqugyfkqejrprynrjvtpslagae"

if [ -z "${NEO4J_PASSWORD:-}" ]; then
    echo "ERROR: NEO4J_PASSWORD environment variable is not set." >&2
    exit 1
fi
if [ -z "${SILICONFLOW_API_KEY:-}" ]; then
    echo "ERROR: SILICONFLOW_API_KEY environment variable is not set." >&2
    exit 1
fi

# --- Configuration Section ---
DATA_PATH="rmanluo"
DATA_NAME="RoG-webqsp"
SPLIT="test"
PREDICT_PATH="results/all_result"

EXPLORE_MODEL_NAME="mpo"
EXPLORE_MODEL_PATH="/mnt/wangjingxiong/think_on_graph/mpo_models/cand_pn_only_pos_shortest_paths/hf_dataset/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec_ep1_loss-mpo_b0.1_sft0.05_lr1e-5_lora-r8-a16_bf16/merged_model"
# EXPLORE_MODEL_NAME="deepseek-chat"
# EXPLORE_MODEL_PATH=""

DTYPE="bf16"
QUANT="4bit" 
ATTN_IMPLEMENTATION="flash_attention_2"
MAX_NEW_TOKENS=512

PREDICT_MODEL_NAME="deepseek-chat"
PREDICT_MODEL_PATH=""
RETRY=50
API_MODEL_NAME="deepseek-ai/DeepSeek-V3"

MAX_ROUNDS=2
MAX_SELECTION_COUNT=3
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"

# --- Script & Path Validation ---
PYTHON_SCRIPT_PATH="${BASE_PROJECT_DIR}/workflow/predict_paths_and_answers.py"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found at '$PYTHON_SCRIPT_PATH'" >&2
    exit 1
fi

CMD=(
    "python"
    "$PYTHON_SCRIPT_PATH"
    "--data_path" "$DATA_PATH"
    "--data_name" "$DATA_NAME"
    "--split" "$SPLIT"
    "--predict_path" "$PREDICT_PATH"
    
    # 探索模型参数
    "--explore_model_name" "$EXPLORE_MODEL_NAME"
    "--explore_model_path" "$EXPLORE_MODEL_PATH"

    # "--dtype" "$DTYPE"
    # "--quant" "$QUANT"
    # "--attn_implementation" "$ATTN_IMPLEMENTATION"
    # "--max_new_tokens" "$MAX_NEW_TOKENS"

    "--predict_model_name" "$PREDICT_MODEL_NAME"
    
    "--retry" "$RETRY"
    "--api_model_name" "$API_MODEL_NAME"
    "--max_rounds" "$MAX_ROUNDS"
    "--max_selection_count" "$MAX_SELECTION_COUNT"
    "--neo4j_uri" "$NEO4J_URI"
    "--neo4j_user" "$NEO4J_USER"
    "--neo4j_password" "$NEO4J_PASSWORD"
    "--debug"
    "--target_ids" ""
)
if [ -n "$PREDICT_MODEL_PATH" ]; then
    CMD+=("--predict_model_path" "$PREDICT_MODEL_PATH")
fi

# --- Execution ---
echo "-----------------------------------------------------"
echo "Starting Knowledge Graph Reasoning Experiment..."
echo "Timestamp: $(date)"
echo "Explorer Model: ${EXPLORE_MODEL_NAME} (${EXPLORE_MODEL_PATH})"
echo "Predictor Model: ${PREDICT_MODEL_NAME}"
echo "-----------------------------------------------------"
echo "Executing command:"
printf "%q " "${CMD[@]}"
echo

# 执行命令
"${CMD[@]}"

EXIT_CODE=$?
echo "-----------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "All tasks completed successfully! Results saved in ${PREDICT_PATH}"
else
    echo "Script FAILED with exit code $EXIT_CODE."
fi
echo "-----------------------------------------------------"

exit $EXIT_CODE
