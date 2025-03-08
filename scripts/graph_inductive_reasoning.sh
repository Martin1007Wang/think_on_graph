export HF_ENDPOINT=https://hf-mirror.com
DATA_PATH=rmanluo
DATA_LIST="RoG-webqsp RoG-cwq"
SPLIT="test"

KGLLM_MODEL="GCR-Qwen2-0.5B-Instruct"
MODEL_NAME="deepseek-r1-7b"
N_THREAD=10

# MODEL_NAME=gpt-4o-mini
# N_THREAD=10

for DATA in ${DATA_LIST}; do
  REASONING_PATH="results/GenPaths/${DATA}/${KGLLM_MODEL}/test/zero-shot-group-beam-k10-index_len2/predictions.jsonl"

  python workflow/predict_final_answer.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --model_name ${MODEL_NAME} --reasoning_path ${REASONING_PATH} --add_path True -n ${N_THREAD}
done
