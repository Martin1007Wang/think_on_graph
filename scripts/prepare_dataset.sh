#!/bin/bash
DATA_PATH="rmanluo/RoG-webqsp"
DATASET_NAME="RoG-webqsp"
SPLIT="train"
OUTPUT_PATH="data/processed"
OUTPUT_NAME=${DATA_PATH}_${SPLIT}

NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang"

MAX_PATH_LENGTH=3
TOP_K_RELATIONS=5
MODEL_NAME="msmarco-distilbert-base-tas-b"
MAX_PAIRS=5
MAX_NEGATIVES_PER_PAIR=5

python workflow/prepare_dataset.py \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET_NAME}\
    --split ${SPLIT} \
    --output_path ${OUTPUT_PATH} \
    --output_name ${OUTPUT_NAME} \
    --neo4j_uri ${NEO4J_URI} \
    --neo4j_user ${NEO4J_USER} \
    --neo4j_password ${NEO4J_PASSWORD} \
    --max_path_length ${MAX_PATH_LENGTH} \
    --top_k_relations ${TOP_K_RELATIONS} \
    --max_pairs ${MAX_PAIRS} \
    --max_negatives_per_pair ${MAX_NEGATIVES_PER_PAIR}\
    --model_name ${MODEL_NAME} \
