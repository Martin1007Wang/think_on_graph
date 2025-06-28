#!/bin/bash

DATASET_NAMES=("RoG-cwq" "RoG-webqsp")
DATA_SPLITS=("test" "test")

NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang"
CLEAR_DB="True"

echo "Loading datasets into Neo4j..."
echo "Datasets: ${DATASET_NAMES[@]}"
echo "Splits: ${DATA_SPLITS[@]}"

python workflow/build_knowledge_graph.py \
    --dataset_names "${DATASET_NAMES[@]}" \
    --splits "${DATA_SPLITS[@]}" \
    --neo4j_uri "${NEO4J_URI}" \
    --neo4j_user "${NEO4J_USER}" \
    --neo4j_password "${NEO4J_PASSWORD}" \
    --clear "${CLEAR_DB}" \

echo "Script execution finished."