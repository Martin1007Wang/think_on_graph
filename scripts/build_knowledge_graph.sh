# #!/bin/bash
# DATASET="RoG-webqsp"
# SPLIT="train"
# NEO4J_URI="bolt://localhost:7687"
# NEO4J_USER="neo4j"
# NEO4J_PASSWORD="Martin1007Wang"
# MODEL_NAME="msmarco-distilbert-base-tas-b"
# CLEAR=True
# echo "Loading ${DATASET} ${SPLIT} into Neo4j..."
# python workflow/build_knowledge_graph.py \
#     --dataset ${DATASET} \
#     --split ${SPLIT} \
#     --neo4j_uri ${NEO4J_URI} \
#     --neo4j_user ${NEO4J_USER} \
#     --neo4j_password ${NEO4J_PASSWORD} \
#     --model_name ${MODEL_NAME} \
#     --clear ${CLEAR} 

#!/bin/bash
DATASET="RoG-webqsp"
SPLIT="test"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="Martin1007Wang"
MODEL_NAME="msmarco-distilbert-base-tas-b"
CLEAR=True
echo "Loading ${DATASET} ${SPLIT} into Neo4j..."
python workflow/build_knowledge_graph.py \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --neo4j_uri ${NEO4J_URI} \
    --neo4j_user ${NEO4J_USER} \
    --neo4j_password ${NEO4J_PASSWORD} \
    --model_name ${MODEL_NAME} \
    --clear ${CLEAR} 