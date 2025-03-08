# For evaluation
DATA_PATH="RoG-webqsp RoG-cwq"
OUTPUT_PATH="data/graph_index_bfs"
STRATEGY="bfs"
SPLIT=test
N_PROCESS=16
HOP=2 # 3
for DATA_PATH in ${DATA_PATH}; do
    python workflow/build_graph_index.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS} --K ${HOP} --output_path ${OUTPUT_PATH} --strategy ${STRATEGY}
done