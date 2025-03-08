export HF_ENDPOINT=https://hf-mirror.com
# For training
DATA_PATH="RoG-cwq"
SPLIT=train
OUTPUT_PATH="data/shortest_path_index_by_mcts"
N_PROCESS=8
for DATA_PATH in ${DATA_PATH}; do
  python workflow/build_shortest_path_index_by_mcts.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS} --output_path ${OUTPUT_PATH}
done

# For evaluation

# DATA_PATH="RoG-webqsp RoG-cwq"
# SPLIT=test
# N_PROCESS=8
# HOP=2 # 3
# for DATA_PATH in ${DATA_PATH}; do
#     python workflow/build_graph_index.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS} --K ${HOP}
# done