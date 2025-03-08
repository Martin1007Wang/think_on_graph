# For training
DATA_PATH="RoG-webqsp RoG-cwq"
OUTPUT_PATH="data/shortest_path_index_with_stats"
SPLIT=train
N_PROCESS=16
for DATA_PATH in ${DATA_PATH}; do
  python workflow/build_shortest_path_index.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS} --output_path ${OUTPUT_PATH}
done