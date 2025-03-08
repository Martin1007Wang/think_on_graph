import os
import argparse
import tqdm
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from functools import partial
from src.utils.graph_utils import build_graph
from src.utils.mcts_utils import mcts_search

def process(sample, undirected=False):
    graph = build_graph(sample['graph'], undirected=undirected)
    start_nodes = sample['q_entity']
    answer_nodes = sample['a_entity']
    paths_list = mcts_search(graph, start_nodes, answer_nodes)
    sample['ground_truth_paths'] = paths_list
    
    return sample
    

def index_graph(args):
    input_file = os.path.join(args.data_path, args.d)
    data_path = f"{args.d}_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.output_path, data_path, args.split)
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    
    # dataset = dataset.map(process, num_proc=args.n, fn_kwargs={'K': args.K})
    # dataset.select_columns(['id', 'paths']).save_to_disk(output_dir)
    results = []
    with Pool(args.n) as p:
        for res in tqdm.tqdm(p.imap_unordered(partial(process, undirected = args.undirected), dataset), total=len(dataset)):
            results.append(res)
    
    index_dataset = Dataset.from_list(results)
    index_dataset.save_to_disk(output_dir)
        
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='rmanluo')
    argparser.add_argument('--d', '-d', type=str, default='RoG-webqsp')
    argparser.add_argument('--split', type=str, default='train')
    argparser.add_argument('--output_path', type=str, default='data/shortest_path_index')
    argparser.add_argument('--undirected', action='store_true', help='whether the graph is undirected')
    argparser.add_argument('--n', type=int, default=1, help='number of processes')
    
    args = argparser.parse_args()
    
    index_graph(args)