import os
import json
import argparse
import tqdm
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from functools import partial
from src.utils.graph_utils import build_graph, dfs, bfs

def process(sample, K, undirected, strategy):
    graph = build_graph(sample['graph'], undirected=undirected)
    start_nodes = sample['q_entity']
    if strategy == 'dfs':
        paths_list, stats = dfs(graph, start_nodes, K)
    elif strategy == 'bfs':
        paths_list, stats = bfs(graph, start_nodes, K)
    sample['ground_truth_paths'] = paths_list
    sample['path_stats'] = stats
    return sample

def index_graph(args):
    input_file = os.path.join(args.data_path, args.d)
    data_path = f"{args.d}_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.output_path, data_path, args.split, f"length-{args.K}")

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)

    results = []
    path_stats_summary = {
        'total_samples': 0,
        'path_length_distribution': {},
        'avg_paths_per_sample': 0,
        'total_paths': 0
    }
    with Pool(args.n) as p:
        for res in tqdm.tqdm(p.imap_unordered(partial(process, K=args.K, undirected=args.undirected, strategy=args.strategy), dataset), total=len(dataset)):
            results.append(res)
            stats = res['path_stats']
            path_stats_summary['total_samples'] += 1
            
            
            path_stats_summary['total_paths'] += stats['total_paths']
            for length in stats['path_lengths']:
                path_stats_summary['path_length_distribution'][length] = \
                    path_stats_summary['path_length_distribution'].get(length, 0) + 1
    
    path_stats_summary['avg_paths_per_sample'] = \
        path_stats_summary['total_paths'] / path_stats_summary['total_samples']
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'path_stats.json'), 'w') as f:
        json.dump(path_stats_summary, f, indent=2)
    
    index_dataset = Dataset.from_list(results)
    index_dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='rmanluo')
    argparser.add_argument('--d', '-d', type=str, default='RoG-webqsp')
    argparser.add_argument('--split', type=str, default='test')
    argparser.add_argument('--output_path', type=str, default='data/graph_index')
    argparser.add_argument('--n', type=int, default=1, help='Number of processes')
    argparser.add_argument('--K', type=int, default=2, help="Maximum length of paths")
    argparser.add_argument('--undirected', action='store_true', help='Whether the graph is undirected')
    argparser.add_argument('--strategy', type=str, default='dfs', help='Strategy to build graph index')

    args = argparser.parse_args()
    index_graph(args)