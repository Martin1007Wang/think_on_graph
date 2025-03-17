import os
import json
import argparse
import tqdm
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from functools import partial
from src.utils.graph_utils import build_graph, get_truth_paths

def process(sample, undirected=False):
    graph = build_graph(sample['graph'], undirected=undirected)
    start_nodes = sample['q_entity']
    answer_nodes = sample['a_entity']
    paths_list, stats = get_truth_paths(start_nodes, answer_nodes, graph)
    sample['ground_truth_paths'] = paths_list
    sample['path_stats'] = stats
    return sample

def index_graph(args):
    input_file = os.path.join(args.data_path, args.d)
    data_path = f"{args.d}_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.output_path, data_path, args.split)
    
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)[:100]
    
    results = []
    path_stats_summary = {
        'total_samples': 0,
        'samples_with_path': 0,
        'path_length_distribution': {},
        'avg_paths_per_sample': 0,
        'total_paths': 0
    }
    
    with Pool(args.n) as p:
        for res in tqdm.tqdm(p.imap_unordered(partial(process, undirected=args.undirected), dataset), total=len(dataset)):
            results.append(res)
            stats = res['path_stats']
            path_stats_summary['total_samples'] += 1
            
            if stats['has_path']:
                path_stats_summary['samples_with_path'] += 1
            
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
    with open(os.path.join(output_dir, 'index_dataset.json'), 'w') as f:
        index_dataset.to_json(f)

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