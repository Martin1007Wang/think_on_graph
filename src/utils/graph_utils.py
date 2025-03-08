import networkx as nx
from collections import deque
from typing import List, Tuple

def build_graph(graph: list, undirected = False) -> nx.DiGraph | nx.Graph:
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G

def dfs(graph, start_node_list, max_length):
    stats = {
        'path_lengths': [],
        'valid_entities': 0,
        'total_paths': 0
    }
    valid_entities = [n for n in start_node_list if n in graph]
    stats['valid_entities'] = len(valid_entities)
    if not valid_entities:
        return [], stats
    def dfs_visit(node: str, path: List[Tuple]) -> None:
        if len(path) > max_length:
            return
        if path:
            path_lists.add(tuple(path))
            # 记录路径长度
            stats['path_lengths'].append(len(path))
        for neighbor in graph.neighbors(node):
            rel = graph[node][neighbor]["relation"]
            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_length:
                dfs_visit(neighbor, new_path)
            
    path_lists = set()
    for start_node in start_node_list:
        if start_node in graph:
            dfs_visit(start_node, [])
    stats['total_paths'] = len(path_lists)
    return list(path_lists), stats

def bfs(graph, start_node_list, max_length):
    stats = {
        'path_lengths': [],
        'valid_entities': 0,
        'total_paths': 0
    }
    valid_entities = [n for n in start_node_list if n in graph]
    stats['valid_entities'] = len(valid_entities)
    if not valid_entities:
        return [], stats

    path_lists = set()
    queue = deque()
    
    for start_node in start_node_list:
        if start_node in graph:
            queue.append((start_node, []))
    
    while queue:
        current_node, current_path = queue.popleft()
        
        if current_path:
            path_lists.add(tuple(current_path))
            # 记录路径长度
            stats['path_lengths'].append(len(current_path))
        
        if len(current_path) >= max_length:
            continue
            
        for neighbor in graph.neighbors(current_node):
            rel = graph[current_node][neighbor]["relation"]
            new_path = current_path + [(current_node, rel, neighbor)]
            queue.append((neighbor, new_path))

    stats['total_paths'] = len(path_lists)
    return list(path_lists), stats

# 定义一个函数来进行宽度优先搜索
def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])  # 使用队列存储待探索节点和对应路径
    while queue:
        current_node, current_path = queue.popleft()

        # 如果当前路径符合规则，将其添加到结果列表中
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            # if len(result_paths) >= max_p:
            #     break

        # 如果当前路径长度小于规则长度，继续探索
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                # 剪枝：如果当前边类型与规则中的对应位置不匹配，不继续探索该路径
                rel = graph[current_node][neighbor]["relation"]
                if rel != target_rule[len(current_path)] or len(current_path) > len(
                    target_rule
                ):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))

    return result_paths


def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> tuple[list, dict]:
    """
    Get shortest paths connecting question and answer entities.
    
    Returns:
        tuple: (paths_list, stats_dict)
        - paths_list: List of paths, where each path is a list of (entity, relation, entity) tuples
        - stats_dict: Dictionary containing path statistics
    """
    # 初始化统计信息
    stats = {
        'has_path': False,
        'path_lengths': [],
        'valid_q_entities': 0,
        'valid_a_entities': 0,
        'total_paths': 0
    }
    
    # 过滤不在图中的实体
    valid_q_entities = [h for h in q_entity if h in graph]
    valid_a_entities = [t for t in a_entity if t in graph]
    
    stats['valid_q_entities'] = len(valid_q_entities)
    stats['valid_a_entities'] = len(valid_a_entities)
    
    if not valid_q_entities or not valid_a_entities:
        return [], stats
    
    result_paths = []
    edge_relations = {}
    
    for h in valid_q_entities:
        for t in valid_a_entities:
            try:
                shortest_paths = nx.all_shortest_paths(graph, h, t)
                stats['has_path'] = True
                for path in shortest_paths:
                    path_with_relations = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_key = (u, v)
                        if edge_key not in edge_relations:
                            edge_relations[edge_key] = graph[u][v]["relation"]
                        path_with_relations.append((u, edge_relations[edge_key], v))
                    result_paths.append(path_with_relations)
                    stats['path_lengths'].append(len(path) - 1)  # 记录跳数
                    
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                print(f"Error finding path from {h} to {t}: {str(e)}")
                continue
    
    stats['total_paths'] = len(result_paths)
    return result_paths, stats


def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    """
    Get all simple paths connecting question and answer entities within given hop
    """
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]["relation"], e[1]) for e in p])
    return result_paths


def get_negative_paths(
    q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2
) -> list:
    """
    Get negative paths for question witin hop
    """
    import walker

    # sample paths
    start_nodes = []
    end_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    for t in a_entity:
        if t in graph:
            end_nodes.append(node_idx.index(t))
    paths = walker.random_walks(
        graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes, verbose=False
    )
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        # remove paths that end with answer entity
        if p[-1] in end_nodes:
            continue
        for i in range(len(p) - 1):
            u = node_idx[p[i]]
            v = node_idx[p[i + 1]]
            tmp.append((u, graph[u][v]["relation"], v))
        result_paths.append(tmp)
    return result_paths


def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2) -> tuple[list, list]:
    """
    Get negative paths for question witin hop
    """
    import walker

    # sample paths
    start_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    paths = walker.random_walks(
        graph, n_walks=n, walk_len=hop, start_nodes=start_nodes, verbose=False
    )
    # Add relation to paths
    result_paths = []
    rules = []
    for p in paths:
        tmp = []
        tmp_rule = []
        for i in range(len(p) - 1):
            u = node_idx[p[i]]
            v = node_idx[p[i + 1]]
            tmp.append((u, graph[u][v]["relation"], v))
            tmp_rule.append(graph[u][v]["relation"])
        result_paths.append(tmp)
        rules.append(tmp_rule)
    return result_paths, rules
