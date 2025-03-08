from collections import deque
import networkx as nx
from typing import List, Tuple

def dfs(graph: nx.DiGraph, start_node_list: List[str], max_length: int) -> List[List[Tuple]]:
    """
    使用DFS查找从起始节点列表出发的所有长度不超过max_length的路径
    
    Args:
        graph: 有向图
        start_node_list: 起始节点列表
        max_length: 最大路径长度

    Returns:
        List[List[Tuple]]: 路径列表，每个路径是(entity, relation, entity)元组的列表
    """
    def dfs_visit(node: str, path: List[Tuple]) -> None:
        if len(path) > max_length:
            return
            
        if path:  # 只添加非空路径
            path_lists.add(tuple(path))
        
        for neighbor in graph.neighbors(node):
            rel = graph[node][neighbor]["relation"]
            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_length:
                dfs_visit(neighbor, new_path)
            
    path_lists = set()
    for start_node in start_node_list:
        if start_node in graph:
            dfs_visit(start_node, [])
            
    return list(path_lists)

def bfs(graph: nx.DiGraph, start_node_list: List[str], max_length: int) -> List[List[Tuple]]:
    """
    使用BFS查找从起始节点列表出发的所有长度不超过max_length的路径
    
    Args:
        graph: 有向图
        start_node_list: 起始节点列表
        max_length: 最大路径长度

    Returns:
        List[List[Tuple]]: 路径列表，每个路径是(entity, relation, entity)元组的列表
    """
    path_lists = set()
    queue = deque()
    
    # 添加所有有效的起始节点
    for start_node in start_node_list:
        if start_node in graph:
            queue.append((start_node, []))
    
    while queue:
        current_node, current_path = queue.popleft()
        
        if current_path:  # 只添加非空路径
            path_lists.add(tuple(current_path))
        
        if len(current_path) >= max_length:
            continue
            
        for neighbor in graph.neighbors(current_node):
            rel = graph[current_node][neighbor]["relation"]
            new_path = current_path + [(current_node, rel, neighbor)]
            queue.append((neighbor, new_path))

    return list(path_lists)

def test_path_search():
    """
    测试DFS和BFS的路径搜索功能
    """
    # 创建一个简单的测试图
    G = nx.DiGraph()
    # 添加边和关系
    edges = [
        ("A", "B", {"relation": "r1"}),
        ("B", "C", {"relation": "r2"}),
        ("A", "C", {"relation": "r3"}),
        ("C", "D", {"relation": "r4"}),
    ]
    G.add_edges_from(edges)
    
    # 测试用例
    start_nodes = ["A"]
    max_length = 2
    
    # 运行两种搜索算法
    dfs_paths = dfs(G, start_nodes, max_length)
    bfs_paths = bfs(G, start_nodes, max_length)
    
    # 将结果转换为集合以便比较
    dfs_paths_set = set(map(tuple, dfs_paths))
    bfs_paths_set = set(map(tuple, bfs_paths))
    
    # 验证结果
    expected_paths = {
        (("A", "r1", "B"),),
        (("A", "r3", "C"),),
        (("A", "r1", "B"), ("B", "r2", "C")),
    }
    
    print("测试结果：")
    print(f"DFS找到的路径数量: {len(dfs_paths)}")
    print(f"BFS找到的路径数量: {len(bfs_paths)}")
    print(f"DFS和BFS结果是否相同: {dfs_paths_set == bfs_paths_set}")
    print(f"结果是否符合预期: {dfs_paths_set == expected_paths and bfs_paths_set == expected_paths}")
    
    # 打印所有路径
    print("\nDFS找到的路径:")
    for path in sorted(dfs_paths):
        print(path)
        
    print("\nBFS找到的路径:")
    for path in sorted(bfs_paths):
        print(path)

if __name__ == "__main__":
    test_path_search()