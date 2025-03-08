import os
from sentence_transformers import SentenceTransformer, util
import torch
import dotenv
from workflow.build_knowledge_graph import KnowledgeGraph
from typing import List, Set, Dict, Tuple
from collections import deque
import warnings
from tqdm import tqdm
from functools import lru_cache

# Load environment variables
dotenv.load_dotenv()

def preprocess_relation(relation):
    """Convert relation format to natural language"""
    # Remove underscores and convert to lowercase
    return relation.replace('_', ' ').lower()

@lru_cache(maxsize=1000)
def encode_text(text: str, model: SentenceTransformer) -> torch.Tensor:
    """缓存文本编码结果"""
    return model.encode(text, convert_to_tensor=True)

def get_top_similar_relations(question, relations, model, top_k=5):
    """Get top-k similar relations for a given question"""
    if not relations:
        return []
        
    # Preprocess relations
    processed_relations = [preprocess_relation(rel) for rel in relations]
    
    # Encode question and relations with caching
    question_embedding = encode_text(question, model)
    relation_embeddings = torch.stack([encode_text(rel, model) for rel in processed_relations])
    
    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(question_embedding, relation_embeddings)[0]
    
    # Get top-k similar relations
    top_results = torch.topk(cosine_scores, min(top_k, len(relations)))
    
    return [(relations[idx], score.item()) for score, idx in zip(top_results.values, top_results.indices)]

def find_path_by_similarity(
    kg: KnowledgeGraph,
    start_entity: str,
    target_entity: str,
    question: str,
    model: SentenceTransformer,
    max_depth: int = 3,
    top_k: int = 5
) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
    """
    使用相似度搜索找到从起始实体到目标实体的路径
    返回：(找到的路径, 访问过的节点集合)
    """
    visited = set()
    path = []
    queue = deque([(start_entity, [], set())])
    pbar = tqdm(total=max_depth, desc="Searching path")
    current_depth = 0
    
    while queue and len(path) == 0:
        current_entity, current_path, current_visited = queue.popleft()
        
        # Update progress bar
        new_depth = len(current_path)
        if new_depth > current_depth:
            pbar.update(new_depth - current_depth)
            current_depth = new_depth
        
        if len(current_path) >= max_depth:
            continue
            
        # 获取当前实体的所有关系
        relations = kg.get_connected_relations(current_entity)
        top_relations = get_top_similar_relations(question, relations, model, top_k)
        
        for relation, similarity_score in top_relations:
            # 获取通过该关系连接的所有尾实体
            tail_entities = kg.get_target_entities(current_entity, relation)
            
            for tail_entity in tail_entities:
                if tail_entity in current_visited:
                    continue
                    
                new_path = current_path + [(current_entity, relation, tail_entity)]
                new_visited = current_visited | {tail_entity}
                
                if tail_entity == target_entity:
                    path = new_path
                    visited = new_visited
                    break
                
                queue.append((tail_entity, new_path, new_visited))
                
            if path:  # 如果找到路径就退出循环
                break
    
    pbar.close()            
    return path, visited

def format_path(path: List[Tuple[str, str, str]]) -> str:
    """格式化路径为易读的字符串"""
    if not path:
        return "No path found"
        
    result = []
    for head, relation, tail in path:
        result.append(f"{head} --[{relation}]--> {tail}")
    return "\n".join(result)

def convert_path_format(path_dicts: List[Dict]) -> List[Tuple[str, str, str]]:
    """将Neo4j路径格式转换为我们使用的格式"""
    return [(d['source'], d['relation'], d['target']) for d in path_dicts]

def main():
    # Initialize the SBERT model with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    
    # Initialize Knowledge Graph
    kg = KnowledgeGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="Martin1007Wang"
    )
    
    try:
        # Test question and entities
        question = "who does brandon dubinsky play for"
        start_entity = "Brandon Dubinsky"
        target_entity = "Columbus Blue Jackets"
        
        print(f"\nSearching path from '{start_entity}' to '{target_entity}'...")
        print("\n1. Path found using similarity-based search:")
        
        # 使用相似度搜索找路径
        similarity_path, visited = find_path_by_similarity(
            kg, start_entity, target_entity, question, model
        )
        print(format_path(similarity_path))
        print(f"\nNodes visited during similarity search: {len(visited)}")
        
        print("\n2. Shortest path from Neo4j:")
        # 获取Neo4j的最短路径
        shortest_paths = kg.get_shortest_paths(start_entity, target_entity)
        if shortest_paths:
            shortest_path = convert_path_format(shortest_paths[0])  # 取第一条最短路径
        else:
            shortest_path = []
        print(format_path(shortest_path))
        
        # 比较两种路径
        print("\n3. Path Comparison:")
        print(f"Similarity path length: {len(similarity_path)}")
        print(f"Shortest path length: {len(shortest_path)}")
        
        if similarity_path == shortest_path:
            print("The paths are identical!")
        else:
            print("The paths are different.")
            if len(similarity_path) > 0 and len(shortest_path) > 0:
                print("\nRelations used:")
                print("Similarity path:", [rel for _, rel, _ in similarity_path])
                print("Shortest path:", [rel for _, rel, _ in shortest_path])
            
    finally:
        kg.close()

if __name__ == "__main__":
    main() 