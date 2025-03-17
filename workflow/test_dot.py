import torch
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple, Dict, Set

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MockKnowledgeGraph:
    def __init__(self):
        # 初始化模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 模拟关系类型和嵌入
        self.relation_types = [
            'sports.sports_team_roster.player',  # 目标关系
            'sports.pro_athlete.teams',
            'people.person.nationality',
            'base.bioventurist.indication.disease',
            'architecture.ownership.structure',
            'film.performance.special_performance_type'
        ]
        self.id_2_relation = {i: rel for i, rel in enumerate(self.relation_types)}
        self.relation_2_id = {rel: i for i, rel in enumerate(self.relation_types)}
        
        # 生成并归一化关系嵌入
        self.relation_embeddings = self.model.encode(
            self.relation_types, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        self.relation_embeddings = self.relation_embeddings / torch.norm(
            self.relation_embeddings, dim=1, keepdim=True
        )

        # 模拟数据库中的实体和关系
        self.entity_relations = {
            'some_athlete': {
                'sports.sports_team_roster.player',
                'sports.pro_athlete.teams',
                'people.person.nationality'
            }
        }

    def get_related_relations_by_question(self, entity_id: str, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k related relations for an entity based on question similarity."""
        # 编码问题
        question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False)
        if question_emb.dim() > 1:
            question_emb = question_emb.squeeze(0)
        question_emb = question_emb / torch.norm(question_emb)  # 归一化
        
        # 模拟数据库查询
        related_relations = self.entity_relations.get(entity_id, set())
        logger.debug(f"Related relations for {entity_id}: {related_relations}")
        if not related_relations:
            return []
        
        # 映射到索引
        rel_indices = [self.relation_2_id[rel] for rel in related_relations if rel in self.relation_2_id]
        logger.debug(f"Relation indices: {rel_indices}")
        
        # 提取嵌入
        related_embs = self.relation_embeddings[rel_indices].to(question_emb.device)
        similarities = self.model.similarity(question_emb, related_embs)
        if similarities.dim() > 1:
            similarities = similarities.squeeze(0)
        logger.debug(f"Similarities: {[(self.id_2_relation[rel_indices[i]], s.item()) for i, s in enumerate(similarities)]}")
        
        # 选择 Top-K
        scores, indices = torch.topk(similarities, k=min(top_k, len(similarities)))
        scores = scores.cpu().tolist()
        indices = indices.cpu().tolist()
        
        result = [(self.id_2_relation[rel_indices[idx]], score) for score, idx in zip(scores, indices)]
        logger.debug(f"Top {top_k} results: {result}")
        return result

# 测试函数
def test_get_related_relations():
    kg = MockKnowledgeGraph()
    
    # 测试用例 1：问题是一个关系类型
    entity_id = 'some_athlete'
    question = 'sports.sports_team_roster.player'
    top_k = 5
    print(f"\nTest 1: entity_id={entity_id}, question={question}, top_k={top_k}")
    result = kg.get_related_relations_by_question(entity_id, question, top_k=top_k)
    print(f"Result: {result}")
    
    # 测试用例 2：问题是一个自然语言问句
    question = "Which player is on the team roster?"
    print(f"\nTest 2: entity_id={entity_id}, question={question}, top_k={top_k}")
    result = kg.get_related_relations_by_question(entity_id, question, top_k=top_k)
    print(f"Result: {result}")

if __name__ == "__main__":
    test_get_related_relations()