import logging
from typing import Optional, List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.knowledge_graph import KnowledgeGraph
from src.model_interface import ModelInterface
from src.path_manager import PathManager

from src.utils.utils import error_handler
from src.utils.data_utils import EntityExpansion, EntityRelation, ExplorationRound

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class EntityExpander:
    def __init__(self, kg: KnowledgeGraph, model_interface: ModelInterface, 
                path_manager: PathManager, max_k_relations: int = 10):
        self.kg = kg
        self.model = model_interface
        self.path_manager = path_manager
        self.max_k_relations = max_k_relations
    
    def fetch_relations(self, node: str) -> Optional[List[str]]:
        try:
            relations = self.kg.get_related_relations(node, "out")
            return relations
        except Exception as e:
            logger.error(f"Error getting relations for node '{node}': {e}")
            return None

    def expand_entity(self, entity: str, question: str, context: str, history: str) -> Optional[EntityExpansion]:
        all_relations = self.fetch_relations(entity)
        selected_relations = self.model.select_relations(
            entity, all_relations, question, context, history, self.max_k_relations
        )
        if not selected_relations:
            return None

        expansion = EntityExpansion(entity=entity)
        
        # 处理每个选定的关系
        for relation in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, relation, "out")
                if not targets:
                    continue

                # 分离普通实体和编码实体
                named_entities, coded_entities = self._separate_entities(targets)

                # 处理普通实体
                if named_entities:
                    expansion.relations.append(EntityRelation(relation=relation, targets=named_entities))

                # 处理编码实体
                if coded_entities:
                    self._expand_coded_entities(entity, relation, coded_entities, expansion, question, context, history)
                    
            except Exception as e:
                logger.error(
                    f"Error expanding relation '{relation}' for entity '{entity}': {e}",
                    exc_info=True
                )
                
        return expansion if expansion.relations else None
    
    def _separate_entities(self, entities: List[str]) -> Tuple[List[str], List[str]]:
        is_coded = self.path_manager.is_coded_entity
        named_entities = [e for e in entities if isinstance(e, str) and not is_coded(e)]
        coded_entities = [e for e in entities if isinstance(e, str) and is_coded(e)]
        return named_entities, coded_entities
    
    def _expand_coded_entities(self, source_entity: str, source_relation: str, 
                              coded_entities: List[str], expansion: EntityExpansion,
                              question: str, context: str, history: str) -> None:
        # 并行获取所有编码实体的关系
        entity_relations = {}
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 获取每个编码实体的关系
            future_to_entity = {}
            for coded_entity in coded_entities:
                future = executor.submit(self.fetch_relations, coded_entity)
                future_to_entity[future] = coded_entity
            
            # 收集所有编码实体的关系
            for future in as_completed(future_to_entity):
                coded_entity = future_to_entity[future]
                try:
                    relations = future.result()
                    if relations:
                        entity_relations[coded_entity] = relations
                except Exception as e:
                    logger.error(f"Error getting relations for coded entity '{coded_entity}': {e}")
        
        # 对每个编码实体，使用模型选择相关关系
        selected_entity_relations = {}
        for coded_entity, relations in entity_relations.items():
            # 使用模型选择最相关的关系
            selected_relations = self.model.select_relations(
                coded_entity, relations, question, context, history, self.max_k_relations
            )
            if selected_relations:
                selected_entity_relations[coded_entity] = selected_relations
        
        if not selected_entity_relations:
            return
            
        # 获取选定关系的目标实体
        all_paths = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 为每个编码实体的选定关系获取目标实体
            future_to_path = {}
            for coded_entity, relations in selected_entity_relations.items():
                for relation in relations:
                    future = executor.submit(self.kg.get_target_entities, coded_entity, relation, "out")
                    future_to_path[future] = (coded_entity, relation)
            
            # 收集所有路径
            for future in as_completed(future_to_path):
                coded_entity, relation = future_to_path[future]
                try:
                    targets = future.result()
                    if targets:
                        for target in targets:
                            if isinstance(target, str):
                                path_key = f"{source_relation}>{coded_entity}>{relation}"
                                all_paths.append((path_key, target))
                except Exception as e:
                    logger.error(f"Error getting targets for path '{coded_entity}->{relation}': {e}")
        
        # 添加所有路径到扩展结果
        path_results = {}
        for path_key, target in all_paths:
            path_results.setdefault(path_key, []).append(target)
        
        for path_key, targets in path_results.items():
            if targets:
                expansion.relations.append(EntityRelation(relation=path_key, targets=targets))

    def find_potential_sources(self, node: str, exploration_history: List[ExplorationRound]) -> List[Tuple[str, str]]:
        potential_sources = []
        
        for round_data in exploration_history:
            for expansion in round_data.expansions:
                source = expansion.entity
                for relation_info in expansion.relations:
                    relation = relation_info.relation
                    targets = relation_info.targets
                    
                    if node in targets:
                        potential_sources.append((source, relation))
        
        return potential_sources or [("unknown", "unknown")]
