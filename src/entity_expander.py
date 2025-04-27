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
        for relation in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, relation, "out")
                if not targets:
                    continue

                # self._record_direct_paths(entity, relation, targets)
                named_entities, coded_entities = self._separate_entities(targets)

                if named_entities:
                    expansion.relations.append(EntityRelation(relation=relation, targets=named_entities))

                if coded_entities:
                    paths = self._get_multi_hop_paths(
                        entity, relation, coded_entities, question, context, history
                    )
                    for path_relation, path_targets in paths.items():
                        if path_targets:
                            expansion.relations.append(EntityRelation(relation=path_relation, targets=path_targets))
            except Exception as e:
                logger.error(
                    f"[expand_entity] Error expanding relation '{relation}' for entity '{entity}': {e}",
                    exc_info=True
                )
        return expansion if expansion.relations else None
    
    def _separate_entities(self, entities: List[str]) -> Tuple[List[str], List[str]]:
        is_coded = self.path_manager.is_coded_entity
        named_entities = [e for e in entities if isinstance(e, str) and not is_coded(e)]
        coded_entities = [e for e in entities if isinstance(e, str) and is_coded(e)]
        return named_entities, coded_entities
    
    def _get_multi_hop_paths(self, source_entity: str, source_relation: str, 
                           intermediate_nodes: List[str], question: str, 
                           context: str, history: str) -> Dict[str, List[str]]:
        path_results = {}
        node_relations = self._get_node_relations(intermediate_nodes)
        if not node_relations:
            return path_results
        selected_paths = self._select_relevant_paths(source_entity, source_relation, node_relations, question, context, history)
        for node, relation, target in selected_paths:
            if not isinstance(target, str):
                continue
            path_elements = [
                {'source': source_entity, 'relation': source_relation, 'target': node},
                {'source': node, 'relation': relation, 'target': target}
            ]
            # self.path_manager.add_path(path_elements)
            relation_key = f"{source_relation}>{node}>{relation}"
            path_results.setdefault(relation_key, []).append(target)
        return path_results

    def _get_node_relations(self, nodes: List[str]) -> Dict[str, List[str]]:
        node_relations = {}
        if not nodes:
            return node_relations
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_node = {executor.submit(self.fetch_relations, node): node for node in nodes}
            for future in as_completed(future_to_node):
                relations = future.result()
                if relations:
                    node_relations[future_to_node[future]] = relations
        return node_relations
    
    def _select_relevant_paths(self, source_entity: str, source_relation: str, node_relations: Dict[str, List[str]], question: str, context: str, history: str) -> List[Tuple[str, str, str]]:
        complete_path_options = []
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for node, relations in node_relations.items():
                for relation in relations:
                    future = executor.submit(self.kg.get_target_entities, node, relation, "out")
                    tasks.append((node, relation, future))
            for node, relation, future in tasks:
                try:
                    targets = future.result()
                    if targets:
                        complete_path_options.extend(
                            (node, relation, target)
                            for target in targets if isinstance(target, str)
                        )
                except Exception as e:
                    logger.error(f"Error getting targets for node '{node}' with relation '{relation}': {e}")
        if not complete_path_options:
            return []
        option_lines = []
        path_dict = {}
        for i, (node, relation, target) in enumerate(complete_path_options):
            option_id = f"PATH_{i}"
            path_dict[option_id] = (node, relation, target)
            option_lines.append(f"[{option_id}] {source_entity}-[{source_relation}]->{node}-[{relation}]->{target}\n")
        option_text = ''.join(option_lines)
        selection_output = self.model.generate_output(
            "intermediate_path_selection",
            question=question,
            source_entity=source_entity,
            source_relation=source_relation,
            path_options=option_text,
            max_paths=min(5, len(complete_path_options))
        )
        parser = self.model.parser
        selected_path_ids = parser.parse_paths(selection_output, list(path_dict))
        selected_paths = [path_dict[pid] for pid in selected_path_ids if pid in path_dict]
        return selected_paths or [complete_path_options[0]]

    
    def find_potential_sources(self, node: str, exploration_history: List[ExplorationRound]) -> List[Tuple[str, str]]:
        """查找可能指向节点的源节点和关系"""
        potential_sources = []
        
        # 在探索历史中搜索
        for round_data in exploration_history:
            for expansion in round_data.expansions:
                source = expansion.entity
                for relation_info in expansion.relations:
                    relation = relation_info.relation
                    targets = relation_info.targets
                    
                    if node in targets:
                        potential_sources.append((source, relation))
        
        # 如果没有找到，返回默认值
        return potential_sources or [("unknown", "unknown")]
