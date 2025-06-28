import logging
import threading
from typing import Optional, List
from collections import defaultdict

from src.utils.data_utils import EntityExpansion, Relation
from .knowledge_graph import KnowledgeGraph
from .model_interface import ModelInterface  # 依赖项不变

logger = logging.getLogger(__name__)

class EntityExpander:
    def __init__(self,
                 kg: KnowledgeGraph,
                 explore_model_interface: ModelInterface,
                 max_relation_selection_count: int = 5):
            
        self.kg = kg
        self.explore_model_interface = explore_model_interface
        self.max_relation_selection_count = max_relation_selection_count
        logger.info(f"EntityExpander (Batch Mode) initialized with max_relation_selection_count={self.max_relation_selection_count}")

    def _fetch_relations(self, node: str) -> List[str]:
        try:
            return self.kg.get_related_relations(node)
        except Exception as e:
            logger.error(f"Error getting relations for node '{node}': {e}", exc_info=True)
            return []

    def _create_expansion_from_relations(self, entity: str, question: str, relation_names: List[str]) -> Optional[EntityExpansion]:
        expansion_result = EntityExpansion(entity=entity, question=question)
        for relation_name in relation_names:
            try:
                targets = self.kg.get_target_entities(entity, relation_name)
                if targets:
                    expansion_result.relations.append(
                        Relation(name=relation_name, targets=targets)
                    )
            except Exception as e:
                logger.error(f"Error getting targets for '{entity}'-['{relation_name}']: {e}")
        if expansion_result.relations:
            return self._merge_and_deduplicate_relations(expansion_result)
        return None
    
    def expand_entities_in_batch(self, entities: List[str], question: str, history: str) -> List[EntityExpansion]:
        logger.info(f"Expanding batch of {len(entities)} entities...")
        if not entities:
            return []

        final_expansions: List[EntityExpansion] = []
        prompts_for_llm_batch = []
        relations_map_for_llm = {}
        entities_for_llm_batch = []
        for entity in entities:
            all_relations = self._fetch_relations(entity)
            if not all_relations:
                continue
            if len(all_relations) <= self.max_relation_selection_count:
                logger.debug(f"Entity '{entity}' has {len(all_relations)} relations (<= max), skipping LLM call.")
                expansion = self._create_expansion_from_relations(entity, question, all_relations)
                if expansion:
                    final_expansions.append(expansion)
            else:
                logger.debug(f"Entity '{entity}' has {len(all_relations)} relations (> max), adding to LLM batch.")
                entities_for_llm_batch.append(entity)
                relations_map_for_llm[entity] = all_relations
                numbered_relations_str = "\n".join(
                    [f"[REL{i+1}] {rel}" for i, rel in enumerate(all_relations)]
                )
                prompt = self.explore_model_interface.templates.format_template(
                    "relation_selection_with_history" if history else "relation_selection",
                    entity=entity, 
                    relations=numbered_relations_str,
                    max_selection_count=self.max_relation_selection_count,
                    question=question, 
                    history=history
                )
                prompts_for_llm_batch.append(prompt)
        if entities_for_llm_batch:
            logger.info(f"--- Calling LLM for a sub-batch of {len(entities_for_llm_batch)} entities that require selection. ---")
            try:
                batch_raw_outputs = self.explore_model_interface.generate_output_batch(prompts_for_llm_batch)
                if batch_raw_outputs and len(batch_raw_outputs) == len(entities_for_llm_batch):
                    for i, entity in enumerate(entities_for_llm_batch):
                        raw_output = batch_raw_outputs[i] if batch_raw_outputs[i] else ""
                        selected_relations = self.explore_model_interface.parser.parse_relations(
                            raw_output, 
                            candidate_items=relations_map_for_llm[entity]
                        )
                        if selected_relations:
                            expansion = self._create_expansion_from_relations(entity, question, selected_relations)
                            if expansion:
                                final_expansions.append(expansion)
                else:
                    logger.error("Batch generation for selection failed or returned mismatched number of results.")
            except Exception as e:
                logger.error(f"Exception during batch LLM relation selection: {e}", exc_info=True)

        logger.info(f"Batch expansion complete. Total successfully expanded entities: {len(final_expansions)}.")
        return final_expansions
    
    
    def expand_entity(self, entity: str, question: str, history: str) -> Optional[EntityExpansion]:
        logger.debug(f"Expanding entity: '{entity}'...")
        all_relations = self._fetch_relations(entity)
        if not all_relations:
            return None
        selected_relations = self._execute_llm_relation_selection(
            entity, all_relations, question, history, self.max_relation_selection_count
        )
        if not selected_relations:
            return None
            
        expansion_result = EntityExpansion(entity=entity, question=question)
        for relation_name in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, relation_name)
                if targets:
                    expansion_result.relations.append(
                        Relation(name=relation_name, targets=targets)
                    )
            except Exception as e:
                logger.error(f"Error getting targets for '{entity}'-['{relation_name}']: {e}")
        if expansion_result.relations:
            expansion_result = self._merge_and_deduplicate_relations(expansion_result)

        return expansion_result if expansion_result.relations else None
    
    def _merge_and_deduplicate_relations(self, expansion: EntityExpansion) -> EntityExpansion:
        if not expansion or not expansion.relations:
            return expansion
        merged_targets = defaultdict(set)
        for rel in expansion.relations:
            merged_targets[rel.name].update(rel.targets)
        
        deduplicated_relations = [
            Relation(name=rel_name, targets=sorted(list(targets)))
            for rel_name, targets in merged_targets.items()
        ]
        expansion.relations = deduplicated_relations
        return expansion