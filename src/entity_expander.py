import logging
from src.utils.data_utils import ExplorationRound, TraversalState
from typing import Optional, List
from collections import defaultdict

from .knowledge_graph import KnowledgeGraph
from .model_interface import ModelInterface

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
    
    def execute_expansion_round(self, state: TraversalState, entities_to_expand: List[str], round_num: int, current_question: str) -> TraversalState:
        logger.info(f"Executing expansion round {round_num} for {len(entities_to_expand)} entities...")
        if not entities_to_expand:
            return state

        # =================================================================================
        # Phase 1: Preparation & Data Aggregation (准备和数据聚合)
        # 我们不再在循环中调用模型，而是先收集所有需要评分的任务。
        # =================================================================================
        
        # 用于存储需要LLM评分的任务
        batch_prompts_to_score = []
        batch_candidates_to_score = []
        
        # 用于追踪批次中的任务对应哪个原始实体
        # (entity_name, list_of_all_relations)
        entity_metadata_for_scoring = []

        history_path_str = "\n".join(state.reasoning_path)

        for entity in entities_to_expand:
            all_relations = self._fetch_relations(entity)
            if not all_relations:
                continue

            # --- BRANCH 1: Auto-selection (自动选择，逻辑不变，但只处理数据，不更新state) ---
            if len(all_relations) <= self.max_relation_selection_count:
                logger.debug(f"Entity '{entity}' has {len(all_relations)} relations (<= max), auto-selecting all.")
                # 对于自动选择的实体，直接处理并更新state
                self._process_selected_relations_for_entity(state, entity, all_relations, round_num, current_question)

            # --- BRANCH 2: Aggregate for LLM Scoring (为LLM评分聚合数据) ---
            else:
                logger.debug(f"Entity '{entity}' has {len(all_relations)} relations (> max), adding to LLM scoring batch.")
                numbered_relations_str = "\n".join(
                    [f"[REL{i+1}] {rel}" for i, rel in enumerate(all_relations)]
                )
                
                base_prompt = self.explore_model_interface.templates.format_template(
                    "relation_selection_with_history" if state.reasoning_path else "relation_selection",
                    entity=entity,
                    relations=numbered_relations_str,
                    max_selection_count=self.max_relation_selection_count,
                    question=current_question,
                    history=history_path_str
                )
                
                batch_prompts_to_score.append(base_prompt)
                batch_candidates_to_score.append(all_relations)
                entity_metadata_for_scoring.append(entity)
                
        # =================================================================================
        # Phase 2: Batched Model Call (批量化模型调用)
        # 对所有收集到的任务，进行一次集中的、批量化的模型调用。
        # =================================================================================
        
        if batch_prompts_to_score:
            logger.info(f"Scoring relations for a batch of {len(batch_prompts_to_score)} entities...")
            try:
                list_of_relation_scores = self.explore_model_interface.score_relations_in_batch(
                    batch_prompts_to_score,
                    batch_candidates_to_score
                )

                # =================================================================================
                # Phase 3: Processing the Batched Results (处理批量结果)
                # =================================================================================
                
                for i, relation_scores in enumerate(list_of_relation_scores):
                    entity_name = entity_metadata_for_scoring[i]
                    
                    if relation_scores:
                        sorted_relations = sorted(relation_scores, key=relation_scores.get, reverse=True)
                        selected_relation_names = sorted_relations[:self.max_relation_selection_count]
                        
                        logger.debug(f"Scores for '{entity_name}': {relation_scores}")
                        logger.info(f"Top-{self.max_relation_selection_count} selected for '{entity_name}': {selected_relation_names}")
                        
                        # 使用选出的关系来处理和更新state
                        self._process_selected_relations_for_entity(state, entity_name, selected_relation_names, round_num, current_question)

            except Exception as e:
                logger.error(f"Exception during batched MPO scoring: {e}", exc_info=True)

        logger.info(f"Expansion round {round_num} complete. {len(state.exploration_trace)} total rounds logged in trace.")
        return state


    # 为了代码整洁，我建议你将处理逻辑封装成一个辅助函数
    def _process_selected_relations_for_entity(self, state: TraversalState, entity: str, selected_relations: List[str], round_num: int, question: str):
        """
        Helper function to process the selected relations for a single entity and update the state.
        """
        if not selected_relations:
            return

        round_data = ExplorationRound(
            round_num=round_num,
            question=question,
            entity=entity,
            relations=[]
        )

        for rel_name in selected_relations:
            targets = self.kg.get_target_entities(entity, rel_name)
            if targets:
                round_data.relations.append({"name": rel_name, "targets": targets})
        
        if round_data.relations:
            state.exploration_trace.append(round_data)