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
    
    def execute_expansion_round(self, state: TraversalState, entities_to_expand: List[str], round_num: int) -> TraversalState:
        logger.info(f"Executing expansion round {round_num} for {len(entities_to_expand)} entities...")
        if not entities_to_expand:
            return state

        prompts_for_llm_batch = []
        relations_map_for_llm = {}
        entities_for_llm_batch = []

        question = state.original_question
        history = " -> ".join(state.current_path)

        for entity in entities_to_expand:
            all_relations = self._fetch_relations(entity)
            if not all_relations:
                continue

            if len(all_relations) <= self.max_relation_selection_count:
                logger.debug(f"Entity '{entity}' has {len(all_relations)} relations (<= max), skipping LLM call.")

                round_data = ExplorationRound(
                    round_num=round_num,
                    question_posed_to_llm="N/A (Auto-selected due to few relations)",
                    expanded_entity=entity,
                    candidate_relations=all_relations,
                    chosen_relations=[], # To be filled below
                    model_reasoning="Auto-selected all relations as the count was below the threshold."
                )
                
                for rel_name in all_relations:
                    targets = self.kg.get_target_entities(entity, rel_name)
                    if targets:
                        round_data.chosen_relations.append({"name": rel_name, "targets": targets})
                
                if round_data.chosen_relations:
                    state.exploration_trace.append(round_data)

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

        # --- Step 2: Call LLM for entities that need selection ---
        if entities_for_llm_batch:
            logger.info(f"--- Calling LLM for a sub-batch of {len(entities_for_llm_batch)} entities that require selection. ---")
            try:
                batch_raw_outputs = self.explore_model_interface.generate_output_batch(prompts_for_llm_batch)
                
                if batch_raw_outputs and len(batch_raw_outputs) == len(entities_for_llm_batch):
                    for i, entity in enumerate(entities_for_llm_batch):
                        raw_output = batch_raw_outputs[i] or ""
                        selected_relation_names, reasoning = self.explore_model_interface.parser.parse_relations_with_reasoning(
                            raw_output, 
                            candidate_items=relations_map_for_llm[entity]
                        )
                        round_data = ExplorationRound(
                            round_num=round_num,
                            question_posed_to_llm=prompts_for_llm_batch[i],
                            expanded_entity=entity,
                            candidate_relations=relations_map_for_llm[entity],
                            chosen_relations=[],
                            model_reasoning=reasoning
                        )
                        
                        if selected_relation_names:
                            for rel_name in selected_relation_names:
                                targets = self.kg.get_target_entities(entity, rel_name)
                                if targets:
                                    round_data.chosen_relations.append({"name": rel_name, "targets": targets})
                        
                        if round_data.chosen_relations:
                            state.exploration_trace.append(round_data)

                else:
                    logger.error("Batch generation for selection failed or returned mismatched number of results.")
            except Exception as e:
                logger.error(f"Exception during batch LLM relation selection: {e}", exc_info=True)
                
        logger.info(f"Expansion round {round_num} complete. {len(state.exploration_trace)} total rounds logged in trace.")
        return state