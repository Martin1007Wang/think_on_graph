'''import concurrent.futures
import datetime
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, OrderedDict, Union
from dataclasses import dataclass, field

# 依赖项
from src.entity_expander import EntityExpander
from src.knowledge_graph import KnowledgeGraph
from src.model_interface import ModelInterface
from src.result_formatter import ResultFormatter
from src.template import KnowledgeGraphTemplates
from src.utils.data_utils import SingleRoundExecutionResult, EntityExpansion, TraversalState, ExplorationRound
from src.llm_output_parser import LLMOutputParser

logger = logging.getLogger(__name__)
LOG_DIR = "logs/unanswerable_questions"

class KnowledgeExplorer:
    def __init__(self,
                 kg: KnowledgeGraph,
                 explore_model: ModelInterface,
                 predict_model: ModelInterface,
                 max_rounds: int = 2,
                 max_selection_count: int = 5,
                 max_frontier_size: Optional[int] = 20,
                 max_exploration_history_size: int = 100000
                 ) -> None:
        self.kg = kg
        self.explore_model = explore_model
        self.predict_model = predict_model
        self.max_rounds = max_rounds
        self.max_frontier_size = max_frontier_size
        self.max_exploration_history_size = max_exploration_history_size
        self.logger = logging.getLogger(__name__)
        self.templates = KnowledgeGraphTemplates()
        self.parser = LLMOutputParser()
        self.explore_model_interface = ModelInterface(self.explore_model, self.templates, self.parser)
        self.predict_model_interface = ModelInterface(self.predict_model, self.templates, self.parser)
        self.formatter = ResultFormatter()

        self.entity_expander = EntityExpander(
            kg=self.kg,
            explore_model_interface=self.explore_model_interface,
            max_relation_selection_count=max_selection_count
        )
        self.logger.info(f"KnowledgeExplorer initialized: max_rounds={max_rounds}")

    def _normalize_start_entities(self, data: Dict[str, Any]) -> List[str]:
        entity_key = "q_entity"
        start_entities_raw = data.get(entity_key, [])
        if not isinstance(start_entities_raw, list):
            start_entities_raw = [start_entities_raw]
        
        normalized_entities: List[str] = []
        seen_entities: Set[str] = set()
        for entity in start_entities_raw:
            if entity is None: continue
            entity_str = str(entity).strip()
            if entity_str and entity_str not in seen_entities:
                normalized_entities.append(entity_str)
                seen_entities.add(entity_str)
        return normalized_entities

    def explore_knowledge_graph(self, traversal_state: TraversalState, start_entities: List[str], ground_truth: List[str] = None) -> TraversalState:
        self.logger.info(f"Starting KG exploration for: '{traversal_state.original_question[:100]}...' ")
        current_frontier = start_entities[:]
        entities_explored: Set[str] = set(start_entities)
        traversal_state.current_path.extend(start_entities)

        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"--- Controller: Starting Round {round_num}/{self.max_rounds} ---")
            if not current_frontier:
                self.logger.warning("Controller: Empty frontier. Stopping exploration.")
                break

            current_entity = current_frontier[0]
            current_path_str = " -> ".join(traversal_state.current_path)
            
            # Get candidate relations
            relations = self.kg.get_relations_for_entity(current_entity)
            if not relations:
                self.logger.warning(f"No relations found for entity: {current_entity}. Stopping exploration for this path.")
                break

            relations_str = "\n".join([f"- {r}" for r in relations])

            question_posed = self.templates.format_template(
                "state_aware_relation_selection",
                original_question=traversal_state.original_question,
                current_path_str=current_path_str,
                current_entity=current_entity,
                relations=relations_str,
                max_selection_count=self.max_selection_count
            )

            # Call the LLM to get the chosen relation
            llm_response = self.explore_model_interface.predict(question_posed)
            parsed_response = self.parser.parse_relation_selection(llm_response)
            
            if not parsed_response:
                self.logger.warning(f"Could not parse LLM response for relation selection in round {round_num}. Stopping exploration.")
                break

            chosen_relation = parsed_response[0]['relation']
            model_reasoning = parsed_response[0]['reasoning']

            # Get next entities
            next_entities = self.kg.get_next_entities(current_entity, chosen_relation)

            # Create and append the exploration round to the trace
            exploration_round = ExplorationRound(
                round_num=round_num,
                current_entity=current_entity,
                question_posed=question_posed,
                candidate_relations=[{"name": r} for r in relations],
                chosen_relation=chosen_relation,
                next_entity=next_entities,
                model_reasoning=model_reasoning
            )
            traversal_state.exploration_trace.append(exploration_round)

            # Update path and frontier
            traversal_state.current_path.append(chosen_relation)
            traversal_state.current_path.extend(next_entities)
            current_frontier = next_entities
            entities_explored.update(next_entities)

            # Check for answer
            if ground_truth and any(entity in ground_truth for entity in next_entities):
                self.logger.info("Controller: SUCCESS - Answer found. Terminating exploration.")
                traversal_state.answer_found = True
                traversal_state.final_answer_entities = [e for e in next_entities if e in ground_truth]
                break

        self.logger.info(f"Finished KG exploration. Final answer found: {traversal_state.answer_found}")
        return traversal_state

    def get_fallback_answer(self, question: str, traversal_state: TraversalState) -> Dict[str, str]:
        self.logger.warning(f"Generating fallback for '{question[:100]}...' ")
        # The fallback can also be improved to use the rich trace, but for now, we keep it simple.
        history_text = ""
        if traversal_state.exploration_trace:
            history_text = self.formatter.format_exploration_trace(traversal_state.exploration_trace)
        return self.predict_model_interface.generate_fallback_answer(question, history_text)

    def process_question(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = data.get("question")
        question_id = data.get("id", "UNKNOWN")
        if not question:
            self.logger.error(f"Invalid/empty question for ID: {question_id}"); return None

        self.logger.info(f"Processing question {question_id}: '{question[:100]}...' ")
        
        traversal_state = TraversalState(original_question=question)

        start_time = time.monotonic()
        self.explore_model.reset_counters()
        if id(self.explore_model) != id(self.predict_model):
            self.predict_model.reset_counters()

        try:
            if not self.explore_model.is_ready or not self.predict_model.is_ready:
                raise RuntimeError("One or both models are not ready. Please call prepare_for_inference() in the main script.")

            start_entities = self._normalize_start_entities(data)
            if not start_entities:
                self.logger.warning(f"[{question_id}] No valid start entities. Using fallback.")
                final_answer_data = self.get_fallback_answer(question, traversal_state)
                fallback_used = True
            else:
                traversal_state = self.explore_knowledge_graph(traversal_state, start_entities, data.get("answer", []))
                
                if traversal_state.answer_found:
                    # The final answer data can be constructed from the traversal state
                    final_answer_data = {
                        "answer_entities": traversal_state.final_answer_entities,
                        "reasoning_summary": "Answer found through exploration."
                    }
                    fallback_used = False
                else:
                    self.logger.warning(f"[{question_id}] Exploration ended without answer. Generating fallback.")
                    final_answer_data = self.get_fallback_answer(question, traversal_state)
                    fallback_used = True
            
            duration_s = time.monotonic() - start_time
            explore_calls, explore_tokens = self.explore_model.get_counters()
            predict_calls, predict_tokens = (0, 0)
            if id(self.explore_model) != id(self.predict_model):
                predict_calls, predict_tokens = self.predict_model.get_counters()

            total_calls = explore_calls + predict_calls
            total_tokens = explore_tokens + predict_tokens

            final_result = self.formatter.create_question_result(
                question_id=question_id, 
                traversal_state=traversal_state,
                ground_truth=data.get("answer", []),
                start_entities=start_entities, 
                final_answer_data=final_answer_data,
                fallback_used=fallback_used,
                runtime_s=duration_s,
                llm_calls=total_calls,
                llm_tokens=total_tokens
            )
            
        except Exception as e:
            self.logger.critical(f"Critical error processing question {question_id}: {e}", exc_info=True)
            return {"id": question_id, "question": question, "error": f"Critical processing error: {e}"}
        
        self.logger.info(f"Question {question_id} processed in {duration_s:.2f} seconds.")
        return final_result
'''