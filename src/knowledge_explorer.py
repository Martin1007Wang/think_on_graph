import concurrent.futures
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
from src.utils.data_utils import TraversalState, ExplorationRound
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

    def expand_frontier(self, traversal_state: TraversalState, frontier_to_expand: List[str], round_num: int) -> Tuple[TraversalState, Set[str]]:
        if not frontier_to_expand:
            return traversal_state, set()
        traversal_state = self.entity_expander.execute_expansion_round(
            traversal_state,
            frontier_to_expand,
            round_num
        )
        return traversal_state

    def get_fallback_answer(self, question: str, traversal_state: TraversalState) -> Dict[str, str]:
        self.logger.warning(f"Generating fallback for '{question[:100]}...' ")
        history_path_str = "\n".join(traversal_state.reasoning_path)
        return self.predict_model_interface.generate_fallback_answer(question, history_path_str)

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
        
        fallback_used = False
        start_entities = [] # Initialize here

        try:
            if not self.explore_model.is_ready or not self.predict_model.is_ready:
                raise RuntimeError("One or both models are not ready.")

            start_entities = self._normalize_start_entities(data)
            
            if not start_entities:
                self.logger.warning(f"[{question_id}] No valid start entities. Using fallback.")
                fallback_data = self.get_fallback_answer(question, traversal_state)
                traversal_state.final_answer_entities = fallback_data.get("answer_entities", [])
                traversal_state.final_reasoning_summary = fallback_data.get("reasoning_summary", "")
                fallback_used = True
            else:
                current_frontier = start_entities[:]
                entities_explored = set(start_entities)
                current_question = question
                fallback_used = False
                # --- DYNAMIC EXPLORATION & REASONING LOOP ---
                for round_num in range(1, self.max_rounds + 1):
                    self.logger.info(f"--- Controller: Starting Reasoning Cycle, Round {round_num} ---")
                    self.logger.info(f"Current task: '{current_question}'")
                    traversal_state = self.entity_expander.execute_expansion_round(traversal_state,current_frontier,round_num, current_question)
                    self.logger.info("Controller: Checking answerability with PredictModel...")
                    decision = self.predict_model_interface.check_answerability(traversal_state)
                    traversal_state.reasoning_trace.append(decision)
                    if decision.get("answer_found"):
                        self.logger.info("Controller: PredictModel confirmed answer found. Terminating exploration.")
                        traversal_state.answer_found = True
                        traversal_state.final_answer_entities = decision.get("answer_entities", [])
                        traversal_state.final_reasoning_paths = decision.get("reasoning_paths", [])
                        traversal_state.final_reasoning_summary = decision.get("reasoning_summary", "")
                        break
                    self.logger.info("Controller: PredictModel suggests continuing exploration.")
                    next_step = decision.get("next_exploration_step")
                    
                    if not next_step:
                        self.logger.warning("Controller: PredictModel suggested continuing but gave no next entity. Stopping.")
                        break
                    current_question = next_step.get("question", current_question)
                    current_frontier = [next_step.get("entity")]
                    traversal_state.reasoning_path.extend(decision.get("pruned_reasoning_paths",[]))
                    entities_explored.update(current_frontier) 
                    if round_num == self.max_rounds:
                        self.logger.warning(f"Controller: Max rounds ({self.max_rounds}) reached. Stopping exploration.")
                
                # --- POST-LOOP FALLBACK LOGIC (Correctly placed) ---
                if not traversal_state.answer_found:
                    self.logger.warning(f"[{question_id}] Exploration cycle ended without a confirmed answer. Using fallback.")
                    fallback_data = self.get_fallback_answer(question, traversal_state)
                    traversal_state.final_answer_entities = fallback_data.get("answer_entities", [])
                    traversal_state.final_reasoning_summary = fallback_data.get("reasoning_summary", "")
                    fallback_used = True
            duration_s = time.monotonic() - start_time
            explore_calls, explore_tokens = self.explore_model.get_counters()
            predict_calls, predict_tokens = self.predict_model.get_counters()
            total_calls = explore_calls + predict_calls
            total_tokens = explore_tokens + predict_tokens

            final_result = self.formatter.create_question_result(
                question_id=question_id, 
                traversal_state=traversal_state,
                ground_truth=data.get("answer", []),
                start_entities=start_entities,
                fallback_used=fallback_used,
                runtime_s=duration_s,
                explore_calls = explore_calls,
                predict_calls = predict_calls,
                llm_calls=total_calls,
                explore_tokens = explore_tokens,
                predict_tokens = predict_tokens,
                llm_tokens=total_tokens,
            )
            
        except Exception as e:
            self.logger.critical(f"Critical error processing question {question_id}: {e}", exc_info=True)
            # Using the formatter's error response for consistency
            return self.formatter.create_error_response(question_id, question, str(e))
        
        self.logger.info(f"Question {question_id} processed in {duration_s:.2f} seconds.")
        return final_result