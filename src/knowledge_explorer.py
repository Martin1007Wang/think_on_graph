import concurrent.futures
import datetime
import logging
import os
import re
import gc
import time
from collections import OrderedDict
from typing import Any, Dict, List, Match, Optional, Pattern, Set, Tuple

# --- Assuming the optimized class is here ---
from src.entity_expander import EntityExpander # MODIFIED: Import the optimized class
# --- End Assumption ---

from src.knowledge_graph import KnowledgeGraph
from src.model_interface import ModelInterface
from src.path_manager import PathManager
from src.result_formatter import ResultFormatter
from src.template import KnowledgeGraphTemplates
from src.utils.data_utils import ExplorationRound, EntityExpansion # Import EntityExpansion if needed for type hints

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Constants ---
MCODE_PATTERN: Pattern[str] = re.compile(r'm\.[0-9a-z_]+')
GCODE_PATTERN: Pattern[str] = re.compile(r'g\.[0-9a-z_]+')
LOG_DIR = "logs/unanswerable_questions"

class KnowledgeExplorer:
    def __init__(self,
                 kg: KnowledgeGraph,
                 explore_model: Any, # Should ideally be ModelInterface or similar protocol
                 predict_model: Any, # Should ideally be ModelInterface or similar protocol
                 max_rounds: int = 2,
                 max_selection_count: int = 5, # Max relations for EntityExpander
                 max_workers: int = 8,
                 max_frontier_size: Optional[int] = 20,
                 min_cvts_for_parallel: int = 3, # Parameter for EntityExpander
                 max_exploration_history_size: int = 100000 # Limit in chars
                 ) -> None:
        self.kg = kg
        self.max_rounds = max_rounds
        self.max_frontier_size = max_frontier_size
        self.max_workers = max_workers
        self.max_exploration_history_size = max_exploration_history_size
        self.logger = logging.getLogger(__name__) # Use instance logger

        # Assuming templates are handled appropriately
        self.templates = KnowledgeGraphTemplates()
        self.path_manager = PathManager()

        # Wrap models in interfaces
        self.explore_model_interface = ModelInterface(explore_model, self.templates)
        self.predict_model_interface = ModelInterface(predict_model, self.templates)

        # --- Instantiate the OPTIMIZED EntityExpander ---
        self.entity_expander = EntityExpander(
            kg=self.kg,
            explore_model_interface=self.explore_model_interface,
            predict_model_interface=self.predict_model_interface,
            path_manager=self.path_manager,
            max_workers=self.max_workers, # Can differ from KE's max_workers if needed
            llm_concurrency_limit=self.max_workers, # Default semaphore limit to max_workers
            max_relation_selection_count=max_selection_count, # Pass the KE param
            min_cvts_for_parallel_processing=min_cvts_for_parallel # Pass the KE param
        )
        # --- END EntityExpander ---

        # Instantiate the OPTIMIZED ResultFormatter
        self.formatter = ResultFormatter(self.path_manager)

        self.logger.info(f"KnowledgeExplorer initialized: max_rounds={max_rounds}, "
                         f"max_workers={self.max_workers}, max_frontier_size={max_frontier_size or 'Unlimited'}, "
                         f"max_hist_size={max_exploration_history_size}, "
                         f"EE_max_rels={max_selection_count}, EE_min_cvts={min_cvts_for_parallel}")

    def _normalize_start_entities(self, data: Dict[str, Any]) -> List[str]:
        """Extracts, cleans, validates, and deduplicates start entities."""
        entity_key = "q_entity"
        start_entities_raw = data.get(entity_key, [])
        if not isinstance(start_entities_raw, list):
            self.logger.debug(f"Input '{entity_key}' not a list, converting.")
            if hasattr(start_entities_raw, '__iter__') and not isinstance(start_entities_raw, (str, bytes)):
                 try: start_entities_raw = list(start_entities_raw)
                 except Exception: start_entities_raw = [start_entities_raw]
            else: start_entities_raw = [start_entities_raw]
        if not start_entities_raw: return []

        normalized_entities: List[str] = []
        seen_entities: Set[str] = set()
        for entity in start_entities_raw:
            if entity is None: continue
            try: entity_str = str(entity).strip()
            except Exception as e: self.logger.warning(f"String conversion failed for entity: {entity!r}: {e}"); continue
            if not entity_str: continue
            if entity_str in seen_entities: continue
            # Add more validation if needed (e.g., pattern matching)
            normalized_entities.append(entity_str)
            seen_entities.add(entity_str)
        self.logger.debug(f"Normalized start entities: {normalized_entities}")
        return normalized_entities


    def explore_knowledge_graph(
        self, question: str, start_entities: List[str], ground_truth: List[str] = None # ground_truth often not list
    ) -> Tuple[List[ExplorationRound], bool]:
        """Explores the knowledge graph iteratively."""
        self.logger.info(f"Starting KG exploration for: '{question[:100]}...'")
        self.logger.info(f"Start entities: {start_entities}")

        exploration_history: List[ExplorationRound] = []
        current_frontier = start_entities[:] # Use copy
        entities_explored: Set[str] = set(start_entities)
        answer_found = False
        exploration_history_text = "" # Accumulates text for LLM checks

        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"--- Starting Exploration Round {round_num}/{self.max_rounds} ---")
            if not current_frontier:
                self.logger.warning(f"Round {round_num}: Empty frontier. Stopping exploration.")
                break

            # Process entities in the current frontier concurrently
            round_exploration = self._process_exploration_round(
                round_num, current_frontier, question, exploration_history_text
            )
            exploration_history.append(round_exploration)

            # Update history text *after* processing the round
            try:
                 round_text = self.formatter.format_round_results(round_exploration)
                 # Check size *before* potentially large append
                 if (len(exploration_history_text) + len(round_text)) > self.max_exploration_history_size:
                     self.logger.warning(f"Round {round_num}: Adding this round's text would exceed history size limit ({self.max_exploration_history_size} chars). Stopping.")
                     round_exploration.exceeded_history_limit = True
                     # Optionally append a truncated version or note?
                     # exploration_history_text += "\n[History Truncated Due To Size Limit]"
                     break # Stop exploration
                 else:
                     exploration_history_text += round_text
            except Exception as format_e:
                 self.logger.error(f"Round {round_num}: Error formatting round results: {format_e}", exc_info=True)
                 # Decide how to handle: stop, continue with potentially bad history text?
                 exploration_history_text += f"\n[Error Formatting Round {round_num}]"
                 # Maybe break if formatting is critical
                 # break

            # Check for answer after processing round and updating history
            answer_check_result = self._check_for_answer(question, start_entities, exploration_history_text)
            can_answer = answer_check_result.get("can_answer", False)

            if can_answer:
                self.logger.info(f"Round {round_num}: Answer check returned positive.")
                round_exploration.answer_found = answer_check_result # Store the dict
                answer_found = True
                break # Answer found, stop exploring

            # If not the last round, update frontier
            if round_num < self.max_rounds:
                self.logger.debug(f"Round {round_num}: Answer not found, updating frontier.")
                current_frontier = self._update_frontier(round_exploration, entities_explored)
                self.logger.info(f"Round {round_num + 1}: New frontier size = {len(current_frontier)}")
                if not current_frontier:
                    self.logger.warning(f"Round {round_num}: Frontier empty after update. Stopping exploration.")
                    break
            else:
                self.logger.info(f"Reached max rounds ({self.max_rounds}) without finding answer.")

        self.logger.info(f"Finished KG exploration. Answer found: {answer_found}")
        return exploration_history, answer_found


    def _process_exploration_round(self, round_num: int, frontier: List[str], question: str, history_str: str) -> ExplorationRound:
        """Processes entity expansions for a single round using ThreadPoolExecutor."""
        round_exploration = ExplorationRound(round_num=round_num)
        successful_expansions: List[EntityExpansion] = []
        processed_count = 0
        actual_workers = min(len(frontier), self.max_workers)
        self.logger.info(f"Round {round_num}: Submitting {len(frontier)} entities using {actual_workers} workers.")

        # defensive check for entity_expander
        if not hasattr(self, 'entity_expander') or not self.entity_expander:
             self.logger.error(f"Round {round_num}: EntityExpander not initialized!")
             return round_exploration # Return empty round

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers, thread_name_prefix=f"EntityExpand_R{round_num}") as executor:
                future_to_entity = {
                    executor.submit(self.entity_expander.expand_entity, entity, question, history_str): entity
                    for entity in frontier
                }
                for future in concurrent.futures.as_completed(future_to_entity):
                    entity = future_to_entity[future]
                    processed_count += 1
                    try:
                        expansion_result = future.result() # Optional[EntityExpansion]
                        # Ensure result is valid and has relations before adding
                        if expansion_result and getattr(expansion_result, 'relations', None):
                            successful_expansions.append(expansion_result)
                            self.logger.debug(f"Round {round_num}: Expanded '{entity}' -> {len(expansion_result.relations)} paths.")
                        #else:
                        #    self.logger.debug(f"Round {round_num}: Processed '{entity}', no expansion results.")
                    except Exception as exc:
                        self.logger.error(f"Round {round_num}: Error expanding entity '{entity}': {exc}", exc_info=False) # Keep False for brevity
        except Exception as pool_exc:
            self.logger.error(f"Round {round_num}: ThreadPool execution error: {pool_exc}", exc_info=True)

        round_exploration.expansions = successful_expansions
        self.logger.info(f"Round {round_num}: Completed processing {processed_count}/{len(frontier)} entities. Got {len(successful_expansions)} successful expansions.")
        return round_exploration


    def _update_frontier(self, round_exploration: ExplorationRound, entities_explored: Set[str]) -> List[str]:
        """Updates the frontier based on the latest round's expansions."""
        # Using OrderedDict ensures deterministic order if expansions are ordered
        new_frontier_candidates = OrderedDict()
        expansions = getattr(round_exploration, 'expansions', [])

        for expansion in expansions:
            relations = getattr(expansion, 'relations', [])
            for relation in relations:
                targets = getattr(relation, 'targets', [])
                for target in targets:
                    # Validate target
                    if target is None: continue
                    try: target_clean = str(target).strip()
                    except Exception: continue # Ignore conversion errors
                    if not target_clean: continue

                    # Add if not explored and not already a candidate in this round
                    if target_clean not in entities_explored and target_clean not in new_frontier_candidates:
                        new_frontier_candidates[target_clean] = True

        new_frontier = list(new_frontier_candidates.keys())
        newly_added_count = len(new_frontier)

        # Apply size limit
        if self.max_frontier_size is not None and newly_added_count > self.max_frontier_size:
            limited_frontier = new_frontier[:self.max_frontier_size]
            self.logger.info(f"Frontier limit applied: Kept {self.max_frontier_size}/{newly_added_count} new unique candidates.")
        else:
            limited_frontier = new_frontier
            #self.logger.info(f"Collected {newly_added_count} new unique entities for next frontier.")

        # Update the master set of explored entities *before* returning
        entities_explored.update(limited_frontier)
        return limited_frontier


    def _check_for_answer(self, question: str, start_entities: List[str], exploration_history: str) -> Dict[str, Any]:
        """Calls the predict model to check answerability based on history."""
        default_fail = {"can_answer": False, "reasoning_path": "Check failed", "answer_entities": [], "analysis": ""}
        if not exploration_history or not exploration_history.strip():
            self.logger.warning("Empty history provided to _check_for_answer.")
            default_fail["reasoning_path"] = "No exploration history provided."
            return default_fail

        self.logger.debug("Checking answerability with predict model...")
        try:
            # Ensure interface and method exist
            if not hasattr(self.predict_model_interface, 'check_answerability') or not callable(self.predict_model_interface.check_answerability):
                 self.logger.error("Predict model interface missing 'check_answerability' method.")
                 default_fail["reasoning_path"] = "Internal Error: Predict model cannot check answerability."
                 return default_fail

            result = self.predict_model_interface.check_answerability(question, start_entities, exploration_history)

            if not isinstance(result, dict) or 'can_answer' not in result:
                self.logger.error(f"Invalid structure from check_answerability: {result!r}")
                default_fail["reasoning_path"] = "Internal Error: Invalid response from answer check model."
                return default_fail

            self.logger.debug(f"Answerability check result: can_answer={result.get('can_answer')}")
            # Ensure essential keys exist, even if False/empty
            result.setdefault("reasoning_path", "")
            result.setdefault("answer_entities", [])
            result.setdefault("analysis", "")
            return result
        except Exception as e:
            self.logger.error(f"Error during check_answerability call: {e}", exc_info=True)
            default_fail["reasoning_path"] = f"Internal Error during check: {str(e)}"
            return default_fail


    def get_fallback_answer(self, question: str, exploration_history: Optional[List[ExplorationRound]] = None) -> Dict[str, str]:
        """Generates a fallback answer using the predict model interface."""
        exploration_history = exploration_history or []
        history_size_exceeded = any(getattr(round_data, "exceeded_history_limit", False) for round_data in exploration_history)
        reason = "History size limit exceeded" if history_size_exceeded else "Exploration did not yield a conclusive answer"
        self.logger.warning(f"Generating fallback for '{question[:100]}...' (Reason: {reason})")

        # Default fallback structure
        default_fallback = {
             "answer": "Based on the available information, I cannot provide a definitive answer.",
             "reasoning": f"The exploration process encountered limitations ({reason})."
        }

        try:
            # Ensure interface and method exist
            if not hasattr(self.predict_model_interface, 'generate_fallback_answer') or not callable(self.predict_model_interface.generate_fallback_answer):
                 self.logger.error("Predict model interface missing 'generate_fallback_answer' method.")
                 default_fallback["reasoning"] += " Internal Error: Predict model cannot generate fallback."
                 return default_fallback

            # Call the actual fallback generation
            fallback_answer = self.predict_model_interface.generate_fallback_answer(question, exploration_history)

            # Validate the result
            if not isinstance(fallback_answer, dict) or 'answer' not in fallback_answer or 'reasoning' not in fallback_answer:
                self.logger.error(f"Invalid structure from generate_fallback_answer: {fallback_answer!r}. Using default.")
                # Maybe merge reason into default if structure is bad?
                default_fallback["reasoning"] = f"The exploration process encountered limitations ({reason}), and the fallback generation returned an invalid format."
                fallback_answer = default_fallback
            else:
                 # Ensure fields aren't empty strings (optional, depends on desired behaviour)
                 if not fallback_answer.get("answer", "").strip():
                      fallback_answer["answer"] = default_fallback["answer"] # Use default answer text
                 if not fallback_answer.get("reasoning", "").strip():
                      fallback_answer["reasoning"] = default_fallback["reasoning"] # Use default reasoning text

            # Save log *after* potentially correcting the fallback answer
            self._save_unanswerable_question(question, reason=reason) # Log attempt
            return fallback_answer

        except Exception as e:
            self.logger.error(f"Error calling generate_fallback_answer: {e}", exc_info=True)
            default_fallback["reasoning"] = f"An error occurred during fallback generation: {str(e)}."
            self._save_unanswerable_question(question, reason=f"Error: {e}") # Log attempt even on error
            return default_fallback


    def _save_unanswerable_question(self, question: str, reason: Optional[str] = None) -> None:
        """Logs questions for which fallback answer generation was attempted."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            now = datetime.datetime.now(datetime.timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            log_file = os.path.join(LOG_DIR, f"unanswerable_{date_str}.log")
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")
            reason_str = f" [Reason: {reason}]" if reason else ""
            log_line = f"[{timestamp}]{reason_str} {question}\n"
            # Use 'a' for append mode
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except OSError as e:
            self.logger.error(f"OS Error interacting with log file {log_file}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving unanswerable question log: {e}")


    def process_question(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single question end-to-end. (Incorporates previous fixes)
        """
        question = data.get("question")
        question_id = data.get("id")

        # --- Input Validation ---
        if not question or not isinstance(question, str) or not question.strip():
            self.logger.error(f"Invalid/empty question. ID: {question_id or 'N/A'}. Data: {data!r}")
            return None # Cannot proceed
        if question_id is None:
            ts_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f')
            question_id = f"gen_id_{ts_id}_{abs(hash(question))}"
            self.logger.warning(f"No 'id' found, generated: {question_id}")

        ground_truth = data.get("answer", []) # Use 'answer' key for ground truth per user input
        if not isinstance(ground_truth, list): # Ensure GT is list
             self.logger.warning(f"[{question_id}] Ground truth was not list, converting: {type(ground_truth)}")
             try: ground_truth = list(ground_truth) if hasattr(ground_truth, '__iter__') and not isinstance(ground_truth, str) else [ground_truth]
             except Exception: ground_truth = []

        self.logger.info(f"Processing question {question_id}: '{question[:100]}...'")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        final_result: Optional[Dict[str, Any]] = None
        exploration_history: List[ExplorationRound] = []
        answer_found: bool = False
        fallback_used: bool = False
        final_answer_data: Dict[str, Any] = {} # Initialize

        try:
            # --- Normalize Start Entities ---
            start_entities = self._normalize_start_entities(data)

            if not start_entities:
                self.logger.warning(f"[{question_id}] No valid start entities. Using fallback immediately.")
                # **Directly assign to final_answer_data**
                final_answer_data = self.get_fallback_answer(question)
                fallback_used = True
                answer_found = False
            else:
                # --- Explore Knowledge Graph ---
                exploration_history, answer_found = self.explore_knowledge_graph(
                    question, start_entities, ground_truth
                )

                # --- Determine Final Answer Data ---
                if answer_found:
                    potential_answer_data = getattr(exploration_history[-1], 'answer_found', None)
                    if isinstance(potential_answer_data, dict):
                        final_answer_data = potential_answer_data
                        fallback_used = False
                        self.logger.info(f"[{question_id}] Answer found during exploration.")
                    else:
                        self.logger.error(f"[{question_id}] Answer flagged found, but data invalid: {type(potential_answer_data)}. Falling back.")
                        answer_found = False # Correct status

                if not answer_found: # Handles initial failure or invalid found data
                    fallback_used = True
                    history_exceeded = any(getattr(r, 'exceeded_history_limit', False) for r in exploration_history)
                    log_reason = "History size limit exceeded" if history_exceeded else "Exploration ended without answer"
                    self.logger.warning(f"[{question_id}] {log_reason}. Generating fallback.")
                    # **Directly assign to final_answer_data**
                    final_answer_data = self.get_fallback_answer(question, exploration_history)
                    if history_exceeded:
                         self.logger.info(f"[{question_id}] Fallback context: History limit exceeded.")
                    # **No destructive modifications needed here**

            # --- Final check on final_answer_data before formatting ---
            if not isinstance(final_answer_data, dict):
                 self.logger.error(f"[{question_id}] final_answer_data is not a dict after processing (type: {type(final_answer_data)}). Using emergency default.")
                 # Provide a structure that create_question_result can handle
                 final_answer_data = {
                      "answer": "Error: Failed to determine final answer data.",
                      "reasoning": "Processing error led to invalid answer data structure."
                 }
                 # Ensure flags are consistent
                 fallback_used = True
                 answer_found = False


            # --- Create Final Result ---
            if not hasattr(self, 'formatter') or not self.formatter:
                 self.logger.critical(f"[{question_id}] Result formatter missing!")
                 return { "id": question_id, "error": "Result formatter not configured." }

            start_entities_list = start_entities or [] # Ensure list

            final_result = self.formatter.create_question_result(
                question_id=question_id, question=question, ground_truth=ground_truth,
                start_entities=start_entities_list,
                answer=final_answer_data, # Pass the unified dictionary
                exploration_history=exploration_history,
                answer_found=answer_found, # Original exploration status
                fallback_used=fallback_used
            )

        except Exception as e:
            self.logger.error(f"Critical error processing question {question_id}: {e}", exc_info=True)
            try:
                # Attempt to format an error response
                start_entities_list_err = start_entities if 'start_entities' in locals() and start_entities is not None else []
                hist_err = exploration_history if 'exploration_history' in locals() else []
                # Use the dedicated error formatter if available
                if hasattr(self.formatter, 'create_error_response'):
                     # Provide a basic fallback dict for the error formatter
                      basic_fallback = {"answer": f"Critical Error: {e}", "reasoning": "Processing failed"}
                      final_result = self.formatter.create_error_response(
                           question=question, question_id=question_id, ground_truth=ground_truth,
                           start_entities=start_entities_list_err, error_msg=str(e), fallback=basic_fallback
                      )
                else: # Absolute fallback if error formatter missing
                     final_result = { "id": question_id, "question": question, "error": f"Critical processing error: {e}" }

            except Exception as format_e:
                 self.logger.error(f"[{question_id}] Failed even to format critical error result: {format_e}")
                 final_result = { "id": question_id, "question": question, "error": f"Critical processing error: {e}; Formatting error: {format_e}" }

        finally:
            # Log Duration and Outcome
            end_time = datetime.datetime.now(datetime.timezone.utc)
            duration = (end_time - start_time).total_seconds()
            outcome = "Failure (Critical Error)"
            if final_result and "error" not in final_result: # Check if result formatting itself failed
                # Refined outcome logic based on flags
                if not fallback_used and answer_found:
                    outcome = "Success (Exploration)"
                elif fallback_used:
                    # Check if prediction text indicates inability to answer
                    final_pred = final_result.get("prediction", "").lower()
                    # More robust check for negative/error indicators
                    negative_indicators = ["error", "could not", "sorry", "cannot", "unable", "failed"]
                    if any(indicator in final_pred for indicator in negative_indicators):
                        outcome = "Failure (Fallback Could Not Answer)"
                    else:
                        outcome = "Success (Fallback)"
                else: # Not fallback, not found
                    outcome = "Failure (Exploration No Answer)"
            elif final_result and "error" in final_result:
                 outcome = f"Failure ({final_result['error'][:50]}...)" # Include part of error message


            self.logger.info(f"Question {question_id} processed in {duration:.2f} seconds. Final Outcome: {outcome}")
            # Optional: Explicit GC call. Usually not needed unless specific memory issues are observed.
            gc.collect()

        return final_result