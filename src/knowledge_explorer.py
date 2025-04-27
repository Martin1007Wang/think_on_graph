import logging
import re
import os
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import concurrent.futures
from collections import OrderedDict
import json
from functools import lru_cache

from src.knowledge_graph import KnowledgeGraph
from src.llm_output_parser import LLMOutputParser
from src.template import KnowledgeGraphTemplates

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class KnowledgeExplorer:
    def __init__(self, 
                 kg: KnowledgeGraph, 
                 model: Any, 
                 max_rounds: int = 3, 
                 max_k_relations: int = 10, 
                 max_workers: int = 5, 
                 max_frontier_size: Optional[int] = 20) -> None:
        """Initialize the Knowledge Explorer.
        
        Args:
            kg: Knowledge graph instance
            model: LLM model instance
            max_rounds: Maximum exploration rounds
            max_k_relations: Maximum relations to explore per entity
            max_workers: Maximum parallel workers
            max_frontier_size: Maximum frontier size (None for unlimited)
        """
        self.kg = kg
        self.model = model
        self.max_rounds = max_rounds
        self.max_k_relations = max_k_relations
        self.parser = LLMOutputParser()
        self.templates = KnowledgeGraphTemplates()
        self.max_workers = max_workers
        self.max_frontier_size = max_frontier_size
        # Create a single thread pool executor for the instance
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(
            f"KnowledgeExplorer initialized with max_rounds={max_rounds}, "
            f"max_k_relations={max_k_relations}, max_workers={max_workers}, "
            f"max_frontier_size={max_frontier_size}"
        )

    def __del__(self):
        """Ensure executor is properly shutdown when the object is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def format_exploration_round(self, round_data: Dict[str, Any]) -> str:
        """Format a single exploration round for display."""
        if not isinstance(round_data, dict):
            logger.warning("Invalid round data format")
            return "[Invalid round data format]"
            
        round_num = round_data.get("round", 0)
        result = [f"Round {round_num}:"]
        
        expansions = round_data.get("expansions", [])
        if not isinstance(expansions, list):
            logger.warning(f"Round {round_num} has invalid 'expansions' format")
            return f"Round {round_num}: [Invalid expansions format]"

        for expansion in expansions:
            if not isinstance(expansion, dict) or "entity" not in expansion:
                continue
                
            entity = expansion["entity"]
            result.append(f"  Entity: {entity}")
            
            relations = expansion.get("relations", [])
            if not isinstance(relations, list):
                continue
                
            for relation_info in relations:
                if not isinstance(relation_info, dict) or "relation" not in relation_info or "targets" not in relation_info:
                    continue
                    
                relation = relation_info["relation"]
                targets = relation_info["targets"]
                
                if targets and isinstance(targets, list) and all(isinstance(t, str) for t in targets):
                    result.append(f"    {entity} --[{relation}]--> {', '.join(targets)}")
                else:
                    result.append(f"    {entity} --[{relation}]--> [Invalid Targets]")

        result.append("")  # Add newline after the round
        return "\n".join(result)

    def format_exploration_history(self, exploration_history: List[Dict[str, Any]]) -> str:
        """Format the entire exploration history."""
        if not exploration_history:
            return ""
            
        return "\n".join(self.format_exploration_round(round_data) for round_data in exploration_history).rstrip()

    def _generate_model_output(self, template_name: str, temp_generation_mode: str = "beam", 
                              num_beams: int = 4, **template_args) -> str:
        """Generate output from the model using a template."""
        prompt = self.templates.format_template(template_name, **template_args)
        if not prompt:
            logger.error(f"Failed to format template '{template_name}'")
            return ""
        
        model_input = self.model.prepare_model_prompt(prompt)
        return self.model.generate_sentence(
            model_input, 
            temp_generation_mode=temp_generation_mode,
            num_beams=num_beams
        )

    @lru_cache(maxsize=128)
    def _get_related_relations(self, entity: str, question: str, context: str = "", history: str = "") -> List[str]:
        """Get related relations for an entity that are relevant to the question."""
        out_related_relations = self.kg.get_related_relations(entity, "out")
        if not out_related_relations:
            logger.debug(f"No relations found for entity '{entity}'")
            return []
            
        # If relations are fewer than or equal to max_k_relations, return all of them directly
        if len(out_related_relations) <= self.max_k_relations:
            logger.debug(f"Entity '{entity}' has {len(out_related_relations)} relations, no selection needed")
            return out_related_relations
            
        # Prepare relation dictionary for the prompt
        relation_dict = {f"REL_{i}": rel for i, rel in enumerate(out_related_relations)}
        relation_options = "\n".join(f"[{rel_id}] {rel}" for rel_id, rel in relation_dict.items())

        # Select template based on available context
        template_name = "relation_selection_context" if context and history else "relation_selection"
        template_args = {
            "question": question,
            "entity": entity,
            "relations": relation_options,
            "relation_ids": ", ".join(relation_dict.keys()),
            "max_k_relations": min(self.max_k_relations, len(out_related_relations))
        }
        
        if context and history:
            template_args.update({"history": history, "context": context})

        try:
            # Generate model output
            selection_output = self._generate_model_output(template_name, **template_args)
            if not selection_output:
                return out_related_relations[:self.max_k_relations]
                
            # Parse output and convert to relation names
            selected_relations_or_ids = self.parser.parse_relations(selection_output, out_related_relations)
            selected_names = []
            
            for item in selected_relations_or_ids:
                if item.startswith("REL_") and item in relation_dict:
                    selected_names.append(relation_dict[item])
                elif item in out_related_relations:
                    selected_names.append(item)
            
            # Return unique relation names or fallback to first max_k_relations
            return list(OrderedDict.fromkeys(selected_names)) if selected_names else out_related_relations[:self.max_k_relations]

        except Exception as e:
            logger.error(f"Error selecting relations for '{entity}': {e}", exc_info=True)
            # Fallback to first max_k_relations relations
            return out_related_relations[:self.max_k_relations]

    def _expand_entity(self, entity: str, question: str, context: str, history: str) -> Optional[Dict[str, Any]]:
        """Expand an entity by finding its relations and targets."""
        selected_relations = self._get_related_relations(entity, question, context, history)
        if not selected_relations:
            return None
            
        entity_expansion = {"entity": entity, "relations": []}
        
        for rel_name in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, rel_name, "out")
                if targets:
                    entity_expansion["relations"].append({
                        "relation": rel_name, 
                        "targets": targets
                    })
            except Exception as e:
                logger.error(f"Error retrieving targets for '{entity}' with relation '{rel_name}': {e}")
                
        return entity_expansion if entity_expansion["relations"] else None

    def _create_answer_result(self, can_answer: bool, reasoning: str = "", 
                            entities: List[str] = None, answer: str = "",
                            verification: str = "", is_verified: bool = False) -> Dict[str, Any]:
        """Create a standardized answer result dictionary."""
        return {
            "can_answer": can_answer, 
            "reasoning_path": reasoning, 
            "answer_entities": entities or [], 
            "answer_sentence": answer, 
            "verification": verification, 
            "is_verified": is_verified
        }

    def check_if_can_answer(self, question: str, start_entities: List[str], 
                           exploration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if the question can be answered based on exploration history."""
        updated_history = self.format_exploration_history(exploration_history)
        if not updated_history:
            logger.warning("Empty exploration history for answerability check")
            return self._create_answer_result(False, "No exploration performed.")
            
        try:
            # Generate reasoning based on exploration history
            reasoning_output = self._generate_model_output(
                "reasoning",
                question=question,
                entity=", ".join(start_entities),
                num_rounds=len(exploration_history),
                exploration_history=updated_history
            )
            
            if not reasoning_output:
                return self._create_answer_result(False, "Failed to generate reasoning.")
                
            result = self.parser.parse_reasoning_output(reasoning_output)
            
            # Handle m-codes if present in the answer
            if self._process_m_codes_if_needed(result, exploration_history):
                logger.info("Re-checking answerability after m-code expansion")
                return self._check_after_expansion(question, start_entities, exploration_history)
                
            return result
            
        except Exception as e:
            logger.error(f"Error during answerability check: {e}", exc_info=True)
            return self._create_answer_result(False, f"Error during reasoning: {e}")

    def _check_after_expansion(self, question: str, start_entities: List[str], 
                              exploration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check answerability after additional expansion."""
        updated_history = self.format_exploration_history(exploration_history)
        reasoning_output = self._generate_model_output(
            "reasoning",
            question=question,
            entity=", ".join(start_entities),
            num_rounds=len(exploration_history),
            exploration_history=updated_history
        )
        
        if not reasoning_output:
            return self._create_answer_result(False, "Failed to generate reasoning after expansion.")
            
        return self.parser.parse_reasoning_output(reasoning_output)

    def _extract_m_codes(self, result: Dict[str, Any]) -> List[str]:
        """Extract m-codes from answer text and entities."""
        if not result.get("can_answer", False):
            return []
            
        text_to_check = result.get("answer_sentence", "")
        entities_to_check = result.get("answer_entities", [])
        
        m_codes_from_text = re.findall(r'm\.[0-9a-z_]+', text_to_check) if isinstance(text_to_check, str) else []
        m_codes_from_entities = [
            entity for entity in entities_to_check 
            if isinstance(entity, str) and entity.startswith("m.")
        ]
        
        return list(OrderedDict.fromkeys(m_codes_from_text + m_codes_from_entities))

    def _process_m_codes_if_needed(self, result: Dict[str, Any], 
                                  exploration_history: List[Dict[str, Any]]) -> bool:
        """Process m-codes in the answer if needed by adding an extra exploration round."""
        m_codes = self._extract_m_codes(result)
        if not m_codes:
            return False
            
        logger.info(f"Found m-code(s) in answer: {m_codes}")
        
        # Check if we can add another round
        if len(exploration_history) >= self.max_rounds:
            logger.warning(f"Cannot add m-code expansion round (at max rounds: {self.max_rounds})")
            return False
            
        # Create extra exploration round for m-codes
        extra_round = {"round": len(exploration_history) + 1, "expansions": []}
        additional_exploration_added = False
        
        for m_code in OrderedDict.fromkeys(m_codes):  # Use OrderedDict for deduplication
            try:
                entity_expansion = self._expand_m_code(m_code)
                if entity_expansion and entity_expansion.get("relations"):
                    extra_round["expansions"].append(entity_expansion)
                    additional_exploration_added = True
            except Exception as e:
                logger.error(f"Error expanding m-code '{m_code}': {e}", exc_info=True)
                
        if additional_exploration_added:
            exploration_history.append(extra_round)
            logger.info(f"Added exploration round {extra_round['round']} for m-codes")
            return True
        
        logger.info("No new information found for m-codes")
        return False

    def _expand_m_code(self, m_code: str) -> Optional[Dict[str, Any]]:
        """Expand an m-code entity by finding its relations and targets."""
        outgoing_relations = self.kg.get_related_relations(m_code, "out")
        if not outgoing_relations:
            return None
            
        entity_expansion = {"entity": m_code, "relations": []}
        relations_to_expand = outgoing_relations[:self.max_k_relations]
        
        for relation in relations_to_expand:
            try:
                targets = self.kg.get_target_entities(m_code, relation, "out")
                if targets:
                    entity_expansion["relations"].append({
                        "relation": relation,
                        "targets": targets
                    })
            except Exception as e:
                logger.error(f"Error getting targets for m-code '{m_code}' with relation '{relation}': {e}")
                
        return entity_expansion if entity_expansion["relations"] else None

    def _select_frontier_entities(self, question: str, candidates: List[str]) -> List[str]:
        """Select entities for the next frontier using LLM if needed."""
        num_candidates = len(candidates)
        
        # If frontier size is within limits, return as is
        if self.max_frontier_size is None or num_candidates <= self.max_frontier_size:
            return candidates
            
        logger.info(f"Selecting up to {self.max_frontier_size} entities from {num_candidates} candidates")
        
        try:
            # Prepare entity selection prompt
            entity_dict = {f"ENT_{i}": entity for i, entity in enumerate(candidates)}
            entities_str = "\n".join([f"[{ent_id}] {entity}" for ent_id, entity in entity_dict.items()])
            
            selection_output = self._generate_model_output(
                "entity_selection",
                question=question,
                entities=entities_str,
                max_k_entities=self.max_frontier_size,
                entity_ids=", ".join(entity_dict.keys())
            )
            
            if not selection_output:
                logger.warning("Failed to generate entity selection output")
                return candidates[:self.max_frontier_size]
                
            # Parse selected entities
            selected_entity_ids = self.parser.parse_relations(selection_output, list(entity_dict.values()))
            
            if not selected_entity_ids:
                logger.warning("Entity selection parsing returned no entities")
                return candidates[:self.max_frontier_size]
                
            # Process and deduplicate selected entities
            selected_entities = []
            processed_ids = set()
            
            for ent_id in selected_entity_ids:
                # Handle case where selection returns entity ID (ENT_X)
                if ent_id in entity_dict and ent_id not in processed_ids:
                    selected_entities.append(entity_dict[ent_id])
                    processed_ids.add(ent_id)
                # Handle case where selection returns actual entity name
                elif ent_id in entity_dict.values() and ent_id not in selected_entities:
                    selected_entities.append(ent_id)
                    
            return selected_entities or candidates[:self.max_frontier_size]

        except Exception as e:
            logger.error(f"Error during entity selection: {e}", exc_info=True)
            return candidates[:self.max_frontier_size]

    def _process_exploration_round(self, round_num: int, frontier: List[str], 
                                 question: str, entities_context: str, 
                                 history_str: str, entities_explored: Set[str]) -> Tuple[Dict[str, Any], Set[str]]:
        """Process a single exploration round."""
        logger.info(f"Starting exploration round {round_num} with frontier size {len(frontier)}")
        
        # Initialize round data
        round_exploration = {"round": round_num, "expansions": []}
        new_frontier_candidates = set()
        
        # Submit expansion tasks to thread pool
        future_to_entity = {
            self.executor.submit(self._expand_entity, entity, question, entities_context, history_str): entity
            for entity in frontier
        }
        
        # Process expansion results
        expansion_occurred = False
        processed_entities = set()
        
        for future in concurrent.futures.as_completed(future_to_entity):
            entity = future_to_entity[future]
            processed_entities.add(entity)
            
            try:
                expansion = future.result()
                if not expansion or not expansion.get("relations"):
                    continue
                    
                expansion_occurred = True
                round_exploration["expansions"].append(expansion)
                
                # Process targets from this expansion for the next frontier
                new_frontier_candidates.update(
                    self._process_expansion_targets(expansion, entities_explored)
                )
                
            except Exception as e:
                logger.error(f"Error processing expansion for entity '{entity}': {e}", exc_info=True)
                
        return round_exploration, new_frontier_candidates

    def _process_expansion_targets(self, expansion: Dict[str, Any], entities_explored: Set[str]) -> Set[str]:
        """Process targets from an expansion and identify candidates for the next frontier."""
        new_candidates = set()
        
        for relation_info in expansion.get("relations", []):
            for target in relation_info.get("targets", []):
                if not isinstance(target, str):
                    continue
                    
                if target not in entities_explored and not target.startswith(("m.", "g.")):
                    new_candidates.add(target)
                    
        return new_candidates

    def _handle_coded_entities(self, expansion: Dict[str, Any], 
                             entities_explored: Set[str], 
                             context: str, history: str,
                             question: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Handle coded entities (m., g.) with special expansion."""
        secondary_expansions = []
        secondary_candidates = set()
        
        for relation_info in expansion.get("relations", []):
            for target in relation_info.get("targets", []):
                if not isinstance(target, str) or target in entities_explored:
                    continue
                    
                if target.startswith(("m.", "g.")):
                    entities_explored.add(target)
                    try:
                        secondary_expansion = self._expand_entity(target, question, context, history)
                        if secondary_expansion and secondary_expansion.get("relations"):
                            secondary_expansions.append(secondary_expansion)
                            
                            # Process targets from secondary expansion
                            for sec_rel_info in secondary_expansion.get("relations", []):
                                for sec_target in sec_rel_info.get("targets", []):
                                    if isinstance(sec_target, str) and sec_target not in entities_explored:
                                        if not sec_target.startswith(("m.", "g.")):
                                            secondary_candidates.add(sec_target)
                    except Exception as e:
                        logger.error(f"Error in secondary expansion for '{target}': {e}")
                        
        return secondary_expansions, secondary_candidates

    def explore_knowledge_graph(self, question: str, start_entities: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
        """Explore the knowledge graph starting from the given entities."""
        exploration_history = []
        entities_explored = set(start_entities)
        entities_in_context = set(start_entities)
        frontier = list(start_entities)
        answer_found = False
        
        for round_num in range(1, self.max_rounds + 1):
            if not frontier:
                logger.info("Frontier is empty, stopping exploration")
                break
                
            # Prepare context for entity expansion
            entities_context = ", ".join(sorted(list(entities_in_context)))
            history_str = self.format_exploration_history(exploration_history)
            
            # Process current round
            round_exploration, new_frontier_candidates = self._process_exploration_round(
                round_num, frontier, question, entities_context, history_str, entities_explored
            )
            
            # If no expansions occurred in this round, stop exploration
            if not round_exploration.get("expansions"):
                logger.info(f"No successful expansions in round {round_num}")
                break
                
            # Add the round to exploration history
            exploration_history.append(round_exploration)
            
            # Update explored entities and prepare next frontier
            new_frontier_list = list(new_frontier_candidates - entities_explored)
            entities_explored.update(new_frontier_list)
            
            # Apply entity selection for the next frontier if needed
            frontier = self._select_frontier_entities(question, new_frontier_list)
            logger.info(f"Next round frontier size: {len(frontier)}")
            
            # Check if we can answer the question after this round
            answer_result = self.check_if_can_answer(question, start_entities, exploration_history)
            if answer_result.get("can_answer", False):
                logger.info(f"Found answer after round {round_num}")
                exploration_history[-1]["answer_found"] = answer_result
                answer_found = True
                break
                
        return exploration_history, answer_found

    def get_fallback_answer(self, question: str) -> Dict[str, str]:
        """Generate a fallback answer when no definitive answer is found."""
        logger.info(f"Generating fallback answer for: {question}")
        
        try:
            fallback_output = self._generate_model_output("fallback_answer", question=question)
            if not fallback_output:
                return self._create_default_fallback()
                
            result = self.parser.parse_final_answer(fallback_output)
            
            fallback_answer = {
                "answer": result.get("answer", "I cannot answer this question based on the available knowledge."),
                "reasoning": result.get("reasoning", "The knowledge graph exploration did not yield sufficient information.")
            }
            
            # Replace default parser error messages if present
            if fallback_answer["answer"] == "Could not parse answer.":
                fallback_answer["answer"] = "I cannot answer this question based on the available knowledge."
                
            if fallback_answer["reasoning"] == "Could not parse reasoning.":
                fallback_answer["reasoning"] = "The knowledge graph exploration did not yield sufficient information."
                
        except Exception as e:
            logger.error(f"Error generating fallback answer: {e}", exc_info=True)
            fallback_answer = self._create_default_fallback()
            
        self._save_unanswerable_question(question)
        return fallback_answer

    def _create_default_fallback(self) -> Dict[str, str]:
        """Create a default fallback answer."""
        return {
            "answer": "I am sorry, but I couldn't find a definitive answer to your question.",
            "reasoning": "The knowledge graph lacks the required information to answer this question."
        }

    def _save_unanswerable_question(self, question: str) -> None:
        """Save unanswerable questions to a log file for future analysis."""
        log_dir = "logs/unanswerable_questions"
        try:
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(log_dir, f"unanswerable_{date_str}.log")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {question}\n")
                
        except Exception as e:
            logger.error(f"Error saving unanswerable question: {e}", exc_info=True)

    def process_question(self, data: Dict[str, Any], processed_ids: Set[str]) -> Optional[Dict[str, Any]]:
        """Process a question and return the answer with exploration details."""
        question = data.get("question")
        ground_truth = data.get("answer")
        question_id = data.get("id", f"unknown_id_{hash(question)}")
        
        if question_id in processed_ids:
            logger.info(f"Question {question_id} already processed, skipping")
            return None
            
        start_entities = self._normalize_start_entities(data)
        
        try:
            # Explore knowledge graph
            exploration_history, answer_found = self.explore_knowledge_graph(question, start_entities)
            
            # Determine final answer
            if answer_found and exploration_history:
                final_answer = exploration_history[-1].get("answer_found", {})
                fallback_used = False
                logger.info(f"[{question_id}] Answer found during exploration")
            elif exploration_history:
                # One final check for answer
                final_answer = self.check_if_can_answer(question, start_entities, exploration_history)
                fallback_used = not final_answer.get("can_answer", False)
                
                if fallback_used:
                    fallback = self.get_fallback_answer(question)
                    final_answer = self._create_fallback_answer_structure(fallback)
                    logger.info(f"[{question_id}] Using fallback after exploration")
                else:
                    logger.info(f"[{question_id}] Answer obtained from final reasoning")
            else:
                # No exploration occurred
                fallback = self.get_fallback_answer(question)
                final_answer = self._create_fallback_answer_structure(fallback)
                fallback_used = True
                logger.info(f"[{question_id}] Using fallback due to no exploration")
                
            # Format and return result
            return self._create_question_result(
                question_id, question, ground_truth, start_entities,
                final_answer, exploration_history, answer_found, fallback_used
            )
            
        except Exception as e:
            logger.error(f"Critical error processing question {question_id}: {e}", exc_info=True)
            return self._create_error_response(question, question_id, ground_truth, start_entities, str(e))

    def _normalize_start_entities(self, data: Dict[str, Any]) -> List[str]:
        """Normalize and validate start entities from data."""
        start_entities_raw = data.get("q_entity", [])
        if not isinstance(start_entities_raw, list):
            start_entities_raw = [start_entities_raw]
            
        return [str(e) for e in start_entities_raw if e and isinstance(e, (str, int, float))]

    def _create_fallback_answer_structure(self, fallback: Dict[str, str]) -> Dict[str, Any]:
        """Create a structured answer from fallback response."""
        return self._create_answer_result(
            can_answer=False,
            answer=fallback.get("answer", ""),
            reasoning=fallback.get("reasoning", ""),
            verification="Fallback used due to insufficient information."
        )

    def _create_question_result(self, question_id: str, question: str, ground_truth: Any, 
                               start_entities: List[str], answer_details: Dict[str, Any],
                               exploration_history: List[Dict[str, Any]], 
                               answer_found: bool, fallback_used: bool) -> Dict[str, Any]:
        """Create a standardized question result dictionary."""
        prediction_text = self._format_prediction(answer_details)
        
        return {
            "id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "start_entities": start_entities,
            "prediction": prediction_text,
            "reasoning": answer_details.get("reasoning_path", ""),
            "verification": answer_details.get("verification", ""),
            "is_verified": answer_details.get("is_verified", False),
            "exploration_history": exploration_history,
            "answer_found_during_exploration": answer_found,
            "fallback_used": fallback_used,
            "structured_answer": answer_details
        }

    def _format_prediction(self, answer_details: Dict[str, Any]) -> str:
        """Format the prediction text from answer details."""
        prediction_entities = answer_details.get("answer_entities", [])
        
        if prediction_entities and all(isinstance(e, str) for e in prediction_entities):
            return ", ".join(prediction_entities)
        else:
            return answer_details.get("answer_sentence", "")

    def _create_error_response(self, question: str, question_id: str, 
                              ground_truth: Any, start_entities: List[str], 
                              error_msg: str) -> Dict[str, Any]:
        """Create an error response when processing fails."""
        fallback = self.get_fallback_answer(question)
        
        return {
            "id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "start_entities": start_entities,
            "prediction": fallback.get("answer", "Error during processing"),
            "reasoning": f"Error occurred: {error_msg}. Fallback: {fallback.get('reasoning', '')}",
            "verification": "Error occurred during processing.",
            "is_verified": False,
            "exploration_history": [],
            "answer_found_during_exploration": False,
            "fallback_used": True,
            "error": error_msg,
            "structured_answer": self._create_answer_result(
                can_answer=False,
                answer=fallback.get("answer", "Error during processing"),
                reasoning=f"Error occurred: {error_msg}",
                verification="Error occurred during processing."
            )
        }
