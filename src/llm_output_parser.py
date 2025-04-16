from typing import Any, Dict, List, Tuple, Union
import re
import json
import logging

# Added logger setup
logger = logging.getLogger(__name__)
# Configure logger if needed (e.g., level, handler) - assumes basic config elsewhere

class LLMOutputParser:
    """解析语言模型输出。"""    
    
    @staticmethod
    def parse_relations(llm_output: Union[str, List[str]], available_relations: List[str]) -> List[str]:
        def parse_single_output(output: str) -> List[str]:
            selected = []
            # Simple REL_ID extraction assuming LLM follows instructions
            rel_ids = re.findall(r'REL_\d+', output)
            # Map REL_ID back to relation name if needed, assuming a dict is available
            # This part depends on how relation_dict is managed or passed.
            # For now, let's assume available_relations maps directly or REL_IDs are sufficient downstream
            # If mapping needed:
            # relation_dict = {f"REL_{i}": rel for i, rel in enumerate(available_relations)}
            # selected = [relation_dict[rel_id] for rel_id in rel_ids if rel_id in relation_dict]

            # Fallback or alternative parsing if REL_ID format fails (less robust)
            if not rel_ids:
                logger.debug("Could not find REL_ID format in relation selection output, attempting fallback parsing.")
                lines = output.strip().split('\n')
                for line in lines:
                    line = line.strip().lower() # Lowercase for matching
                    # Try to match relation names directly (less precise)
                    for relation in available_relations:
                        if relation.lower() in line:
                            selected.append(relation)
                            break # Avoid adding multiple times if names overlap in line
            else:
                # If REL_IDs are found, assume they are the primary source
                # Here you might map them back to names if required by downstream code
                # For simplicity now, let's return the found IDs or handle mapping elsewhere
                selected = list(dict.fromkeys(rel_ids)) # Return unique REL_IDs found

            # Ensure uniqueness if fallback parsing added duplicates
            return list(dict.fromkeys(selected))

        if isinstance(llm_output, str):
            return parse_single_output(llm_output)
        
        # Handling list output (e.g., from multiple beams) - Requires adaptation
        # The original logic averaged scores based on parsing names.
        # If using REL_IDs, a simpler approach might be consensus voting on IDs.
        logger.warning("Parsing list of relation outputs is not fully implemented for REL_ID format. Processing first output only.")
        if llm_output:
            return parse_single_output(llm_output[0])
        else:
            return []
        # -- Original averaging logic (needs adaptation for REL_IDs) ---
        # relation_scores = {}
        # total_outputs = len(llm_output)
        # for output in llm_output:
        #     selected = parse_single_output(output) # This now returns REL_IDs or names
        #     for item in selected: # item could be REL_ID or relation name
        #         relation_scores[item] = relation_scores.get(item, 0) + 1
        # scored_relations = [(item, count / total_outputs) for item, count in relation_scores.items()]
        # scored_relations.sort(key=lambda x: x[1], reverse=True)
        # return [item for item, _ in scored_relations]
        # --- End Original Logic ---

    @staticmethod
    def parse_entities(output: str, frontier: List[str]) -> List[Tuple[str, float]]:
        entity_scores = []
        # Regex to capture "[Entity]: Score" format, allowing for variations in spacing
        # It captures the entity name (group 1) and the score (group 2)
        pattern = re.compile(r'^s*\[?([ws.-]+?)\]?:\s*(\d+(?:.\d+)?)$', re.MULTILINE)

        matches = pattern.finditer(output)
        found_entities = set()

        for match in matches:
            try:
                entity_name = match.group(1).strip()
                score = float(match.group(2))

                # Find the canonical entity name from the frontier (case-insensitive match)
                matched_frontier_entity = None
                for frontier_entity in frontier:
                    if frontier_entity.lower() == entity_name.lower():
                        matched_frontier_entity = frontier_entity
                        break

                if matched_frontier_entity and matched_frontier_entity not in found_entities:
                    entity_scores.append((matched_frontier_entity, score))
                    found_entities.add(matched_frontier_entity)
                elif not matched_frontier_entity:
                    logger.warning(f"Parsed entity '{entity_name}' from ranking not found in current frontier: {frontier}")

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse score for entity line: {match.group(0)}. Error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error parsing entity ranking line: {match.group(0)}. Error: {e}")

        if not entity_scores:
            logger.warning(f"Could not parse any valid entity scores from LLM output:\n{output}")
            # Option: Return all frontier entities with default score? Or empty list?
            # Returning empty list seems safer to indicate parsing failure.

        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores

    @staticmethod
    def parse_reasoning_output(output: str) -> Dict[str, Any]:
        """
        Parses the LLM output expected to be in JSON format for reasoning results.

        Args:
            output: The raw string output from the language model.

        Returns:
            A dictionary containing the parsed reasoning results (can_answer, reasoning_path,
            answer_entities, answer_sentence, verification, is_verified) or a default
            dictionary indicating failure if parsing fails.
        """
        default_failure_output = {
            "can_answer": False,
            "reasoning_path": "LLM output parsing failed.",
            "answer_entities": [],
            "answer_sentence": "LLM output parsing failed.",
            "verification": "LLM output parsing failed.",
            "is_verified": False
        }

        if not isinstance(output, str) or not output.strip():
            logger.warning("Received empty or non-string output for reasoning parsing.")
            return default_failure_output

        try:
            # Attempt to find JSON block even if there's surrounding text/markdown
            json_match = re.search(r'```json\n(.*?)\n```', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no markdown block, assume the whole output might be JSON (less reliable)
                json_str = output.strip()
                # Basic check: does it look like JSON?
                if not (json_str.startswith('{') and json_str.endswith('}')):
                    raise json.JSONDecodeError("Output doesn't start/end with braces", json_str, 0)

            parsed_json = json.loads(json_str)

            # Basic validation of the parsed structure
            if not isinstance(parsed_json, dict):
                logger.error(f"Parsed JSON is not a dictionary: {type(parsed_json)}")
                return default_failure_output

            # Ensure essential keys exist and have correct basic types
            # Provide default values matching the 'failure' state if keys are missing or wrong type
            validated_output = {
                "can_answer": parsed_json.get("can_answer") if isinstance(parsed_json.get("can_answer"), bool) else False,
                "reasoning_path": str(parsed_json.get("reasoning_path", default_failure_output["reasoning_path"])),
                "answer_entities": parsed_json.get("answer_entities") if isinstance(parsed_json.get("answer_entities"), list) else [],
                "answer_sentence": str(parsed_json.get("answer_sentence", default_failure_output["answer_sentence"])),
                "verification": str(parsed_json.get("verification", default_failure_output["verification"])),
                "is_verified": parsed_json.get("is_verified") if isinstance(parsed_json.get("is_verified"), bool) else False,
            }

            # Further validation: ensure all items in answer_entities are strings
            if not all(isinstance(item, str) for item in validated_output["answer_entities"]):
                logger.warning(f"answer_entities contains non-string items: {validated_output['answer_entities']}. Clearing list.")
                validated_output["answer_entities"] = []
                # Potentially set can_answer to False if entities are crucial and invalid?
                # validated_output["can_answer"] = False

            logger.debug("Successfully parsed reasoning output JSON.")
            return validated_output

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM reasoning output as JSON: {e}\nRaw output:\n{output}", exc_info=True)
            return default_failure_output
        except Exception as e:
            logger.error(f"Unexpected error parsing reasoning output: {e}\nRaw output:\n{output}", exc_info=True)
            return default_failure_output

    @staticmethod
    def parse_final_answer(output: str) -> Dict[str, str]:
        """解析最终答案输出 (预期为 'Answer: ... Reasoning: ...' 格式).

        Args:
            output: 语言模型生成的最终答案输出

        Returns:
            包含答案和推理的字典
        """
        lines = output.strip().split('\n')
        answer = "Could not parse answer." # Default if not found
        reasoning = "Could not parse reasoning." # Default if not found
        
        # Search for lines starting with the expected prefixes
        answer_found = False
        reasoning_found = False
        
        for line in lines:
            line_strip = line.strip()
            # Use lower() for case-insensitive matching of prefixes
            if line_strip.lower().startswith("answer:") and not answer_found:
                # Take everything after the prefix
                answer = line_strip[len("answer:"):].strip()
                answer_found = True
            elif line_strip.lower().startswith("reasoning:") and not reasoning_found:
                # Take everything after the prefix
                reasoning = line_strip[len("reasoning:"):].strip()
                reasoning_found = True

            # Optimization: stop if both are found
            if answer_found and reasoning_found:
                break
                
        if not answer_found:
            logger.warning(f"Could not find 'Answer:' prefix in final answer output: {output}")
            # Option: use the whole output as answer if no prefix found?
            # answer = output.strip()
            
        if not reasoning_found:
            logger.warning(f"Could not find 'Reasoning:' prefix in final answer output: {output}")

        
        return {"answer": answer, "reasoning": reasoning}