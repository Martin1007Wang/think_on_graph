from typing import Any, Dict, List, Tuple, Union, Optional, Pattern
import re
import json
import logging

# Added logger setup
logger = logging.getLogger(__name__)
# Configure logger if needed (e.g., level, handler) - assumes basic config elsewhere

# Constants for regex patterns
RELATION_PATTERN: Pattern[str] = re.compile(r'REL_\d+')
PATH_PATTERN: Pattern[str] = re.compile(r'PATH_\d+')
ENTITY_SCORE_PATTERN: Pattern[str] = re.compile(r'\[([^\]]+)\]:\s*(\d+(?:\.\d+)?)')

class LLMOutputParser:
    """解析语言模型输出。"""    
    
    @staticmethod
    def parse_relations(llm_output: Union[str, List[str]], available_relations: List[str]) -> List[str]:
        def parse_single_output(output: str) -> List[str]:
            selected = []
            rel_ids = RELATION_PATTERN.findall(output)
            if not rel_ids:
                logger.debug("Could not find REL_ID format in relation selection output, attempting fallback parsing.")
                lines = output.strip().split('\n')
                for line in lines:
                    line = line.strip().lower()
                    for relation in available_relations:
                        if relation.lower() in line:
                            selected.append(relation)
                            break
            else:
                selected = list(dict.fromkeys(rel_ids))
            return list(dict.fromkeys(selected))
        if isinstance(llm_output, str):
            return parse_single_output(llm_output)
        logger.warning("Parsing list of relation outputs is not fully implemented for REL_ID format. Processing first output only.")
        if llm_output:
            return parse_single_output(llm_output[0])
        else:
            return []

    @staticmethod
    def parse_paths(llm_output: str, available_paths: list[str]) -> list[str]:
        path_ids = PATH_PATTERN.findall(llm_output)
        seen = set()
        selected = []
        available_set = set(available_paths)
        for pid in path_ids:
            if pid not in seen and pid in available_set:
                seen.add(pid)
                selected.append(pid)
        return selected


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
        default_failure_output = {
            "can_answer": False,
            "reasoning_path": "LLM output parsing failed.",
            "answer_entities": [],
            "analysis": "LLM output parsing failed.",
        }
        
        if not isinstance(output, str) or not output.strip():
            logger.warning("Received empty or non-string output for reasoning parsing.")
            return default_failure_output
            
        try:
            s = output.strip()
            if s.startswith('```json') and s.endswith('```'):
                json_str = s[7:-3].strip()
            else:
                json_match = re.search(r'```json\n(.*?)\n```', s, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    json_str = s
                    if not (json_str.startswith('{') and json_str.endswith('}')):
                        raise json.JSONDecodeError("Output doesn't start/end with braces", json_str, 0)
            try:
                parsed_json = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r'"reasoning_path"\s*:\s*([^"\{\[].*?)(?=,|\n|\r|\})', r'"reasoning_path": "\1"', json_str)
                json_str = re.sub(r'"analysis"\s*:\s*([^"\{\[].*?)(?=,|\n|\r|\})', r'"analysis": "\1"', json_str)
                json_str = re.sub(r',\s*\}', '}', json_str)  # 修复尾部逗号
                parsed_json = json.loads(json_str)
            answer_entities = parsed_json.get("answer_entities", [])
            if not isinstance(answer_entities, list):
                answer_entities = []
            else:
                answer_entities = [str(x) for x in answer_entities if x is not None]
            return {
                "can_answer": parsed_json.get("can_answer") is True,
                "reasoning_path": str(parsed_json.get("reasoning_path", default_failure_output["reasoning_path"])),
                "answer_entities": answer_entities,
                "analysis": str(parsed_json.get("analysis", "")),
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM reasoning output as JSON: {e}\nRaw output:\n{output}")
            return default_failure_output
        except Exception as e:
            logger.error(f"Unexpected error parsing reasoning output: {e}\nRaw output:\n{output}")
            return default_failure_output

    @staticmethod
    def parse_final_answer(output: str) -> Dict[str, str]:
        lines = output.strip().split('\n')
        answer = "Could not parse answer." # Default if not found
        reasoning = "Could not parse reasoning." # Default if not found
        answer_found = False
        reasoning_found = False
        
        for line in lines:
            line_strip = line.strip()
            if line_strip.lower().startswith("answer:") and not answer_found:
                answer = line_strip[len("answer:"):].strip()
                answer_found = True
            elif line_strip.lower().startswith("reasoning:") and not reasoning_found:
                reasoning = line_strip[len("reasoning:"):].strip()
                reasoning_found = True
            if answer_found and reasoning_found:
                break
                
        if not answer_found:
            logger.warning(f"Could not find 'Answer:' prefix in final answer output: {output}")
            
        if not reasoning_found:
            logger.warning(f"Could not find 'Reasoning:' prefix in final answer output: {output}")

        
        return {"answer": answer, "reasoning": reasoning}