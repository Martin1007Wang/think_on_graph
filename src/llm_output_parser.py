from typing import Any, Dict, List, Tuple, Union, Optional, Pattern, Set
import re
import json
import logging

logger = logging.getLogger(__name__)

# Compile regex patterns at the class level for efficiency
RELATION_PATTERN: Pattern[str] = re.compile(r'REL_\d+')
PATH_PATTERN: Pattern[str] = re.compile(r'PATH_\d+')
FACT_PATTERN: Pattern[str] = re.compile(r'FACT_\d+')
CVT_PATTERN: Pattern[str] = re.compile(r'CVT_\d+')
# MODIFIED: Uncommented and pattern assumes IDs like ATTR_0, ATTR_1 etc.
ATTRIBUTE_PATTERN: Pattern[str] = re.compile(r'ATTR_\d+')
ENTITY_ID_PATTERN: Pattern[str] = re.compile(r'ENT_\d+') # Added pattern for entity IDs if parse_entities uses it


class LLMOutputParser:
    """
    Optimized static methods for parsing various types of LLM outputs.
    Handles ID extraction, ranked lists, JSON blocks, and keyword answers.
    """

    @staticmethod
    def _ensure_string(llm_output: Union[str, List[str]], caller: str) -> Optional[str]:
        """Ensure the input is a single stripped string, joining lists if necessary."""
        if isinstance(llm_output, str):
            output_text = llm_output
        elif isinstance(llm_output, list):
            # Ensure all elements are strings before joining
            try:
                output_text = "\n".join(map(str, llm_output))
            except Exception as e:
                 logger.error(f"[{caller}] Error converting list elements to string before joining: {e}")
                 return None
        else:
            logger.warning(f"[{caller}] Unexpected type for llm_output: {type(llm_output)}. Expected str or List[str].")
            return None
        return output_text.strip()

    @staticmethod
    def parse_ids(llm_output: Union[str, List[str]], available_ids: Union[List[str], Set[str]], pattern: Pattern, id_type: str = "item") -> List[str]:
        """
        (Optimized) Parses LLM output to extract specific IDs matching a pattern.
        Validates against available IDs and ensures uniqueness.

        Args:
            llm_output: Raw LLM output string or list of strings.
            available_ids: A list or set of valid IDs that could be selected.
            pattern: Compiled regex pattern to find individual IDs (e.g., r'REL_\d+').
            id_type: String description of the ID type for logging (e.g., "relation").

        Returns:
            A list of unique, valid IDs found in the output, preserving order of first appearance.
        """
        caller_name = f"parse_{id_type}_ids"
        output_text = LLMOutputParser._ensure_string(llm_output, caller_name)

        # Use set for efficient lookup, handle list or set input
        available_set = set(available_ids) if isinstance(available_ids, (list, set)) else set()

        if not output_text or not available_set:
            # Log only if input text was present but no IDs were available
            # if output_text: logger.debug(f"[{caller_name}] No available IDs provided for validation.")
            return []

        # 1. Try parsing structured output: "Selected {id_type}s: [ID1, ID2]" (case-insensitive)
        structured_pattern_str = r"Selected\s+" + re.escape(id_type) + r"s?:\s*\[(.*?)\]"
        structured_match = re.search(structured_pattern_str, output_text, re.IGNORECASE | re.DOTALL)

        text_to_search = output_text # Default: search the whole string
        ids_part = "" # Content within brackets if structured format found

        if structured_match:
            ids_part = structured_match.group(1).strip()
            if not ids_part: # Handle empty list "[]"
                logger.info(f"[{caller_name}] Parser found empty selection 'Selected {id_type}s: []'.")
                return []
            text_to_search = ids_part # Search only within the brackets
            logger.debug(f"[{caller_name}] Found structured selection format. Parsing content: '{ids_part}'")

        # 2. Find all IDs matching the pattern in the determined search scope
        try:
            found_ids = pattern.findall(text_to_search)
        except TypeError as e:
             logger.error(f"[{caller_name}] Regex pattern error ({pattern.pattern}) or invalid text_to_search type ({type(text_to_search)}): {e}")
             return []

        if not found_ids:
            # Log only if something was expected (non-empty brackets or non-empty output)
            if structured_match and ids_part:
                logger.warning(f"[{caller_name}] Found 'Selected {id_type}s: [...]' but no IDs matching pattern '{pattern.pattern}' inside: '{ids_part}'")
            elif not structured_match and output_text:
                 logger.debug(f"[{caller_name}] No {id_type} IDs matching pattern '{pattern.pattern}' found in output.")
            # If structured_match and !ids_part (i.e., "[]"), already logged empty selection
            return []

        # 3. Validate found IDs and ensure uniqueness (CASE-SENSITIVE validation)
        selected_ids = []
        seen_ids = set()
        for item_id in found_ids:
            if not item_id: continue # Skip empty matches from regex if possible

            # --- Case-Sensitive Check (likely correct for IDs) ---
            if item_id in available_set:
                if item_id not in seen_ids:
                    selected_ids.append(item_id)
                    seen_ids.add(item_id)
            # --- End Case-Sensitive Check ---
            # Optional: Add more detailed logging for mismatches if needed for debugging
            # elif logger.isEnabledFor(logging.DEBUG):
            #      logger.debug(f"[{caller_name}] Parsed {id_type} ID '{item_id}' is not in the available set.")

        if not selected_ids and found_ids:
            logger.warning(f"[{caller_name}] Found raw IDs {found_ids}, but none were valid (present in available_ids) or unique.")

        logger.debug(f"[{caller_name}] Selected {id_type} IDs: {selected_ids}")
        return selected_ids

    # --- Specific ID Parsers ---

    @staticmethod
    def parse_relations(llm_output: Union[str, List[str]], item_dict: Dict[str, str]) -> List[str]:
        """Parses relation IDs (REL_*) from LLM output."""
        return LLMOutputParser.parse_ids(llm_output, list(item_dict.keys()), RELATION_PATTERN, "relation")

    @staticmethod
    def parse_paths(llm_output: Union[str, List[str]], item_dict: Dict[str, str]) -> List[str]:
        """Parses path IDs (PATH_*) from LLM output."""
        return LLMOutputParser.parse_ids(llm_output, list(item_dict.keys()), PATH_PATTERN, "path")

    @staticmethod
    def parse_facts(llm_output: Union[str, List[str]], item_dict: Dict[str, str]) -> List[str]:
        """Parses fact IDs (FACT_*) from LLM output."""
        return LLMOutputParser.parse_ids(llm_output, list(item_dict.keys()), FACT_PATTERN, "fact")

    @staticmethod
    def parse_cvts(llm_output: Union[str, List[str]], item_dict: Dict[str, str]) -> List[str]:
        """Parses CVT node IDs (CVT_*) from LLM output."""
        return LLMOutputParser.parse_ids(llm_output, list(item_dict.keys()), CVT_PATTERN, "cvt")

    # MODIFIED: Renamed method to match expected usage
    @staticmethod
    def parse_attribute_selection(llm_output: Union[str, List[str]], item_dict: Dict[str, str]) -> List[str]:
        """
        Parses selected attribute IDs (ATTR_*) or potentially the full attribute string
        ("Rel -> Tgt") from LLM output. This implementation uses parse_ids assuming
        the LLM returns ATTR_X IDs. If the LLM returns full strings, a different
        parsing logic would be needed here (e.g., extracting lines/strings and validating
        them against item_dict.values()).
        """
        # This currently assumes the LLM is asked to return ATTR_X IDs.
        # If the prompt asks for the full "Rel -> Tgt" string, this parser needs adjustment.
        logger.debug("Parsing attribute selection using ID pattern (ATTR_)...")
        return LLMOutputParser.parse_ids(llm_output, list(item_dict.keys()), ATTRIBUTE_PATTERN, "attribute")


    # --- Other Parsers ---

    @staticmethod
    def parse_entities(output: Union[str, List[str]], item_dict: Dict[str, str]) -> List[Tuple[str, float]]:
        """
        (Optimized) Parses ranked entities with scores from LLM output.
        Validates parsed entity names against the original candidate names.

        Args:
            output: Raw LLM output string or list of strings.
            item_dict: Dictionary mapping temporary IDs (e.g., "ENT_0") to the
                       original candidate entity names (e.g., "Ohio").

        Returns:
            A list of tuples: (original_entity_name, score), sorted by score descending.
            Returns only the names if scores are not parseable or format is different.
        """
        caller_name = "parse_entities"
        output_text = LLMOutputParser._ensure_string(output, caller_name)
        if output_text is None:
            return [] # Return empty list for no input

        entity_scores: List[Tuple[str, float]] = []
        # Regex expects lines like: Entity Name: SCORE or [Entity Name]: SCORE
        # Allows variations, handles simple float/int. Group 1: Name, Group 2: Score
        ranking_pattern = re.compile(r"^\s*\[?(.+?)\]?:\s*(\d+(?:\.\d+)?)\s*$", re.MULTILINE)

        # Build lookup from lowercased original names to original names for validation
        candidate_entities = [str(v) for v in item_dict.values() if v is not None] # Ensure names are strings
        if not candidate_entities:
             logger.warning(f"[{caller_name}] No candidate entities provided in item_dict values for validation.")
             # If no candidates, we cannot validate names. Consider returning raw parsed names/scores or empty list.
             # Let's proceed but without validation in this edge case.
             entity_lookup_lower = {}
             should_validate = False
        else:
             entity_lookup_lower = {name.lower(): name for name in candidate_entities}
             should_validate = True

        found_original_entities: Set[str] = set() # Track added original names for uniqueness

        # Attempt to parse ranked list first
        matches = ranking_pattern.finditer(output_text)
        parsed_ranked_list = False

        for match in matches:
            parsed_ranked_list = True # Mark that we found the ranked pattern
            try:
                entity_name_parsed = match.group(1).strip() if match.group(1) else ""
                score_str = match.group(2).strip() if match.group(2) else ""
                if not entity_name_parsed or not score_str: continue

                score = float(score_str)

                original_entity_name = None
                if should_validate:
                    original_entity_name = entity_lookup_lower.get(entity_name_parsed.lower())
                else: # If not validating, use the parsed name directly (less safe)
                    original_entity_name = entity_name_parsed

                if original_entity_name:
                    if original_entity_name not in found_original_entities:
                        entity_scores.append((original_entity_name, score))
                        found_original_entities.add(original_entity_name)
                elif logger.isEnabledFor(logging.DEBUG) and should_validate:
                        logger.debug(f"[{caller_name}] Parsed entity '{entity_name_parsed}' not found in candidates.")

            except ValueError:
                logger.warning(f"[{caller_name}] Could not parse score '{score_str}' as float for line: {match.group(0)}")
            except Exception as e:
                logger.error(f"[{caller_name}] Error parsing entity ranking line: {match.group(0)}. Error: {e}", exc_info=True)

        # If ranked list pattern wasn't found OR parsing failed to produce results,
        # try parsing as a simple list of IDs/Names (fallback)
        if not entity_scores:
            if parsed_ranked_list: # Pattern matched but validation/parsing failed
                 logger.warning(f"[{caller_name}] Parsed ranked entity lines but none matched candidates or had valid scores. LLM output:\n{output_text[:500]}...")
            else: # Pattern didn't match, try parsing as simple IDs/Names
                logger.debug(f"[{caller_name}] Did not find ranked entity format (Name: Score). Trying to parse as list of entity IDs/names...")
                # Use parse_ids with entity ID pattern AND check against candidate names
                parsed_ids = LLMOutputParser.parse_ids(output_text, list(item_dict.keys()), ENTITY_ID_PATTERN, "entity_id")
                if parsed_ids:
                     # Map valid IDs back to names, add with default score (e.g., 1.0 or 0.0) for consistency
                     default_score = 1.0
                     for entity_id in parsed_ids:
                          original_entity_name = item_dict.get(entity_id)
                          if original_entity_name and original_entity_name not in found_original_entities:
                               entity_scores.append((original_entity_name, default_score))
                               found_original_entities.add(original_entity_name)
                     logger.info(f"[{caller_name}] Parsed entity IDs as fallback: {[e[0] for e in entity_scores]}")
                else:
                     # Final fallback: Check if output directly matches candidate names
                     simple_matches = []
                     for line in output_text.splitlines():
                          cleaned_line = line.strip()
                          if should_validate:
                               original_name = entity_lookup_lower.get(cleaned_line.lower())
                               if original_name and original_name not in found_original_entities:
                                    simple_matches.append((original_name, 1.0))
                                    found_original_entities.add(original_name)
                          elif cleaned_line and cleaned_line not in found_original_entities: # No validation case
                               simple_matches.append((cleaned_line, 1.0))
                               found_original_entities.add(cleaned_line)

                     if simple_matches:
                         entity_scores = simple_matches
                         logger.info(f"[{caller_name}] Parsed entity names directly as fallback: {[e[0] for e in entity_scores]}")
                     else:
                         logger.warning(f"[{caller_name}] Could not parse entities using ranked, ID, or direct name matching.")


        # Sort by score descending (will use default score if fallback parsing occurred)
        entity_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"[{caller_name}] Parsed and sorted entities: {entity_scores}")
        # Return list of (original_name, score)
        return entity_scores


    @staticmethod
    def parse_reasoning_output(output: Union[str, List[str]]) -> Dict[str, Any]:
        """
        (Optimized) Parses LLM output expected to be a JSON object containing reasoning details.
        Handles JSON potentially wrapped in markdown code blocks. Provides defaults.
        """
        caller_name = "parse_reasoning_output"
        default_failure_output = {
            "can_answer": False,
            "reasoning_path": None,
            "answer_entities": [],
            "analysis": "LLM output parsing failed.",
        }
        output_text = LLMOutputParser._ensure_string(output, caller_name)
        if not output_text:
            logger.warning(f"[{caller_name}] Received empty output.")
            default_failure_output["analysis"] = "Received empty output from LLM."
            return default_failure_output

        json_str = output_text
        # Store the original text for later use
        original_text = json_str

        # Robustly find JSON within ```...``` or ```json...``` blocks
        # Handles potential leading/trailing whitespace/newlines around fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL | re.IGNORECASE) \
                   or re.search(r"```(?:json)?\s*(\[.*?\])\s*```", json_str, re.DOTALL | re.IGNORECASE) # Also allow top-level list

        if fence_match:
            json_str = fence_match.group(1).strip()
            logger.debug(f"[{caller_name}] Extracted content from JSON markdown block.")
        else:
            # Attempt to find JSON start/end even without fences as fallback
             json_start = json_str.find('{')
             json_end = json_str.rfind('}') + 1
             if json_start != -1 and json_end != 0:
                  json_str_curly = json_str[json_start:json_end]
                  # Basic check if it looks plausible
                  if json_str_curly.count('{') == json_str_curly.count('}'):
                       json_str = json_str_curly
                       logger.debug(f"[{caller_name}] Found plausible JSON object without markdown fences.")
                  else: json_str = None # Reset if braces don't match
             else: json_str = None # Reset if no curly braces found

             # Try square brackets if curly failed
             if json_str is None:
                  json_start = original_text.find('[')
                  json_end = original_text.rfind(']') + 1
                  if json_start != -1 and json_end != 0:
                      json_str_square = original_text[json_start:json_end]
                      if json_str_square.count('[') == json_str_square.count(']'):
                           json_str = json_str_square
                           logger.debug(f"[{caller_name}] Found plausible JSON array without markdown fences.")

             if json_str is None: # If neither fence nor plausible braces/brackets found
                  logger.error(f"[{caller_name}] Could not find JSON block (fenced or unfenced) in output:\n{output_text[:500]}...")
                  default_failure_output["analysis"] = "Could not locate JSON block in output."
                  return default_failure_output


        try:
            parsed_data = json.loads(json_str)
            if not isinstance(parsed_data, dict):
                logger.error(f"[{caller_name}] Parsed JSON is not a dictionary: {type(parsed_data)}")
                default_failure_output["analysis"] = f"Parsed JSON was not a dictionary (type: {type(parsed_data)})."
                return default_failure_output

            # Extract fields with defaults and type validation/coercion
            can_answer = bool(parsed_data.get("can_answer", False))

            reasoning_path = parsed_data.get("reasoning_path", parsed_data.get("reasoning_paths")) # Allow alternate key
            # Downstream might expect string or list - keep as parsed for now, or coerce:
            # if reasoning_path is not None: reasoning_path = str(reasoning_path)

            answer_entities = parsed_data.get("answer_entities", [])
            if isinstance(answer_entities, list):
                # Ensure elements are strings, filter None
                answer_entities = [str(e) for e in answer_entities if e is not None]
            else:
                logger.warning(f"[{caller_name}] 'answer_entities' field is not a list (type: {type(answer_entities)}), defaulting to empty list.")
                answer_entities = []

            analysis = str(parsed_data.get("analysis", "")) # Ensure string

            parsed_result = {
                "can_answer": can_answer,
                "reasoning_path": reasoning_path, # Keep original type for flexibility
                "answer_entities": answer_entities,
                "analysis": analysis,
            }
            logger.debug(f"[{caller_name}] Successfully parsed reasoning output: {parsed_result}")
            return parsed_result

        except json.JSONDecodeError as e:
            logger.error(f"[{caller_name}] Failed to decode JSON: {e}\nAttempted JSON string:\n{json_str[:500]}...")
            default_failure_output["analysis"] = f"Failed to decode JSON: {e}"
            return default_failure_output
        except Exception as e:
            logger.error(f"[{caller_name}] Unexpected error processing parsed JSON data: {e}\nAttempted JSON string:\n{json_str[:500]}", exc_info=True)
            default_failure_output["analysis"] = f"Unexpected error during JSON processing: {e}"
            return default_failure_output


    @staticmethod
    def parse_fallback_answer(output: Union[str, List[str]]) -> Dict[str, str]:
        caller_name = "parse_fallback_answer"
        default_answer = "Could not parse answer from JSON."
        default_reasoning = "Could not parse reasoning from JSON analysis."
        final_result = {"answer": default_answer, "reasoning": default_reasoning}

        output_text = LLMOutputParser._ensure_string(output, caller_name)
        if not output_text:
            logger.warning(f"[{caller_name}] Received empty output.")
            return final_result # 输入为空，返回默认值

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL | re.IGNORECASE)

        json_string = ""
        if match:
            json_string = match.group(1).strip() # 提取花括号内的内容
            logger.debug(f"[{caller_name}] Extracted JSON string from markdown fences.")
        else:
            # 回退：检查整个输出是否可能就是一个 JSON 字符串（没有代码块包裹）
            stripped_output = output_text.strip()
            if stripped_output.startswith('{') and stripped_output.endswith('}'):
                 logger.warning(f"[{caller_name}] No JSON markdown fences found. Attempting to parse entire input.")
                 json_string = stripped_output
            else:
                 logger.error(f"[{caller_name}] Failed to find JSON markdown fences or valid JSON structure in output. Raw: '{output_text[:200]}...'")
                 final_result["reasoning"] = "Output format error: Expected JSON in markdown fences."
                 return final_result # 格式不匹配，返回默认值

        if not json_string:
             logger.error(f"[{caller_name}] Extracted JSON string is empty.")
             final_result["reasoning"] = "Output format error: Extracted JSON content is empty."
             return final_result

        # 2. 解析 JSON 字符串
        try:
            parsed_data = json.loads(json_string)
            if not isinstance(parsed_data, dict):
                logger.error(f"[{caller_name}] Parsed JSON is not a dictionary (type: {type(parsed_data)}).")
                final_result["reasoning"] = "Parsed JSON content is not a valid object."
                return final_result

        except json.JSONDecodeError as e:
            logger.error(f"[{caller_name}] Failed to decode JSON string. Error: {e}. String approx: '{json_string[:200]}...'")
            final_result["reasoning"] = f"JSON parsing error: {e}"
            return final_result # JSON 格式错误，返回默认值
        except Exception as e: # 捕获其他可能的加载错误
             logger.error(f"[{caller_name}] Unexpected error loading JSON data. Error: {e}")
             final_result["reasoning"] = f"Unexpected error during JSON processing: {e}"
             return final_result

        # 3. 从解析后的字典中提取字段
        #    答案: 来自 "answer_entities" 列表
        answer_entities = parsed_data.get("answer_entities")
        if isinstance(answer_entities, list) and answer_entities: # 确保是列表且不为空
            try:
                # 将列表中的所有元素（确保是字符串）用逗号和空格连接起来
                final_result["answer"] = ", ".join(map(str, answer_entities))
            except Exception as e:
                logger.error(f"[{caller_name}] Error converting answer_entities list items to string: {e}")
                # 保留默认答案
        elif answer_entities is not None:
             logger.warning(f"[{caller_name}] 'answer_entities' key found but is not a non-empty list: {answer_entities!r}")
        else:
             logger.warning(f"[{caller_name}] 'answer_entities' key not found in parsed JSON.")


        #    推理: 来自 "analysis" 字符串
        analysis = parsed_data.get("analysis")
        if isinstance(analysis, str) and analysis.strip(): # 确保是字符串且不为空白
            final_result["reasoning"] = analysis.strip()
        elif analysis is not None:
             logger.warning(f"[{caller_name}] 'analysis' key found but is not a non-empty string: {analysis!r}")
        else:
            # 如果 analysis 缺失，可以考虑使用 reasoning_path 作为备选
            reasoning_path = parsed_data.get("reasoning_path")
            if isinstance(reasoning_path, str) and reasoning_path.strip():
                 logger.info(f"[{caller_name}] 'analysis' missing, using 'reasoning_path' as reasoning.")
                 final_result["reasoning"] = reasoning_path
            else:
                 logger.warning(f"[{caller_name}] 'analysis' key not found or invalid, and no valid 'reasoning_path' fallback.")


        logger.debug(f"[{caller_name}] Parsed final answer: answer='{final_result['answer'][:100]}...', reasoning='{final_result['reasoning'][:100]}...'")
        return final_result