import logging
from typing import Any, List, Dict, Optional, Set, Callable # Added Callable
from src.llm_output_parser import LLMOutputParser # Assume this exists and is correct
# from src.template import KnowledgeGraphTemplates # Add specific type if available
from collections import OrderedDict
from src.utils.data_utils import ExplorationRound

logger = logging.getLogger(__name__)

# Define a more specific type for the model if possible, otherwise use Any
# Example: from some_model_library import BaseModelType
ModelType = Any
TemplateManagerType = Any # Replace with specific class, e.g., KnowledgeGraphTemplates

class ModelInterface:
    """
    Interface for interacting with language models for generation and selection tasks,
    handling prompt formatting, generation calls, and output parsing.
    """
    def __init__(self, model: ModelType, templates: TemplateManagerType, parser: Optional[LLMOutputParser] = None):
        """
        Initializes the ModelInterface.

        Args:
            model: The underlying language model instance. Must have methods like
                   'prepare_model_prompt' and 'generate_sentence'.
            templates: An object responsible for formatting prompts (e.g., KnowledgeGraphTemplates).
                       Must have a 'format_template' method.
            parser: An instance of LLMOutputParser. If None, a default instance is created.
        """
        if not hasattr(model, 'prepare_model_prompt') or not hasattr(model, 'generate_sentence'):
             raise TypeError(f"Model object {model.__class__.__name__} must have 'prepare_model_prompt' and 'generate_sentence' methods.")
        if not hasattr(templates, 'format_template'):
             raise TypeError(f"Templates object {templates.__class__.__name__} must have a 'format_template' method.")

        self.model = model
        self.templates = templates
        self.parser = parser or LLMOutputParser() # Use provided parser or create default

        if not isinstance(self.parser, LLMOutputParser):
             raise TypeError(f"Parser must be an instance of LLMOutputParser, got {type(self.parser)}")

        logger.info(f"ModelInterfaceOptimized initialized with model type: {model.__class__.__name__}")

    def generate_output(self, template_name: str,
                        generation_mode: str = "beam", # Renamed for clarity
                        num_beams: int = 4,
                        **template_args) -> Optional[str]:
        """
        Formats a prompt, generates output using the model, and handles errors.

        Args:
            template_name: The name of the template to use.
            generation_mode: The generation mode for the model.
            num_beams: The number of beams for beam search generation.
            **template_args: Arguments required by the template.

        Returns:
            The generated text output as a string, or None if generation fails.
        """
        log_ctx_parts = [f"{k}='{str(v)[:50]}...'" for k, v in template_args.items() if k in ['entity', 'question', 'cvt_id']]
        log_context = ", ".join(log_ctx_parts) if log_ctx_parts else "No specific context args"
        logger.debug(f"Attempting generation for template='{template_name}'. Context: {log_context}")

        try:
            prompt = self.templates.format_template(template_name, **template_args)
            if not prompt: # Check if formatting failed (e.g., template not found)
                logger.error(f"Failed to format prompt for template '{template_name}' with args: {template_args}")
                return None

            # Assuming model methods exist (checked in __init__)
            model_input = self.model.prepare_model_prompt(prompt)
            output = self.model.generate_sentence(
                model_input,
                temp_generation_mode=generation_mode, # Pass renamed arg
                num_beams=num_beams
            )

            if output is None:
                logger.warning(f"Model generation returned None for template '{template_name}'. Context: {log_context}")
                return None

            logger.debug(f"Generated output length: {len(output)} for template '{template_name}'. Context: {log_context}")
            return output

        except Exception as e:
            logger.error(f"Error during model generation for template '{template_name}'. Context: {log_context}. Error: {e}", exc_info=True)
            return None

    def _select_items(self, item_type: str, items: List[str], max_items: Optional[int],
                      id_prefix: str, template_name: str,
                      parser_method_name: str,
                      fallback_on_error: bool = False, # Default changed to False
                      generation_mode: str = "beam", # Allow overriding generation params
                      num_beams: int = 4,
                      **template_args) -> List[str]:
        """
        Internal helper to select a subset of items using an LLM.

        Handles item formatting, LLM generation, parsing, result mapping, and fallback logic.

        Args:
            item_type: A string describing the items (e.g., "relation", "attribute"). Used for logging and template keys.
            items: The list of string items to select from.
            max_items: The maximum number of items to select. If None or >= len(items), returns all items without calling LLM.
            id_prefix: A prefix used to generate unique IDs for items presented to the LLM (e.g., "REL", "ATTR").
            template_name: The name of the template used for the selection prompt.
            parser_method_name: The name of the method in self.parser to use for parsing the LLM output.
            fallback_on_error: If True, returns the first `max_items` on generation or parsing failure. Defaults to False.
            generation_mode: Generation mode for the LLM call.
            num_beams: Number of beams for the LLM call.
            **template_args: Additional arguments needed for the prompt template.

        Returns:
            A list containing the selected items (strings). Returns an empty list if selection fails
            and fallback_on_error is False.
        """
        if not items:
            logger.debug(f"No {item_type}s provided for selection. Returning empty list.")
            return []

        # If no selection needed (limit is None or not exceeded), return all items
        # Handle max_items == 0 explicitly.
        if max_items is not None and max_items <= 0:
             return []
        if max_items is None or len(items) <= max_items:
            logger.debug(f"Number of {item_type}s ({len(items)}) is within limit ({max_items}), returning all without LLM call.")
            return items

        # --- Prepare items for LLM prompt ---
        # Use OrderedDict to maintain consistent order for IDs if items are duplicated (though list should handle order)
        item_dict = OrderedDict((f"{id_prefix}_{i}", item) for i, item in enumerate(items))
        item_options = "\n".join(f"[{item_id}] {item}" for item_id, item in item_dict.items())

        # Prepare template arguments for the selection prompt
        selection_template_args = template_args.copy()
        selection_template_args.update({
            item_type + "s": item_options, # e.g., relations=...
            item_type + "_ids": ", ".join(item_dict.keys()), # e.g., relation_ids=...
            "max_selection_count": max_items
        })

        # --- Perform LLM Generation and Parsing ---
        selected_output: Optional[str] = None
        parsed_selection: Optional[List[str]] = None # Holds results from parser
        error_occurred = False
        error_message = ""

        try:
            # Step 1: Generate LLM output
            selected_output = self.generate_output(
                template_name,
                generation_mode=generation_mode,
                num_beams=num_beams,
                **selection_template_args
            )
            if selected_output is None:
                raise ValueError(f"LLM generation failed for {item_type} selection.") # Raise specific error

            # Step 2: Parse the output
            if not hasattr(self.parser, parser_method_name):
                raise AttributeError(f"Parser method '{parser_method_name}' not found in {self.parser.__class__.__name__}")

            parser_method: Callable = getattr(self.parser, parser_method_name)
            parsed_selection = parser_method(selected_output, item_dict=item_dict)

            if parsed_selection is None: # Parser explicitly returned None indicating failure
                 raise ValueError(f"Parsing failed for {item_type} selection (parser returned None).")
            if not isinstance(parsed_selection, list):
                 logger.warning(f"Parser method '{parser_method_name}' did not return a list, got {type(parsed_selection)}. Treating as no selection.")
                 parsed_selection = []


        except (ValueError, AttributeError, Exception) as e:
            error_occurred = True
            error_message = str(e)
            logger.error(f"Error during {item_type} selection generation or parsing: {e}", exc_info=isinstance(e, Exception) and not isinstance(e, (ValueError, AttributeError)))


        # --- Process Parsed Results or Handle Fallback ---
        if error_occurred:
            if fallback_on_error:
                logger.warning(f"Using fallback due to error ({error_message}): returning first {max_items} {item_type}s.")
                return items[:max_items]
            else:
                logger.warning(f"Selection failed ({error_message}) and fallback is disabled. Returning empty list.")
                return []

        # --- Map parsed results back to original items (if no error occurred) ---
        final_selections: List[str] = []
        seen_items: Set[str] = set()

        if not parsed_selection: # Handle empty list from parser
             logger.warning(f"LLM output parsed, but no {item_type}s were identified by the parser.")
             # Fallback logic moved here for clarity if needed
        else:
            for selection in parsed_selection:
                item = None
                if selection in item_dict: # Check if parser returned an ID
                    item = item_dict[selection]
                elif selection in items: # Check if parser returned the item string directly
                    # This assumes the parser might return raw strings that were in the original list
                    item = selection
                else:
                    logger.warning(f"Parsed selection '{selection}' is neither a valid ID nor an original item string. Ignoring.")
                    continue

                # Add if valid, unique, and within count limit
                if item and item not in seen_items:
                    final_selections.append(item)
                    seen_items.add(item)
                    if len(final_selections) >= max_items:
                        break # Stop once limit is reached

        # --- Final Decision ---
        if final_selections:
            logger.info(f"LLM selection successful. Selected {len(final_selections)}/{max_items} {item_type}s.")
            return final_selections
        else:
            # Reached if parsing succeeded but yielded no valid/unique items after mapping
            logger.warning(f"Selection processing yielded no valid final {item_type}s (parsed: {parsed_selection}).")
            if fallback_on_error:
                logger.info(f"Using fallback: returning first {max_items} {item_type}s.")
                return items[:max_items]
            else:
                return []


    # --- Public Selection Methods ---
    # Now accept optional generation params and fallback config

    def select_relations(self, entity: str, available_relations: List[str], question: str,
                         history: str = "", max_selection_count: int = 5,
                         fallback_on_error: bool = False, # Configurable fallback
                         generation_mode: str = "beam", num_beams: int = 4) -> List[str]:
        template_name = "relation_selection_with_history" if history else "relation_selection"
        template_args = {"question": question, "entity": entity}
        if history: template_args["history"] = history

        return self._select_items(
            item_type="relation",
            items=available_relations,
            max_items=max_selection_count,
            id_prefix="REL",
            template_name=template_name,
            parser_method_name="parse_relations",
            fallback_on_error=fallback_on_error,
            generation_mode=generation_mode,
            num_beams=num_beams,
            **template_args
        )

    def select_facts(self, entity: str, available_facts: List[str], # Changed type hint
                     question: str, history: str = "",
                     max_selection_count: int = 5,
                     fallback_on_error: bool = False, # Configurable fallback
                     generation_mode: str = "beam", num_beams: int = 4) -> List[str]:
        """Selects relevant facts (previously expanded CVT hops) related to an entity."""
        template_name = "expanded_cvt_hop_selection" # Ensure this template exists
        template_args = {"question": question, "entity": entity}
        if history: template_args["history"] = history

        return self._select_items(
            item_type="fact",
            items=available_facts, # Pass the list directly
            max_items=max_selection_count,
            id_prefix="FACT", # Prefix for potential IDs
            template_name=template_name,
            parser_method_name="parse_facts", # Ensure this parser method exists
            fallback_on_error=fallback_on_error, # Pass config
            generation_mode=generation_mode, # Pass config
            num_beams=num_beams, # Pass config
            **template_args
        )

    def select_entities(self, question: str, candidates: List[str], max_entities: Optional[int],
                        fallback_on_error: bool = False, # Configurable fallback
                        generation_mode: str = "beam", num_beams: int = 4) -> List[str]:
        """Selects relevant entities from a list of candidates based on the question."""
        # max_entities=None or count <= limit is handled by _select_items
        template_args = {"question": question}
        return self._select_items(
            item_type="entity",
            items=candidates,
            max_items=max_entities,
            id_prefix="ENT",
            template_name="entity_selection", # Ensure this template exists
            parser_method_name="parse_entities", # Ensure this parser method exists
            fallback_on_error=fallback_on_error, # Pass config
            generation_mode=generation_mode, # Pass config
            num_beams=num_beams, # Pass config
            **template_args
        )

    def select_attributes(self, source_entity: str, source_relation: str, cvt_id: str,
                          available_attributes: List[str], # List of "Rel -> Tgt" strings
                          question: str, history: str = "",
                          max_selection_count: int = 1, # Default to 1 for attributes usually
                          fallback_on_error: bool = False, # Configurable fallback
                          generation_mode: str = "beam", num_beams: int = 4) -> List[str]:
        """Selects the most relevant attribute (Rel -> Tgt pair) from a CVT node."""
        template_args = {
            "question": question,
            "source_entity": source_entity,
            "source_relation": source_relation,
            "cvt_id": cvt_id,
        }
        if history: template_args["history"] = history

        return self._select_items(
            item_type="attribute",
            items=available_attributes,
            max_items=max_selection_count,
            id_prefix="ATTR",
            template_name="cvt_attribute_selection", # Ensure this template exists
            parser_method_name="parse_attribute_selection", # CRITICAL: Use the correct specific parser method name
            fallback_on_error=fallback_on_error, # Pass config
            generation_mode=generation_mode, # Pass config
            num_beams=num_beams, # Pass config
            **template_args
        )

    # --- Answerability and Fallback ---

    def check_answerability(self, question: str, start_entities: List[str],
                            exploration_history: str) -> Dict[str, Any]:
        """
        Checks if the exploration history is sufficient to answer the question using the predict model.

        Args:
            question: The user's question.
            start_entities: List of starting entity IDs/names.
            exploration_history: A string summarizing the paths explored.

        Returns:
            A dictionary containing at least 'can_answer' (bool) and optionally
            'reasoning_path', 'answer_entities', 'analysis'. Returns a default
            failure dictionary if generation or parsing fails.
        """
        log_ctx = f"Q='{question[:50]}...', Start='{start_entities}'"
        logger.info(f"Checking answerability. {log_ctx}")
        template_name = "reasoning" # Ensure this template exists and is appropriate
        parser_method_name = "parse_reasoning_output" # Parser method for this task

        # Check if parser method exists *before* generation
        if not hasattr(self.parser, parser_method_name):
            logger.error(f"Parser method '{parser_method_name}' not found for answerability check.")
            return {"can_answer": False, "reasoning_path": f"Internal Error: Parser method '{parser_method_name}' missing.", "answer_entities": [], "analysis": ""}

        reasoning_output = self.generate_output(
            template_name,
            # Consider if different generation params are needed here
            # generation_mode="...", num_beams=...,
            question=question,
            entity=", ".join(start_entities), # Join list for prompt template
            exploration_history=exploration_history
        )

        if reasoning_output is None:
            logger.warning(f"Model failed to generate reasoning output for answerability check. {log_ctx}")
            return {"can_answer": False, "reasoning_path": "Failed to generate reasoning.", "answer_entities": [], "analysis": ""}

        try:
            parser_method: Callable = getattr(self.parser, parser_method_name)
            parsed_output = parser_method(reasoning_output)

            if not isinstance(parsed_output, dict) or 'can_answer' not in parsed_output:
                logger.error(f"Invalid structure parsed from reasoning output: {parsed_output}. Raw: '{reasoning_output[:100]}...'")
                return {"can_answer": False, "reasoning_path": "Failed to parse reasoning output structure.", "answer_entities": [], "analysis": ""}

            logger.info(f"Answerability check result: can_answer={parsed_output.get('can_answer')}. {log_ctx}")
            # Ensure all expected keys have default values if missing from parser
            parsed_output.setdefault("reasoning_path", "No reasoning path parsed.")
            parsed_output.setdefault("answer_entities", [])
            parsed_output.setdefault("analysis", "")
            return parsed_output

        except Exception as e:
            logger.error(f"Error parsing reasoning output. Raw: '{reasoning_output[:100]}...'. Error: {e}", exc_info=True)
            return {"can_answer": False, "reasoning_path": f"Failed to parse reasoning output: {e}", "answer_entities": [], "analysis": ""}


    def generate_fallback_answer(self, question: str, exploration_history: List[Any]) -> Dict[str, str]:
        logger.warning(f"Generating fallback answer for question: '{question[:100]}...'")
        template_name = "fallback_answer"
        parser_method_name = "parse_fallback_answer"
        default_answer = "I am sorry, but I couldn't find a definitive answer using the available knowledge."
        default_reasoning = "The knowledge exploration process did not yield a conclusive answer."
        fallback_result = {"answer": default_answer, "reasoning": default_reasoning}

        parser_method: Optional[Callable] = getattr(self.parser, parser_method_name, None)
        if not callable(parser_method):
            logger.error(f"Parser method '{parser_method_name}' not found or not callable.")
            fallback_result["reasoning"] = f"Internal Configuration Error: Required parser method '{parser_method_name}' is unavailable."
            return fallback_result # 关键组件缺失，提前返回

        # 优化：将复杂的格式化逻辑移出或接受简化表示
        history_text = "No exploration history available."
        if exploration_history:
            history_text = f"Exploration history with {len(exploration_history)} rounds is available (details omitted for fallback)."
            logger.debug("Using simplified/generic history representation for fallback LLM call.")
        else:
            logger.warning("No exploration history provided for fallback.")

        try:
            fallback_output = self.generate_output(
                template_name,
                question=question,
                exploration_history=history_text # 使用简化后的历史文本
            )
            if fallback_output is None:
                logger.error("LLM failed to generate fallback answer (returned None).")
                return fallback_result # LLM 失败，返回默认值

        except Exception as e:
            logger.error(f"Exception during LLM call for fallback: {e}", exc_info=True)
            fallback_result["reasoning"] = f"Error during fallback generation step: {str(e)}"
            return fallback_result # LLM 调用异常，返回默认值

        # --- 5. 解析和验证 LLM 输出 ---
        try:
            parsed_result = parser_method(fallback_output)

            if isinstance(parsed_result, dict):
                parsed_answer = parsed_result.get("answer")
                if isinstance(parsed_answer, str) and parsed_answer.strip():
                    fallback_result["answer"] = parsed_answer # 使用解析出的有效 answer
                else:
                    logger.warning("Parsed fallback result missing or has invalid 'answer'. Using default.")

                # 尝试获取并验证 reasoning
                parsed_reasoning = parsed_result.get("reasoning")
                if isinstance(parsed_reasoning, str) and parsed_reasoning.strip():
                    fallback_result["reasoning"] = parsed_reasoning # 使用解析出的有效 reasoning
                else:
                    logger.warning("Parsed fallback result missing or has invalid 'reasoning'. Using default.")

                logger.info(f"Successfully generated and parsed fallback answer.")

            else:
                # 解析结果不是期望的字典格式
                logger.warning(f"Fallback parser '{parser_method_name}' did not return a dictionary. Output: '{fallback_output[:100]}...'")
                # 保留默认值，因为无法从中提取有效信息

            return fallback_result

        except Exception as e:
            # 解析过程中发生任何其他异常
            logger.error(f"Error parsing fallback answer output. Raw: '{fallback_output[:100]}...'. Error: {e}", exc_info=True)
            # 更新原因为解析错误，但保留默认回答
            fallback_result["reasoning"] = f"Error while processing the fallback answer: {str(e)}"
            return fallback_result