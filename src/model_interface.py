import logging
from typing import Any, List, Dict, Optional, Set, Callable, Tuple, Union
from enum import Enum

from src.llm_output_parser import LLMOutputParser
from src.utils.data_utils import ExplorationRound

logger = logging.getLogger(__name__)

class TemplateName(str, Enum):
    RELATION_SELECTION = "relation_selection"
    RELATION_SELECTION_WITH_HISTORY = "relation_selection_with_history"
    FACT_SELECTION = "expanded_cvt_hop_selection"
    ENTITY_SELECTION = "entity_selection"
    ATTRIBUTE_SELECTION = "cvt_attribute_selection"
    REASONING = "reasoning"
    FALLBACK_ANSWER = "fallback_answer"

class ParserMethod(str, Enum):
    PARSE_RELATIONS = "parse_relations"
    PARSE_FACTS = "parse_facts"
    PARSE_ENTITIES = "parse_entities"
    PARSE_ATTRIBUTE_SELECTION = "parse_attribute_selection"
    PARSE_REASONING_OUTPUT = "parse_reasoning_output"
    PARSE_FALLBACK_ANSWER = "parse_fallback_answer"


class ModelInterface:
    SELECTION_CONFIGS = {
        "relation": {
            "template_name_provider": lambda history: TemplateName.RELATION_SELECTION_WITH_HISTORY if history else TemplateName.RELATION_SELECTION,
            "parser_method_name": ParserMethod.PARSE_RELATIONS
        },
        "fact": {
            "template_name_provider": lambda history: TemplateName.FACT_SELECTION,
            "parser_method_name": ParserMethod.PARSE_FACTS
        },
        "entity": {
            "template_name_provider": lambda history: TemplateName.ENTITY_SELECTION,
            "parser_method_name": ParserMethod.PARSE_ENTITIES
        },
        "attribute": {
            "template_name_provider": lambda history: TemplateName.ATTRIBUTE_SELECTION,
            "parser_method_name": ParserMethod.PARSE_ATTRIBUTE_SELECTION
        }
    }

    def __init__(self, model: Any, templates: Any, parser: Optional[LLMOutputParser] = None):
        if not hasattr(model, 'prepare_model_prompt') or not hasattr(model, 'generate_sentence'):
            raise TypeError(f"Model object {model.__class__.__name__} must have 'prepare_model_prompt' and 'generate_sentence' methods.")
        if not hasattr(templates, 'format_template'):
            raise TypeError(f"Templates object {templates.__class__.__name__} must have a 'format_template' method.")

        self.model = model
        self.templates = templates
        self.parser = parser or LLMOutputParser()

        if not isinstance(self.parser, LLMOutputParser):
            raise TypeError(f"Parser must be an instance of LLMOutputParser, got {type(self.parser)}")
        
        logger.info(f"ModelInterface refactored initialized with model type: {model.__class__.__name__}")

    def generate_output(self, template_name: str, **template_args) -> Optional[str]:
        try:
            prompt = self.templates.format_template(template_name, **template_args)
            if not prompt:
                logger.error(f"Failed to format prompt for template '{template_name}'")
                return None
            
            model_input = self.model.prepare_model_prompt(prompt)
            batch_output = self.model.generate_sentence(model_input, **template_args)
            if not batch_output or not batch_output[0]:
                logger.warning(f"Model generation returned None or empty list for template '{template_name}'.")
                return None
            return batch_output
        except Exception as e:
            logger.error(f"Error during model generation for template '{template_name}'. Error: {e}", exc_info=True)
            return None
    
    def generate_output_batch(self, prompts: List[str], **generation_args) -> Optional[List[List[str]]]:
        try:
            if not prompts:
                return []
            outputs = self.model.generate_sentence(prompts, **generation_args)
            return outputs
        except Exception as e:
            logger.error(f"Error during model batch generation. Error: {e}", exc_info=True)
            return None

    def _select_items(self, item_type: str, items: List[str], max_items: Optional[int],
                      template_name: str, parser_method_name: str, fallback_on_error: bool = False,
                      **template_args: Any) -> List[str]:
        if not items: return []
        if max_items is not None and max_items <= 0: return []
        if max_items is None or len(items) <= max_items: return items

        item_options = "\n".join(items)
        selection_template_args = template_args.copy()
        selection_template_args.update({
            item_type + "s": item_options,
            "max_selection_count": max_items
        })
        
        parsed_selection = None
        error_occurred = False
        
        try:
            selected_output = self.generate_output(template_name, **selection_template_args)
            if selected_output is None:
                raise ValueError(f"LLM generation failed for {item_type} selection.")

            parser_method = getattr(self.parser, parser_method_name)
            parsed_selection = parser_method(selected_output, candidate_items=items)

            if parsed_selection is None:
                raise ValueError(f"Parsing failed for {item_type} selection.")
            if not isinstance(parsed_selection, list):
                logger.warning(f"Parser for {item_type} did not return a list. Treating as no selection.")
                parsed_selection = []
        
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            logger.error(f"Error during {item_type} selection: {e}", exc_info=True)

        if error_occurred:
            return items[:max_items] if fallback_on_error else []
        final_selections, seen_items = [], set()
        candidate_items_set = set(items)
        if parsed_selection:
            for selection in parsed_selection:
                if selection in candidate_items_set and selection not in seen_items:
                    final_selections.append(selection)
                    seen_items.add(selection)
                    if len(final_selections) >= max_items: break
        return final_selections

    def select(self, item_type: str, items: List[str], max_items: Optional[int], **kwargs) -> List[str]:
        config = self.SELECTION_CONFIGS.get(item_type)
        if not config:
            raise ValueError(f"Invalid item_type '{item_type}' provided for selection.")

        history = kwargs.get("history", "")
        template_name = config["template_name_provider"](history)
        fallback = kwargs.pop("fallback_on_error", False)

        return self._select_items(
            item_type=item_type,
            items=items,
            max_items=max_items,
            template_name=template_name,
            parser_method_name=config["parser_method_name"],
            fallback_on_error=fallback,
            **kwargs
        )

    def check_answerability(self, question: str, start_entities: List[str],
                              exploration_history: str) -> Dict[str, Any]:
        template_name = TemplateName.REASONING
        parser_method_name = ParserMethod.PARSE_REASONING_OUTPUT

        if not hasattr(self.parser, parser_method_name):
            error_msg = f"Internal Error: Parser method '{parser_method_name}' missing."
            return {"answer_found": False, "next_exploration_step": {"justification": error_msg}}

        reasoning_output_str = self.generate_output(
            template_name,
            question=question,
            entity=", ".join(start_entities),
            exploration_history=exploration_history
        )
        if not reasoning_output_str:
             return {"answer_found": False, "next_exploration_step": {"justification": "Model failed to generate output."}}
        
        try:
            parser_method: Callable = getattr(self.parser, parser_method_name)
            parsed_output = parser_method(reasoning_output_str)
            return parsed_output
        except Exception as e:
             error_msg = f"Critical error parsing reasoning output: {e}"
             return {"answer_found": False, "next_exploration_step": {"justification": error_msg}}

    def generate_fallback_answer(self, question: str, history_text: str) -> Dict[str, any]:
        logger.warning(f"Generating fallback answer for question: '{question[:100]}...'")
        template_name = "fallback_answer"
        parser_method_name = "parse_fallback_answer"

        default_result = {
            "can_answer": True,
            "reasoning_path": "[No conclusive path found]",
            "answer_entities": [],
            "reasoning_summary": "The knowledge exploration process did not yield a conclusive answer."
        }

        parser_method: Optional[Callable] = getattr(self.parser, parser_method_name, None)
        if not callable(parser_method):
            logger.error(f"Parser method '{parser_method_name}' not found or not callable.")
            default_result["reasoning_summary"] = f"Internal Configuration Error: Required parser method '{parser_method_name}' is unavailable."
            return default_result

        try:
            raw_output = self.generate_output(
                template_name,
                question=question,
                exploration_history=history_text
            )

            if not (raw_output and raw_output.strip()):
                logger.error("LLM failed to generate fallback answer (returned None or empty).")
                default_result["reasoning_summary"] = "The language model failed to produce a response during the fallback step."
                return default_result

            try:
                parsed_result = parser_method(raw_output)
                if isinstance(parsed_result, dict) and "can_answer" in parsed_result and "reasoning_summary" in parsed_result:
                    
                    logger.info("Successfully generated and parsed fallback answer. Using the parsed result directly.")
                    final_result = parsed_result
                    final_result["can_answer"] = True
                    return final_result
                else:
                    logger.warning(f"Fallback parser returned an invalid or malformed dictionary. Raw output: {raw_output[:200]}...")
                    default_result["reasoning_summary"] = f"Parsing failed: The model's output was not in the expected format."
                    return default_result

            except Exception as e:
                logger.error(f"An exception occurred while parsing the fallback answer. Raw: '{raw_output[:100]}...'. Error: {e}", exc_info=True)
                default_result["reasoning_summary"] = f"Parsing failed due to an exception: {str(e)}"
                return default_result

        except Exception as e:
            logger.error(f"An exception occurred during the LLM call for fallback: {e}", exc_info=True)
            default_result["reasoning_summary"] = f"A critical error occurred during the LLM call: {str(e)}"
            return default_result