import logging
import gc
from typing import Any, List, Dict
from src.llm_output_parser import LLMOutputParser
from collections import OrderedDict
import torch
from src.utils.utils import error_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class ModelInterface:
    def __init__(self, model: Any, templates: Any):
        self.model = model
        self.templates = templates
        self.parser = LLMOutputParser()
        
    def generate_output(self, template_name: str, temp_generation_mode: str = "beam", 
                      num_beams: int = 4, **template_args) -> str:
        prompt = self.templates.format_template(template_name, **template_args)
        model_input = self.model.prepare_model_prompt(prompt)
        return self.model.generate_sentence(model_input, temp_generation_mode=temp_generation_mode, num_beams=num_beams)
    
    def select_relations(self, entity: str, available_relations: List[str], question: str, context: str = "", history: str = "",max_relations: int = 10) -> List[str]:
        if len(available_relations) <= max_relations:
            return available_relations
        relation_dict = {f"REL_{i}": rel for i, rel in enumerate(available_relations)}
        relation_options = "\n".join(f"[{rel_id}] {rel}" for rel_id, rel in relation_dict.items())
        template_name = "relation_selection_context" if context and history else "relation_selection"
        template_args = {
            "question": question,
            "entity": entity,
            "relations": relation_options,
            "relation_ids": ", ".join(relation_dict.keys()),
            "max_k_relations": min(max_relations, len(available_relations))
        }
        if context and history:
            template_args.update({"history": history, "context": context})
        selection_output = self.generate_output(template_name, **template_args)
        if not selection_output:
            return available_relations[:max_relations]
        selected_relations_or_ids = self.parser.parse_relations(selection_output, available_relations)
        selected_names = []
        for item in selected_relations_or_ids:
            if item.startswith("REL_") and item in relation_dict:
                selected_names.append(relation_dict[item])
            elif item in available_relations:
                selected_names.append(item)
        return list(OrderedDict.fromkeys(selected_names)) if selected_names else available_relations[:max_relations]
    
    def select_entities(self, question: str, candidates: List[str], max_entities: int) -> List[str]:
        if len(candidates) <= max_entities or max_entities is None:
            return candidates
        entity_dict = {f"ENT_{i}": entity for i, entity in enumerate(candidates)}
        entities_str = "\n".join([f"[{ent_id}] {entity}" for ent_id, entity in entity_dict.items()])
        selection_output = self.generate_output(
            "entity_selection",
            question=question,
            entities=entities_str,
            max_k_entities=max_entities,
            entity_ids=", ".join(entity_dict.keys())
        )
        if not selection_output:
            return candidates[:max_entities]
        selected_entity_ids = self.parser.parse_relations(selection_output, list(entity_dict.keys()))
        selected_entities = []
        processed_ids = set()
        for ent_id in selected_entity_ids:
            if ent_id in entity_dict and ent_id not in processed_ids:
                selected_entities.append(entity_dict[ent_id])
                processed_ids.add(ent_id)
            elif ent_id in entity_dict.values() and ent_id not in selected_entities:
                selected_entities.append(ent_id)
        return selected_entities or candidates[:max_entities]
    
    def check_answerability(self, question: str, start_entities: List[str], 
                          exploration_history: str) -> Dict[str, Any]:
        template_name = "reasoning"
        reasoning_output = self.generate_output(
            template_name,
            question=question,
            entity=", ".join(start_entities),
            exploration_history=exploration_history
        )
        if not reasoning_output:
            return {
                "can_answer": False, 
                "reasoning_path": "Failed to generate reasoning."
            }
        return self.parser.parse_reasoning_output(reasoning_output)
    
    def generate_fallback_answer(self, question: str) -> Dict[str, str]:
        fallback_output = self.generate_output("fallback_answer", question=question)
        if not fallback_output:
            return {
                "answer": "I am sorry, but I couldn't find a definitive answer to your question.",
                "reasoning": "The knowledge graph lacks the required information to answer this question."
            }
        result = self.parser.parse_final_answer(fallback_output)
        fallback_answer = {
            "answer": result.get("answer", "I cannot answer this question based on the available knowledge."),
            "reasoning": result.get("reasoning", "The knowledge graph exploration did not yield sufficient information.")
        }
        if fallback_answer["answer"] == "Could not parse answer.":
            fallback_answer["answer"] = "I cannot answer this question based on the available knowledge."
        if fallback_answer["reasoning"] == "Could not parse reasoning.":
            fallback_answer["reasoning"] = "The knowledge graph exploration did not yield sufficient information."
        return fallback_answer
