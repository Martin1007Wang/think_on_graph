
from typing import List, Dict, Any
from src.path_manager import PathManager
from src.utils.data_utils import ExplorationRound

class ResultFormatter:    
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
    
    def format_round(self, round_data: ExplorationRound) -> str:
        assert hasattr(round_data, 'expansions'), "round_data must have 'expansions' attribute"
        result = [f"Round {round_data.round_num}:"]
        for expansion in round_data.expansions:
            entity = expansion.entity
            result.append(f"  Entity: {entity}")
            for relation_info in expansion.relations:
                relation = relation_info.relation
                targets = relation_info.targets
                hops = [f"[{hop}]" if hop.startswith(('m.', 'g.')) else hop for hop in relation.split('>')]
                path_str = ' > '.join(hops)
                if not targets:
                    result.append(f"    {entity}-[{path_str}]->[No Targets]")
                elif all(isinstance(t, str) for t in targets):
                    result.append(f"    {entity}-[{path_str}]->{', '.join(targets)}")
                else:
                    result.append(f"    {entity}-[{path_str}]->[Invalid Targets]")
        result.append("")
        return "\n".join(result)

    
    def format_exploration_history(self, exploration_history: List[ExplorationRound]) -> str:
        return "\n".join(self.format_round(r) for r in exploration_history).rstrip()

    
    def create_question_result(self, question_id: str, question: str, ground_truth: Any, 
                              start_entities: List[str], answer_details: Dict[str, Any],
                              exploration_history: List[ExplorationRound], 
                              answer_found: bool, fallback_used: bool) -> Dict[str, Any]:
        prediction_text = self._format_prediction(answer_details)
        
        return {
            "id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction_text,
            "start_entities": start_entities,
            "reasoning": answer_details.get("reasoning_path", ""),
            "verification": answer_details.get("verification", ""),
            "is_verified": answer_details.get("is_verified", False),
            "exploration_history": [self._round_to_dict(r) for r in exploration_history],
            "answer_found_during_exploration": answer_found,
            "fallback_used": fallback_used,
            "structured_answer": answer_details
        }
    
    def _format_prediction(self, answer_details: Dict[str, Any]) -> str:
        entities = answer_details.get("answer_entities", [])
        return ", ".join(entities) if entities and all(isinstance(e, str) for e in entities) else answer_details.get("answer_sentence", "")

    
    def _round_to_dict(self, round_data: ExplorationRound) -> Dict[str, Any]:
        return {
            "round": round_data.round_num,
            "expansions": [
                {
                    "entity": expansion.entity,
                    "relations": [
                        {"relation": rel.relation, "targets": rel.targets}
                        for rel in expansion.relations
                    ]
                }
                for expansion in round_data.expansions
            ],
            **({"answer_found": round_data.answer_found} if getattr(round_data, "answer_found", None) else {})
        }

    
    def create_error_response(self, question: str, question_id: str, 
                             ground_truth: Any, start_entities: List[str], 
                             error_msg: str, fallback: Dict[str, str]) -> Dict[str, Any]:
        return {
            "id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": fallback.get("answer", "Error during processing"),
            "start_entities": start_entities,
            "reasoning": f"Error occurred: {error_msg}. Fallback: {fallback.get('reasoning', '')}",
            "verification": "Error occurred during processing.",
            "is_verified": False,
            "exploration_history": [],
            "answer_found_during_exploration": False,
            "fallback_used": True,
            "error": error_msg,
            "structured_answer": {
                "can_answer": False,
                "reasoning_path": f"Error occurred: {error_msg}",
                "answer_entities": [],
                "answer_sentence": fallback.get("answer", "Error during processing"),
                "verification": "Error occurred during processing.",
                "is_verified": False
            }
        }
