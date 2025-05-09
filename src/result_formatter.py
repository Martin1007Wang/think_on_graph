import logging
import io # 保留导入以备将来可能的优化
from typing import List, Dict, Any, Optional

# --- 假设的数据类定义 (用于类型提示和示例) ---
# from src.path_manager import PathManager # 假设 PathManager 被正确导入

class PathManager: # Placeholder
     def is_coded_entity(self, entity: str) -> bool:
         # 示例逻辑，根据需要替换
         return isinstance(entity, str) and entity.startswith(('m.', 'g.'))

class EntityRelation:
    # 使用 __slots__ 可以略微减少内存占用和加快属性访问速度 (可选优化)
    __slots__ = ('relation', 'targets')
    def __init__(self, relation: str, targets: List[Any]):
        self.relation = relation
        self.targets = targets

class EntityExpansion:
    __slots__ = ('entity', 'relations')
    def __init__(self, entity: str):
        self.entity = entity
        self.relations: List[EntityRelation] = []

class ExplorationRound:
    __slots__ = ('round_num', 'expansions', 'exceeded_history_limit', 'answer_found')
    def __init__(self, round_num: int):
        self.round_num = round_num
        self.expansions: List[EntityExpansion] = []
        self.exceeded_history_limit: bool = False
        # 假设 answer_found 要么是 None，要么是一个可序列化的字典
        self.answer_found: Optional[Dict[str, Any]] = None

    # **推荐**: 如果控制 ExplorationRound 类，实现一个高效的 to_dict 方法
    # def to_dict(self) -> Dict[str, Any]:
    #     return {
    #         "round": self.round_num,
    #         "expansions": [
    #             {
    #                 "entity": exp.entity,
    #                 "relations": [{"relation": rel.relation, "targets": rel.targets} for rel in exp.relations]
    #             } for exp in self.expansions
    #         ],
    #         # 只在存在时添加可选字段
    #         **({"exceeded_history_limit": self.exceeded_history_limit} if hasattr(self, 'exceeded_history_limit') else {}),
    #         **({"answer_found": self.answer_found} if self.answer_found is not None else {})
    #     }

# --- 日志配置 ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

# --- 优化后的 ResultFormatter ---
class ResultFormatter:
    """
    Formats exploration results into various representations.
    Performance considerations:
    - Assumes efficiency relies heavily on the size of exploration data.
    - String operations are generally efficient but can add up in deep loops.
    - Serialization (`_round_to_dict`) called in a loop is a key area.
    """
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager

    def format_round(self, round_data: ExplorationRound) -> str:
        """Formats a single exploration round into a human-readable string."""
        result_lines = [f"Round {round_data.round_num}:"]
        expansions = getattr(round_data, 'expansions', [])
        if not expansions:
             result_lines.append("  No expansions in this round.")

        for expansion in expansions:
            entity = getattr(expansion, 'entity', '[Missing Entity]')
            result_lines.append(f"  Entity: {entity}")
            relations = getattr(expansion, 'relations', [])
            if not relations:
                result_lines.append("    No relations found for this entity.")
                continue

            for relation_info in relations:
                relation = getattr(relation_info, 'relation', '[Missing Relation]')
                targets = getattr(relation_info, 'targets', [])

                try:
                    # Use self.path_manager if it exists and has the method
                    is_coded_func = getattr(self.path_manager, 'is_coded_entity', lambda x: False)
                    hops = [f"[{hop.strip()}]" if is_coded_func(hop.strip()) else hop.strip()
                            for hop in relation.split('>')]
                    path_str = ' > '.join(hops)
                except Exception as e:
                    logger.error(f"Error processing relation path '{relation}': {e}")
                    path_str = f"[Error Processing Path: {relation}]"

                target_str: str
                if not targets:
                    target_str = "[No Targets]"
                else:
                    try:
                        str_targets = []
                        all_valid = True
                        for t in targets:
                             if isinstance(t, str): str_targets.append(t)
                             elif t is None: str_targets.append("None")
                             else:
                                 try: str_targets.append(str(t))
                                 except Exception: str_targets.append("[Conversion Error]"); all_valid = False
                        target_str = ", ".join(str_targets)
                        if not all_valid: logger.warning(f"Round {round_data.round_num}, Path {path_str}: Target conversion issues.")
                    except Exception as e:
                         target_str = "[Error Formatting Targets]"
                         logger.error(f"Round {round_data.round_num}, Path {path_str}: Could not format targets. Error: {e}", exc_info=False)

                result_lines.append(f"    {entity}-[{path_str}]->{target_str}")

        if getattr(round_data, "exceeded_history_limit", False):
            result_lines.append("    [HISTORY SIZE LIMIT EXCEEDED - Exploration stopped]")

        result_lines.append("")
        return "\n".join(result_lines)

    def format_round_results(self, round_data: ExplorationRound) -> str:
        """Alias for format_round."""
        return self.format_round(round_data)

    def format_exploration_history(self, exploration_history: List[ExplorationRound]) -> str:
        """Formats the entire exploration history."""
        if not exploration_history: return ""
        return "\n".join(self.format_round(r) for r in exploration_history).rstrip()

    def create_question_result(self, question_id: str, question: str, ground_truth: Any,
                               start_entities: List[str], answer: Dict[str, Any],
                               exploration_history: List[ExplorationRound],
                               answer_found: bool, fallback_used: bool) -> Dict[str, Any]:
        """(REVISED) Creates the final structured result dictionary."""
        prediction_text = "Error: Could not determine prediction text."
        reasoning_text = ""
        analysis_text = ""

        try:
            if fallback_used:
                prediction_text = answer.get("answer", "Fallback answer missing.")
                reasoning_text = answer.get("reasoning", "Fallback reasoning missing.")
                analysis_text = answer.get("analysis", "")
                logging.debug(f"[{question_id}] Formatting fallback result.")
            elif answer_found and isinstance(answer, dict):
                try: prediction_text = self._format_prediction(answer)
                except Exception as e:
                     logging.error(f"[{question_id}] Error calling _format_prediction: {e}. Using fallback format.")
                     entities = answer.get("answer_entities")
                     prediction_text = ", ".join(map(str, entities)) if isinstance(entities, list) and entities else "Formatting Error"
                reasoning_text = answer.get("reasoning_path", "No reasoning path provided.")
                analysis_text = answer.get("analysis", "No analysis provided.")
                logging.debug(f"[{question_id}] Formatting exploration success result.")
            else:
                 logging.error(f"[{question_id}] Inconsistent state in create_question_result. answer_found={answer_found}, fallback_used={fallback_used}, type={type(answer)}")
                 prediction_text = "Internal Error: Inconsistent state."

            serializable_history = []
            if exploration_history:
                 try: serializable_history = [self._round_to_dict(r) for r in exploration_history]
                 except Exception as e:
                     logging.error(f"[{question_id}] Failed to serialize exploration history: {e}", exc_info=True)
                     serializable_history = [{"error": "Failed to serialize round", "round_index": i} for i, r in enumerate(exploration_history)]

            return {
                "id": question_id, "question": question, "ground_truth": ground_truth,
                "prediction": prediction_text, "start_entities": start_entities,
                "reasoning": reasoning_text, "analysis": analysis_text,
                "exploration_history": serializable_history,
                "answer_found_during_exploration": answer_found, "fallback_used": fallback_used,
            }
        except Exception as e:
             logging.error(f"[{question_id}] Critical error within create_question_result: {e}", exc_info=True)
             return { "id": question_id, "question": question, "error": f"Failed to format final result: {e}" }

    def _format_prediction(self, answer: Dict[str, Any]) -> str:
        """(Optimized) Formats prediction string."""
        sentence = answer.get("answer_sentence")
        if isinstance(sentence, str) and sentence.strip(): return sentence
        entities = answer.get("answer_entities")
        if isinstance(entities, list) and entities:
             try: return ", ".join(map(str, entities))
             except Exception as e: logging.error(f"Error converting entities to string: {e}"); return "[Error formatting entities]"
        return answer.get("answer", "Prediction could not be formatted.")

    def _round_to_dict(self, round_data: ExplorationRound) -> Dict[str, Any]:
        """(Optimized) Serializes ExplorationRound, preferring its own to_dict."""
        if hasattr(round_data, 'to_dict') and callable(round_data.to_dict):
             try: return round_data.to_dict()
             except Exception as e: logging.error(f"Error calling to_dict on round {getattr(round_data, 'round_num', '?')}: {e}", exc_info=True)
        try:
             round_num = getattr(round_data, 'round_num', -1)
             expansions_data = []
             for expansion in getattr(round_data, 'expansions', []):
                 relations_data = [{"relation": getattr(rel, 'relation', None), "targets": getattr(rel, 'targets', [])} for rel in getattr(expansion, 'relations', [])]
                 expansions_data.append({"entity": getattr(expansion, 'entity', None), "relations": relations_data})
             result_dict = {"round": round_num, "expansions": expansions_data}
             if hasattr(round_data, 'exceeded_history_limit'): result_dict["exceeded_history_limit"] = round_data.exceeded_history_limit
             answer_found_data = getattr(round_data, 'answer_found', None)
             if answer_found_data is not None: result_dict["answer_found"] = answer_found_data
             return result_dict
        except Exception as e:
             logging.error(f"Error manually serializing round {getattr(round_data, 'round_num', '?')}: {e}", exc_info=True)
             return {"error": "Failed to manually serialize round", "round_num": getattr(round_data, 'round_num', -1)}

    def create_error_response(self, question: str, question_id: str, ground_truth: Any, start_entities: List[str], error_msg: str, fallback: Dict[str, str]) -> Dict[str, Any]:
        """Creates a standardized dictionary for critical error responses."""
        return {
            "id": question_id, "question": question, "ground_truth": ground_truth,
            "prediction": fallback.get("answer", "Error during processing"),
            "start_entities": start_entities,
            "reasoning": f"Error occurred: {error_msg}. Fallback Reason: {fallback.get('reasoning', 'N/A')}",
            "analysis": "", "exploration_history": [], "answer_found_during_exploration": False,
            "fallback_used": True, "error": error_msg,
        }