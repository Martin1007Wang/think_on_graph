import logging
from typing import List, Dict, Any, Optional, Set, Union
from src.utils.data_utils import ExplorationRound, EntityExpansion, Path
from dataclasses import asdict

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class ResultFormatter:
    @staticmethod
    @staticmethod
    def is_coded_entity(entity: str) -> bool:
        return isinstance(entity, str) and (entity.startswith("m.") or entity.startswith("g."))

    @staticmethod
    def extract_paths_from_history(
        start_entities: Set[str],
        history: List[ExplorationRound]
    ) -> List[Path]:
        if not history:
            return []

        graph_view: Dict[str, List[Dict[str, str]]] = {}
        for round_data in history:
            for expansion in round_data.expansions:
                if expansion.entity not in graph_view:
                    graph_view[expansion.entity] = []
                for relation in expansion.relations:
                    for target in relation.targets:
                        graph_view[expansion.entity].append({
                            "relation": relation.name,
                            "target": target
                        })

        all_paths: List[Path] = []
        
        def _dfs_recursive(
            current_node: str,
            current_path_elements: List[Dict[str, str]],
            visited_in_path: Set[str]
        ):

            visited_in_path.add(current_node)

            if current_node not in graph_view or not graph_view[current_node]:
                if current_path_elements:
                    all_paths.append(Path(elements=list(current_path_elements)))
                return

            for edge in graph_view[current_node]:
                new_target = edge["target"]
                
                path_element = {
                    "source": current_node,
                    "relation": edge["relation"],
                    "target": new_target
                }
                new_path = current_path_elements + [path_element]
                if new_target in visited_in_path:
                    logger.debug(f"Cycle detected at '{new_target}', terminating this path.")
                    all_paths.append(Path(elements=new_path))
                    continue
                _dfs_recursive(new_target, new_path, visited_in_path.copy())
        for start_node in start_entities:
            if start_node in graph_view:
                _dfs_recursive(start_node, [], set())

        return all_paths
            
    def format_history(self, history: Union[ExplorationRound, List[ExplorationRound]]) -> str:
        if not isinstance(history, list):
            history = [history]

        # --- 从这里开始，函数其余部分的逻辑与之前完全相同 ---

        # 如果历史记录为空（或传入的是一个空列表），则返回提示信息
        if not history:
            return "No exploration history was recorded."

        # 用于存储所有轮次格式化后的文本行
        all_lines = []

        # 遍历历史记录中的每一轮
        for round_item in history:
            # round_item 可能是 None 或者其他无效类型，增加一个健壮性检查
            if not isinstance(round_item, ExplorationRound) or not round_item.expansions:
                continue

            # 添加当前轮次的标题
            all_lines.append(f"\n--- Round {round_item.round_num} ---")
            
            for expansion in round_item.expansions:
                for relation in expansion.relations:
                    # 将格式化后的行直接添加到主列表
                    target_str = ', '.join(relation.targets)
                    all_lines.append(f"{expansion.entity}--[{relation.name}]-->{target_str}")
        
        # 将所有行用换行符合并成一个最终的字符串
        # 如果 all_lines 为空（例如，所有轮次都没有扩展），则返回提示信息
        return "\n".join(all_lines) if all_lines else "No valid expansions found in the history."

    def format_paths_to_history_string(self, paths: List[str]) -> str:
        """将一个剪枝后的路径列表转换回一个单一的、可用于下一轮探索的历史字符串。"""
        if not paths:
            return ""
        # 确保末尾有换行符，以便下一轮追加
        return "\n".join(paths) + "\n"

    def format_exploration_trace(self, exploration_trace: List[ExplorationRound]) -> str:
        if not exploration_trace:
            return "No exploration trace was recorded."
        
        all_lines = []
        for round_item in exploration_trace:
            all_lines.append(f"\n--- Round {round_item.round_num} ---")
            all_lines.append(f"Current Entity: {round_item.current_entity}")
            all_lines.append(f"Chosen Relation: {round_item.chosen_relation}")
            next_entities = round_item.next_entity
            if isinstance(next_entities, list):
                next_entities = ", ".join(next_entities)
            all_lines.append(f"Next Entity: {next_entities}")
        return "\n".join(all_lines)

    '''    def create_question_result(
        self,
        question_id: str, 
        traversal_state: TraversalState,
        ground_truth: Any,
        start_entities: List[str],
        final_answer_data: Dict[str, Any],
        fallback_used: bool,
        runtime_s: float,
        llm_calls: int,
        llm_tokens: int
    ) -> Dict[str, Any]:
        try:
            serializable_trace = [asdict(r) for r in traversal_state.exploration_trace]
            result = {
                "id": question_id,
                "original_question": traversal_state.original_question,
                "ground_truth": ground_truth,
                "final_answer_entities": traversal_state.final_answer_entities,
                "final_reasoning_summary": traversal_state.final_reasoning_summary,
                "exploration_trace": serializable_trace,
                "answer_found": traversal_state.answer_found,
                "fallback_used": fallback_used,
                "start_entities": start_entities,
                "runtime_s": runtime_s,
                "llm_calls": llm_calls,
                "llm_tokens": llm_tokens,
            }
            return result
        except Exception as e:
            logger.critical(f"[{question_id}] Critical error within create_question_result: {e}", exc_info=True)
            return {"id": question_id, "question": traversal_state.original_question, "error": f"Failed to format final result: {e}"}
'''

    def create_error_response(
        self,
        question_id: str,
        question: str,
        error_msg: str,
        **kwargs
    ) -> Dict[str, Any]:
        """为严重错误创建标准的响应字典。"""
        return {
            "id": question_id,
            "question": question,
            "error": error_msg,
            "prediction": {"answer": "An error occurred during processing.", "reasoning": f"Error: {error_msg}"},
            "status": {
                "answer_found_during_exploration": False,
                "fallback_used": True,
            },
        }