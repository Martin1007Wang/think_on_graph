import logging
from typing import List, Dict, Any, Optional, Set, Union
from src.utils.data_utils import ExplorationRound, TraversalState, Path
from dataclasses import asdict

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class ResultFormatter:
    @staticmethod
    def is_coded_entity(entity: str) -> bool:
        return isinstance(entity, str) and (entity.startswith("m.") or entity.startswith("g."))

    @staticmethod
    def extract_paths_from_history(start_entities: Set[str],history: List[ExplorationRound]) -> List[Path]:
        if not history:
            return []
        graph_view: Dict[str, List[Dict[str, str]]] = {}
        for round_data in history:
            source_entity = round_data.entity
            
            if source_entity not in graph_view:
                graph_view[source_entity] = []
            
            # 'chosen_relations' is a list of dicts like {'name': ..., 'targets': [...]}
            for relation_info in round_data.relations:
                relation_name = relation_info.get("name")
                targets = relation_info.get("targets", [])
                for target_entity in targets:
                    graph_view[source_entity].append({
                        "relation": relation_name,
                        "target": target_entity
                    })
        # --- END OF CORRECTION ---

        all_paths: List[Path] = []
        
        def _dfs_recursive(
            current_node: str,
            current_path_elements: List[Dict[str, str]],
            visited_in_path: Set[str]
        ):
            # The DFS logic itself was correct and does not need to be changed.
            visited_in_path.add(current_node)

            # Base case: if we hit a leaf node in our explored graph
            if current_node not in graph_view or not graph_view[current_node]:
                if current_path_elements:
                    all_paths.append(Path(elements=list(current_path_elements)))
                return

            is_leaf = True
            for edge in graph_view[current_node]:
                is_leaf = False
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
                
            # Handle cases where a node was expanded but all its children led to cycles
            if is_leaf and current_path_elements:
                all_paths.append(Path(elements=list(current_path_elements)))

        for start_node in start_entities:
            if start_node in graph_view:
                _dfs_recursive(start_node, [], set())

        return all_paths
    
    def format_exploration_trace_to_string(self, exploration_trace: List[ExplorationRound]) -> str:
        if not exploration_trace:
            return "No exploration trace was recorded."
        
        all_lines = ["Exploration Trace Summary:"]
        for round_item in sorted(exploration_trace, key=lambda r: r.round_num):
            all_lines.append(f"\n--- Round {round_item.round_num} ---")
            all_lines.append(f"Expanded Entity: {round_item.expanded_entity}")
            
            if round_item.chosen_relations:
                all_lines.append("  Chosen Paths:")
                for rel_info in round_item.chosen_relations:
                    target_str = ', '.join(rel_info.get('targets', ['N/A']))
                    all_lines.append(f"  - [{rel_info.get('name')}] --> {target_str}")
            else:
                all_lines.append("  (No new relations were chosen or expanded for this entity)")
                
        return "\n".join(all_lines)

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

    def create_question_result(
        self,
        question_id: str, 
        traversal_state: TraversalState,
        ground_truth: Any,
        start_entities: List[str],
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
                "reasoning_path": traversal_state.reasoning_path,
                "final_reasoning_summary": traversal_state.final_reasoning_summary,
                "answer_found": traversal_state.answer_found,
                "fallback_used": fallback_used,
                "exploration_trace": serializable_trace,
                "runtime_s": runtime_s,
                "llm_calls": llm_calls,
                "llm_tokens": llm_tokens,
            }
            return result
        except Exception as e:
            logger.critical(f"[{question_id}] Critical error within create_question_result: {e}", exc_info=True)
            return {"id": question_id, "question": traversal_state.original_question, "error": f"Failed to format final result: {e}"}

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