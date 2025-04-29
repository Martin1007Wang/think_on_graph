import concurrent.futures
import datetime
import logging
import os
import re
import gc
from collections import OrderedDict
from typing import Any, Dict, List, Match, Optional, Pattern, Set, Tuple

from src.entity_expander import EntityExpander
from src.knowledge_graph import KnowledgeGraph
from src.model_interface import ModelInterface
from src.path_manager import PathManager
from src.result_formatter import ResultFormatter
from src.template import KnowledgeGraphTemplates
from src.utils.data_utils import ExplorationRound

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

MCODE_PATTERN: Pattern[str] = re.compile(r'm\.[0-9a-z_]+')
GCODE_PATTERN: Pattern[str] = re.compile(r'g\.[0-9a-z_]+')

LOG_DIR = "logs/unanswerable_questions"

class KnowledgeExplorer:
    def __init__(self, 
                 kg: KnowledgeGraph, 
                 model: Any, 
                 max_rounds: int = 3, 
                 max_k_relations: int = 10, 
                 max_workers: int = 5, 
                 max_frontier_size: Optional[int] = 20) -> None:
        self.kg = kg
        self.max_rounds = max_rounds
        self.max_frontier_size = max_frontier_size
        
        self.templates = KnowledgeGraphTemplates()
        self.path_manager = PathManager()
        self.model_interface = ModelInterface(model, self.templates)
        self.entity_expander = EntityExpander(kg, self.model_interface, self.path_manager, max_k_relations)
        self.formatter = ResultFormatter(self.path_manager)
        
        logger.info(
            f"KnowledgeExplorer initialized with max_rounds={max_rounds}, "
            f"max_k_relations={max_k_relations}, max_workers={max_workers}, "
            f"max_frontier_size={max_frontier_size}"
        )
    
    def explore_knowledge_graph(self, question: str, start_entities: List[str], ground_truth: List[str]) -> Tuple[List[ExplorationRound], bool]:
        exploration_history: List[ExplorationRound] = []
        entities_explored: Set[str] = set(start_entities)
        entities_in_context: Set[str] = set(start_entities)
        frontier: List[str] = list(start_entities)
        answer_found: bool = False
        for round_num in range(1, self.max_rounds + 1):
            entities_context = ", ".join(sorted(list(entities_in_context)))
            history_str = self.formatter.format_exploration_history(exploration_history)
            round_exploration = self._process_exploration_round(round_num, frontier, question, entities_context, history_str, entities_explored)
            exploration_history.append(round_exploration)
            history_formatted = self._filter_history(exploration_history, question)
            answer_result = self._check_for_answer(question, start_entities, history_formatted)
            if answer_result.get("can_answer", False):
                exploration_history[-1].answer_found = answer_result
                answer_found = True
                break
        return exploration_history, answer_found
    
    def _filter_history(self, exploration_history: List[ExplorationRound], question: str, max_paths: int = 10) -> str:
        paths = self._split_paths(exploration_history)
        if len(paths) <= max_paths:
            return self.formatter.format_exploration_history(exploration_history)
        selected_paths = self.model_interface.select_paths(question, paths, max_paths)
        return selected_paths
            
    def _split_paths(self, exploration_history: List[ExplorationRound]) -> Tuple[List[str], Dict[str, str]]:
        formatted_paths = []
        for round_data in exploration_history:
            for expansion in round_data.expansions:
                entity = expansion.entity
                for relation_data in expansion.relations:
                    relation = relation_data.relation
                    for target in relation_data.targets:
                        path = f"{entity}-[{relation}]->{target}"
                        formatted_paths.append(path)
        return formatted_paths
    
    def _process_exploration_round(self, round_num: int, frontier: List[str], question: str, 
                                  entities_context: str, history_str: str, 
                                  entities_explored: Set[str]) -> Tuple[ExplorationRound, Set[str]]:
        logger.info(f"Starting exploration round {round_num} with frontier size {len(frontier)}")
        round_exploration = ExplorationRound(round_num=round_num)
        for entity in frontier:
            expansion = self.entity_expander.expand_entity(entity, question, entities_context, history_str)
            if expansion:
                round_exploration.expansions.append(expansion)
        return round_exploration
    
    def _check_for_answer(self, question: str, start_entities: List[str], exploration_history: List[ExplorationRound]) -> Dict[str, Any]:
        return self.model_interface.check_answerability(question, start_entities, exploration_history)

    def get_fallback_answer(self, question: str) -> Dict[str, str]:
        logger.info(f"Generating fallback answer for: {question}")
        fallback_answer = self.model_interface.generate_fallback_answer(question)
        self._save_unanswerable_question(question)
        return fallback_answer

    def _save_unanswerable_question(self, question: str) -> None:
        """Save unanswerable questions to a log file"""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(LOG_DIR, f"unanswerable_{date_str}.log")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {question}\n")
        except Exception as e:
            logger.error(f"Error saving unanswerable question: {e}", exc_info=True)

    def _create_fallback_answer_structure(self, fallback: Dict[str, str]) -> Dict[str, Any]:
        """Create a structured fallback answer"""
        return {
            "can_answer": False,
            "reasoning_path": fallback.get("reasoning", ""),
            "answer_entities": [],
            "analysis": fallback.get("answer", "")
        }

    def process_question(self, data: Dict[str, Any], processed_ids: Set[str]) -> Optional[Dict[str, Any]]:
        question = data.get("question")
        ground_truth = data.get("answer")
        question_id = data.get("id", f"unknown_id_{hash(question)}")
        start_entities = self._normalize_start_entities(data)
        exploration_history, answer_found = self.explore_knowledge_graph(question, start_entities, ground_truth)
        if answer_found and exploration_history:
            final_answer = exploration_history[-1].answer_found
            fallback_used = False
            logger.info(f"[{question_id}] Answer found during exploration")
        else:
            fallback = self.get_fallback_answer(question)
            final_answer = self._create_fallback_answer_structure(fallback)
            fallback_used = True
            logger.info(f"[{question_id}] Using fallback due to no exploration")
        result = self.formatter.create_question_result(question_id, question, ground_truth, start_entities, final_answer, exploration_history, answer_found, fallback_used)
        return result

    def _normalize_start_entities(self, data: Dict[str, Any]) -> List[str]:
        """Normalize start entities from data"""
        start_entities_raw = data.get("q_entity", [])
        if not isinstance(start_entities_raw, list):
            start_entities_raw = [start_entities_raw]      
        return [str(e) for e in start_entities_raw if e and isinstance(e, (str, int, float))]