from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ExplorationRound:
    round_num: int
    question_posed_to_llm: str
    expanded_entity: str
    candidate_relations: List[str]
    chosen_relations: List[Dict[str, Any]]
    model_reasoning: Optional[str] = None

@dataclass
class TraversalState:
    original_question: str
    exploration_trace: List[ExplorationRound] = field(default_factory=list)
    current_path: List[str] = field(default_factory=list)
    answer_found: bool = False
    final_answer_entities: List[str] = field(default_factory=list)
    final_reasoning_summary: str = ""