'''from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class ExplorationRound:
    round_num: int
    current_entity: str
    question_posed: str
    candidate_relations: List[Dict[str, Any]]
    chosen_relation: str
    next_entity: Union[str, List[str]]
    model_reasoning: Optional[str] = None

@dataclass
class TraversalState:
    original_question: str
    exploration_trace: List[ExplorationRound] = field(default_factory=list)
    current_path: List[str] = field(default_factory=list)
    answer_found: bool = False
    final_answer_entities: List[str] = field(default_factory=list)
    final_reasoning_summary: str = ""
'''