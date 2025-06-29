from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ExplorationRound:
    round_num: int
    question: str
    entity: str
    relations: List[Dict[str, Any]]

@dataclass
class TraversalState:
    original_question: str
    exploration_trace: List[ExplorationRound] = field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    answer_found: bool = False
    final_answer_entities: List[str] = field(default_factory=list)
    final_reasoning_summary: str = ""
    
@dataclass
class Path:
    elements: List[Dict[str, str]] = field(default_factory=list)

    def to_string(self) -> str:
        if not self.elements:
            return ""
        
        path_str = self.elements[0]['source']
        for hop in self.elements:
            path_str += f" --[{hop['relation']}]--> {hop['target']}"
        return path_str