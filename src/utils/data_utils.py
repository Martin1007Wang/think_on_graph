from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# 数据类定义
@dataclass
class EntityRelation:
    """存储实体关系信息"""
    relation: str
    targets: List[str] = field(default_factory=list)

@dataclass
class EntityExpansion:
    """存储实体扩展结果"""
    entity: str
    relations: List[EntityRelation] = field(default_factory=list)

@dataclass
class ExplorationRound:
    """存储一轮探索的结果"""
    round_num: int
    expansions: List[EntityExpansion] = field(default_factory=list)
    answer_found: Optional[Dict[str, Any]] = None

@dataclass
class Path:
    """存储路径信息"""
    elements: List[Dict[str, str]]
    
    @property
    def path_str(self) -> str:
        """获取路径的字符串表示"""
        if not self.elements:
            return ""
        path = self.elements[0]['source']
        for element in self.elements:
            path += f"-[{element['relation']}]->{element['target']}"
        return path
    
    @property
    def path_length(self) -> int:
        """获取路径长度"""
        return len(self.elements)