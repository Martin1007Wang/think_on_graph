from typing import List, Dict, Any
from src.utils.data_utils import Path

class PathManager:    
    def __init__(self):
        self.paths = []
        
    def is_coded_entity(self, entity: str) -> bool:
        return isinstance(entity, str) and (entity.startswith("m.") or entity.startswith("g."))
    
    def add_path(self, path_elements: List[Dict[str, str]]) -> None:
        if not path_elements:
            return
        path = Path(elements=path_elements)
        self.paths.append(path)
    
    def get_all_paths(self) -> List[Path]:
        return self.paths
    
    def get_multi_hop_paths(self) -> List[Dict[str, Any]]:
        multi_hop_paths = []
        for path in self.paths:
            if path.path_length < 2:
                continue
            elements = path.elements
            last_element = elements[-1]
            if self.is_coded_entity(last_element['target']):
                continue
            legacy_info = {
                "source": elements[0]['source'],
                "source_relation": elements[0]['relation'],
                "intermediate": elements[1]['source'],
                "target_relation": elements[1]['relation'],
                "targets": [last_element['target']]
            }
            for existing_path in multi_hop_paths:
                if (existing_path.get("source") == legacy_info["source"] and
                    existing_path.get("source_relation") == legacy_info["source_relation"] and
                    existing_path.get("intermediate") == legacy_info["intermediate"] and
                    existing_path.get("target_relation") == legacy_info["target_relation"]):
                    if legacy_info["targets"][0] not in existing_path.get("targets", []):
                        existing_path["targets"].append(legacy_info["targets"][0])
                    break
            else:
                multi_hop_paths.append(legacy_info)
        return multi_hop_paths
    
    def format_paths(self) -> str:
        if not self.paths:
            return ""
        result = ["Exploration Paths:"]
        for path in self.paths:
            result.append(f"  {path.path_str}")
        return "\n".join(result)