import json
from pathlib import Path
from typing import List, Dict, Any, Set

def clean_jsonl_file(input_path: str, output_path: str = None, id_file: str = None) -> int:
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")
    else:
        output_path = Path(output_path)
    
    if id_file is None:
        id_file = input_path.parent / "removed_ids.txt"
    else:
        id_file = Path(id_file)
    
    # Error message to filter out
    error_msg = "'NoneType' object is not subscriptable. The knowledge graph does not contain sufficient information to address this query."
    
    # Read the input file and filter out problematic entries
    filtered_data: List[Dict[str, Any]] = []
    removed_ids: Set[str] = set()
    removed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                
                # Check conditions for removal
                should_remove = False
                
                # Check if prediction is empty or matches the specific message
                if ("prediction" not in entry or entry["prediction"] is None or entry["prediction"] == ""
                    or entry["prediction"] == "I cannot answer this question based on the available knowledge."):
                    should_remove = True
                
                if ("is_verified" not in entry or entry["is_verified"] is None or entry["is_verified"] == False):
                    should_remove = True
                # Check if reasoning contains the error message
                if "reasoning" in entry and entry["reasoning"] and error_msg in entry["reasoning"]:
                    should_remove = True
                
                if should_remove:
                    removed_count += 1
                    if "id" in entry:
                        removed_ids.add(entry["id"])
                else:
                    filtered_data.append(entry)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
    
    # Write the filtered data to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Write the removed IDs to a text file
    with open(id_file, 'w', encoding='utf-8') as f:
        for id_val in sorted(removed_ids):
            f.write(f"{id_val}\n")
    
    print(f"Processed: {input_path}")
    print(f"Removed {removed_count} entries")
    print(f"Saved {len(removed_ids)} IDs to: {id_file}")
    print(f"Output written to: {output_path}")
    
    return removed_count

if __name__ == "__main__":
    clean_jsonl_file("/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v8/RoG-webqsp/GCR-lora-sft_v3_Llama-3.1-8B-Instruct/deepseek-v3/iterative-rounds3-topk5/predictions.jsonl")