from typing import Any, Dict, List, Tuple, Union, Optional, Set
import re
import json
import logging
import difflib

# 日志记录器
logger = logging.getLogger(__name__)

class LLMOutputParser:
    @staticmethod
    def _ensure_string(llm_output: Union[str, List[str]], caller: str) -> Optional[str]:
        if llm_output is None:
            return None
        if isinstance(llm_output, str):
            output_text = llm_output
        elif isinstance(llm_output, list):
            try:
                output_text = "\n".join(map(str, llm_output))
            except TypeError as e:
                logger.error(f"[{caller}] 无法将列表元素转换为字符串进行连接: {e}")
                return None
        else:
            logger.warning(f"[{caller}] 接收到意外的输入类型: {type(llm_output)}。应为 str 或 List[str]。")
            return None
        return output_text.strip()

class LLMOutputParser:
    @staticmethod
    def _ensure_string(output: Union[str, List[str]], caller: str) -> str:
        if isinstance(output, list): return "\n".join(output)
        if isinstance(output, str): return output
        return ""

    @staticmethod
    def parse_indexed_selections(
        llm_output: Union[str, List[str]],
        candidate_items: List[str]
    ) -> List[str]:
        caller_name = "parse_indexed_selections"
        output_text = LLMOutputParser._ensure_string(llm_output, caller_name)
        if not output_text or not candidate_items:
            return []
        try:
            indices_str = re.findall(r'\[REL(\d+)\]', output_text)
            indices = [int(i) - 1 for i in indices_str]
        except (ValueError, TypeError):
            logger.error(f"[{caller_name}] Could not parse numbers from regex matches in output: {output_text}")
            return []
            
        selected_items: Set[str] = set()
        for idx in indices:
            if 0 <= idx < len(candidate_items):
                selected_items.add(candidate_items[idx])
            else:
                logger.warning(f"[{caller_name}] LLM returned out-of-bounds index: {idx+1} "
                               f"(candidate list size: {len(candidate_items)})")
        ordered_selection = [item for item in candidate_items if item in selected_items]
        logger.debug(f"[{caller_name}] Indexed selection result: {ordered_selection}")
        return ordered_selection

    @staticmethod
    def parse_relations(
        llm_output: Union[str, List[str]], 
        candidate_items: List[str]
    ) -> List[str]:
        return LLMOutputParser.parse_indexed_selections(
            llm_output,
            candidate_items=candidate_items
        )

    @staticmethod
    def parse_reasoning_output(output: Union[str, List[str]]) -> Optional[Dict[str, Any]]:
        caller_name = "parse_reasoning_output"
        output_text = LLMOutputParser._ensure_string(output, caller_name)
        if not output_text:
            logger.warning(f"[{caller_name}] 接收到空输出。")
            return None

        json_str = None
        
        # 尝试多种JSON提取方式
        patterns = [
            r"```(?:json)?\s*(\{.*?\})\s*```",  # 代码块格式
            r"```(?:json)?\s*(\[.*?\])\s*```",  # 数组格式
            r"(\{(?:[^{}]|{[^{}]*})*\})",       # 更宽松的对象匹配
        ]
        
        for pattern in patterns:
            fence_match = re.search(pattern, output_text, re.DOTALL)
            if fence_match:
                json_str = fence_match.group(1).strip()
                break
        
        # 如果没找到，尝试寻找裸JSON
        if not json_str:
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1
            if 0 <= json_start < json_end:
                potential_json = output_text[json_start:json_end]
                # 简单的括号平衡检查
                if potential_json.count('{') == potential_json.count('}'):
                    json_str = potential_json
    
        if not json_str:
            logger.error(f"[{caller_name}] 在输出中找不到有效的JSON对象块: {output_text[:500]}...")
            # 返回一个默认的结构而不是None
            return {
                "answer_found": False,
                "error": "Failed to parse JSON from output",
                "raw_output": output_text[:200]  # 保留部分原始输出用于调试
            }

        try:
            parsed_data = json.loads(json_str)
            if not isinstance(parsed_data, dict):
                logger.error(f"[{caller_name}] 解析的JSON不是字典类型。")
                return {
                    "answer_found": False,
                    "error": "Parsed JSON is not a dictionary"
                }
            
            # 确保必需字段存在
            if 'answer_found' not in parsed_data:
                logger.warning(f"[{caller_name}] 解析的JSON缺少'answer_found'字段，设置默认值。")
                parsed_data['answer_found'] = False
                
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"[{caller_name}] JSON解码失败: {e}")
            return {
                "answer_found": False,
                "error": f"JSON decode error: {str(e)}",
                "raw_json": json_str[:200]
            }
        
    @staticmethod
    def parse_fallback_answer(output: Union[str, List[str]]) -> Dict[str, Any]:
        caller_name = "parse_fallback_answer"
        
        # 1. 修改：定义符合新格式的、安全的默认返回字典
        final_result = {
            "can_answer": True,
            "reasoning_path": "[Parsing Failed: No path extracted]",
            "answer_entities": [],
            "reasoning_summary": "Failed to parse a valid response from the language model."
        }

        output_text = LLMOutputParser._ensure_string(output, caller_name)
        if not output_text:
            logger.warning(f"[{caller_name}] Received empty output.")
            return final_result

        # --- JSON 提取逻辑 (这部分是健壮的，予以保留) ---
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL | re.IGNORECASE)
        json_string = ""
        if match:
            json_string = match.group(1).strip()
        else:
            stripped_output = output_text.strip()
            if stripped_output.startswith('{') and stripped_output.endswith('}'):
                logger.warning(f"[{caller_name}] No JSON markdown fences found. Attempting to parse entire input as JSON.")
                json_string = stripped_output
            else:
                logger.error(f"[{caller_name}] Failed to find JSON structure in output. Raw: '{output_text[:200]}...'")
                final_result["reasoning_summary"] = "Output format error: Expected a JSON object."
                return final_result

        if not json_string:
            logger.error(f"[{caller_name}] Extracted JSON string is empty.")
            final_result["reasoning_summary"] = "Output format error: Extracted JSON content is empty."
            return final_result

        # --- JSON 解析逻辑 (保留健壮的 try-except 块) ---
        try:
            parsed_data = json.loads(json_string)
            if not isinstance(parsed_data, dict):
                logger.error(f"[{caller_name}] Parsed JSON is not a dictionary.")
                final_result["reasoning_summary"] = "Parsed JSON content is not a valid object."
                return final_result
        except json.JSONDecodeError as e:
            logger.error(f"[{caller_name}] Failed to decode JSON string. Error: {e}")
            final_result["reasoning_summary"] = f"JSON parsing error: {e}"
            return final_result
        
        # --- 2. 核心修改：从解析后的字典中提取新字段 ---
        
        # 使用 .get() 安全地提取每个字段，如果缺失则保留默认值
        path = parsed_data.get("reasoning_path")
        if isinstance(path, str) and path.strip():
            final_result["reasoning_path"] = path

        entities = parsed_data.get("answer_entities")
        if isinstance(entities, list): # 允许是空列表
            final_result["answer_entities"] = [str(e) for e in entities] # 确保列表内都是字符串

        summary = parsed_data.get("reasoning_summary")
        if isinstance(summary, str) and summary.strip():
            final_result["reasoning_summary"] = summary
            
        # 3. 强制规则：无论LLM返回什么，都严格遵守 `can_answer` 必须为 True 的指令
        final_result["can_answer"] = True

        logger.debug(f"[{caller_name}] Successfully parsed fallback answer.")
        return final_result