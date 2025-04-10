from typing import Any, Dict, List, Tuple, Union
import re

class LLMOutputParser:
    """解析语言模型输出。"""    
    
    @staticmethod
    def parse_relations(llm_output: Union[str, List[str]], available_relations: List[str]) -> List[str]:
        def parse_single_output(output: str) -> List[str]:
            selected = []
            lines = output.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                try:
                    if '[' in line and ']' in line:
                        start_idx = line.find('[')
                        end_idx = line.find(']', start_idx)
                        if start_idx != -1 and end_idx != -1:
                            relation_part = line[start_idx+1:end_idx].strip()
                            if relation_part in available_relations:
                                selected.append(relation_part)
                                continue
                            matching_relations = [r for r in available_relations if r.lower() == relation_part.lower()]
                            if matching_relations:
                                selected.append(matching_relations[0])
                            continue
                    if '. ' in line:
                        parts = line.split('. ', 1)[1]
                        if ' - ' in parts:
                            relation_part = parts.split(' - ', 1)[0].strip('[]')
                        else:
                            relation_part = parts.strip('[]')
                        matching_relations = [r for r in available_relations if r.lower() == relation_part.lower()]
                        if matching_relations:
                            selected.append(matching_relations[0])
                except Exception as e:
                    continue
            return selected
        
        if isinstance(llm_output, str):
            return parse_single_output(llm_output)
        relation_scores = {}
        total_outputs = len(llm_output)
        for output in llm_output:
            selected = parse_single_output(output)
            for relation in selected:
                relation_scores[relation] = relation_scores.get(relation, 0) + 1
        scored_relations = [(relation, count / total_outputs) for relation, count in relation_scores.items()]
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        return [relation for relation, _ in scored_relations]

    @staticmethod
    def parse_entities(output: str, frontier: List[str]) -> List[Tuple[str, float]]:
        entity_scores = []
        for line in output.strip().split('\n'):
            if ':' in line and '-' in line:
                try:
                    entity_part = line.split(':', 1)[0].strip()
                    score_part = line.split(':', 1)[1].split('-', 1)[0].strip()
                    score = float(score_part)
                    entity = entity_part
                    matching_entities = [e for e in frontier if e.lower() == entity.lower()]
                    if matching_entities:
                        entity_scores.append((matching_entities[0], score))
                except (ValueError, IndexError):
                    continue        
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores

    @staticmethod
    def parse_reasoning_output(output: str) -> Dict[str, Any]:
        sections = output.split('\n\n')
        decision = ""
        reasoning_path = ""
        preliminary_answer = ""
        verification = ""
        final_answer = ""
        
        for section in sections:
            section = section.strip()
            if section.startswith('Decision:'):
                decision = section.replace('Decision:', '').strip()
            elif section.startswith('Reasoning path:'):
                # 处理多行推理路径，去掉序号
                path_lines = section.replace('Reasoning path:', '').strip().split('\n')
                for line in path_lines:
                    if '--[' in line and ']-->' in line:
                        # 如果有序号，去掉序号
                        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                            reasoning_path = line.split('.', 1)[1].strip()
                        else:
                            reasoning_path = line.strip()
                        break
            elif section.startswith('Preliminary Answer:'):
                preliminary_answer = section.replace('Preliminary Answer:', '').strip()
            elif section.startswith('Verification:'):
                verification = section.replace('Verification:', '').strip()
            elif section.startswith('Final Answer:'):
                final_answer = section.replace('Final Answer:', '').strip()
        
        answer = final_answer or preliminary_answer
        
        if decision.lower() == 'yes' and answer:
            is_verified = verification.lower().startswith("yes") or "directly follows" in verification.lower()
            
            return {
                "can_answer": True,
                "answer": answer,
                "reasoning_path": reasoning_path,
                "preliminary_answer": preliminary_answer,
                "verification": verification,
                "is_verified": is_verified
            }
        else:
            return {
                "can_answer": False
            }

    @staticmethod
    def parse_final_answer(output: str) -> Dict[str, str]:
        """解析最终答案输出。

        Args:
            output: 语言模型生成的最终答案输出

        Returns:
            包含答案和推理的字典
        """
        lines = output.strip().split('\n')
        answer = ""
        reasoning = ""
        
        for line in lines:
            line = line.strip('[]')
            if line.startswith("Final Answer: "):
                answer = line.split("Final Answer: ", 1)[1].strip()
            elif line.startswith("Reasoning: "):
                reasoning = line.split("Reasoning: ", 1)[1].strip()
        
        return {"answer": answer, "reasoning": reasoning}