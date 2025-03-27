from typing import Any, Dict, List, Tuple, Union

class LLMOutputParser:
    """解析语言模型输出。"""    
    
    @staticmethod
    def parse_selected_relations(llm_output: Union[str, List[str]], 
                                 available_relations: List[str]) -> List[str]:
        """解析语言模型输出以提取选定的关系。

        Args:
            llm_output: 语言模型生成的输出，可以是单个字符串或字符串列表
            available_relations: 可用关系列表

        Returns:
            按相关性排序的关系列表
        """
        def parse_single_output(output: str) -> List[str]:
            """解析单个输出字符串。"""
            selected = []
            lines = output.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or not line[0].isdigit() or '. ' not in line:
                    continue
                try:
                    # 提取关系部分
                    relation_part = line.split('. ', 1)[1].split(' - ', 1)[0].strip('[]')
                    # 不区分大小写匹配
                    matching_relations = [r for r in available_relations if r.lower() == relation_part.lower()]
                    if matching_relations:
                        selected.append(matching_relations[0])
                except IndexError:
                    continue
            return selected

        # 处理单个字符串输出
        if isinstance(llm_output, str):
            return parse_single_output(llm_output)
        
        # 处理多个输出并统计频率
        relation_scores = {}
        total_outputs = len(llm_output)
        
        for output in llm_output:
            selected = parse_single_output(output)
            for relation in selected:
                relation_scores[relation] = relation_scores.get(relation, 0) + 1

        # 计算每个关系的得分（频率）并排序
        scored_relations = [
            (relation, count / total_outputs)
            for relation, count in relation_scores.items()
        ]
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的关系列表
        return [relation for relation, _ in scored_relations]

    @staticmethod
    def parse_entity_scores(output: str, frontier: List[str]) -> List[Tuple[str, float]]:
        """解析实体评分输出。

        Args:
            output: 语言模型生成的输出
            frontier: 待评分的实体列表

        Returns:
            按评分降序排序的实体和分数对列表
        """
        entity_scores = []
        for line in output.strip().split('\n'):
            if ':' in line and '-' in line:
                try:
                    entity_part = line.split(':', 1)[0].strip()
                    score_part = line.split(':', 1)[1].split('-', 1)[0].strip()
                    score = float(score_part)
                    entity = entity_part
                    
                    # 确保实体在前沿列表中（不区分大小写匹配）
                    matching_entities = [e for e in frontier if e.lower() == entity.lower()]
                    if matching_entities:
                        entity_scores.append((matching_entities[0], score))
                except (ValueError, IndexError):
                    continue
        
        # 按评分降序排序
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores

    @staticmethod
    def parse_reasoning_output(output: str) -> Dict[str, Any]:
        """解析推理输出。

        Args:
            output: 语言模型生成的推理输出

        Returns:
            包含决策、答案和推理路径的字典
        """
        lines = output.strip().split('\n')
        decision = "No"
        answer = ""
        reasoning_path = ""
        missing_info = ""
        
        if lines:
            first_line = lines[0].strip('[]')
            if ': ' in first_line and first_line.startswith("Decision"):
                decision = first_line.split(': ', 1)[1].strip()
        
        for line in lines[1:]:
            line = line.strip('[]')
            if line.startswith("Answer: "):
                answer = line.split("Answer: ", 1)[1].strip()
            elif line.startswith("Reasoning path: "):
                reasoning_path = line.split("Reasoning path: ", 1)[1].strip()
            elif line.startswith("Missing information: "):
                missing_info = line.split("Missing information: ", 1)[1].strip()
        
        if decision.lower() == "yes" and answer:
            return {
                "can_answer": True,
                "answer": answer,
                "reasoning_path": reasoning_path
            }
        else:
            return {
                "can_answer": False,
                "missing_info": missing_info
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