from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import time
from src.knowledge_graph import KnowledgeGraph
from src.llm_output_parser import LLMOutputParser
from src.template import KnowledgeGraphTemplates
class KnowledgeExplorer:
    """知识图谱探索器，负责探索知识图谱和生成答案。"""
    
    def __init__(self, kg: KnowledgeGraph, model: Any, max_rounds: int = 3, 
                 relation_k: int = 5, use_cache: bool = True) -> None:
        """初始化知识图谱探索器。

        Args:
            kg: 知识图谱接口
            model: 语言模型接口
            max_rounds: 最大探索轮数
            relation_k: 每个实体选择的关系数量
            use_cache: 是否使用缓存
        """
        self.kg = kg
        self.model = model
        self.max_rounds = max_rounds
        self.relation_k = relation_k
        self.use_cache = use_cache
        self.parser = LLMOutputParser()
        self.templates = KnowledgeGraphTemplates()
        
        # 初始化缓存
        if use_cache:
            # 使用LRU缓存对频繁调用的方法进行装饰
            self.get_related_relations = lru_cache(maxsize=1000)(self._get_related_relations)
            self.get_related_relations_with_context = lru_cache(maxsize=1000)(self._get_related_relations_with_context)
            self.score_frontier_entities = lru_cache(maxsize=1000)(self._score_frontier_entities)
        else:
            self.get_related_relations = self._get_related_relations
            self.get_related_relations_with_context = self._get_related_relations_with_context
            self.score_frontier_entities = self._score_frontier_entities
    
    def _get_related_relations(self, entity: str, question: str) -> List[str]:
        """获取与实体相关的关系，并让模型选择最相关的前k个。

        Args:
            entity: 待探索的实体
            question: 问题文本

        Returns:
            按相关性排序的关系列表
        """
        # 获取实体的所有相关关系
        start_time = time.time()
        related_relations = self.kg.get_related_relations(entity)
        logger.debug(f"Retrieved {len(related_relations)} relations in {time.time() - start_time:.2f}s")
        
        if not related_relations:
            logger.warning(f"No relations found for entity '{entity}'")
            return []
            
        if len(related_relations) <= self.relation_k:
            return related_relations
            
        # 让语言模型选择最相关的关系
        prompt = self.templates.RELATION_SELECTION.format(
            question=question,
            entity=entity,
            relations="\n".join(f"- {rel}" for rel in related_relations),
            relation_k=min(self.relation_k, len(related_relations))
        )
        
        model_input = self.model.prepare_model_prompt(prompt)
        selection_output = self.model.generate_sentence(model_input)
        
        # 解析输出，提取选定的关系
        selected_relations = self.parser.parse_selected_relations(selection_output, related_relations)
        return selected_relations[:self.relation_k]
    
    def _get_related_relations_with_context(self, entity: str, question: str, 
                                          entities_in_context_str: str, 
                                          exploration_history_str: str) -> List[str]:
        """考虑已探索上下文，获取相关关系。

        Args:
            entity: 待探索的实体
            question: 问题文本
            entities_in_context_str: 已探索实体的字符串表示
            exploration_history_str: 探索历史的字符串表示

        Returns:
            按相关性排序的关系列表
        """
        related_relations = self.kg.get_related_relations(entity)
        if not related_relations:
            logger.warning(f"No relations found for entity '{entity}'")
            return []
            
        if len(related_relations) <= self.relation_k:
            return related_relations
        
        # 让语言模型选择最相关的关系，考虑已有上下文
        prompt = self.templates.RELATION_SELECTION_WITH_CONTEXT.format(
            question=question,
            entity=entity,
            history=exploration_history_str,
            relations="\n".join(f"- {rel}" for rel in related_relations),
            relation_k=min(self.relation_k, len(related_relations))
        )
        
        model_input = self.model.prepare_model_prompt(prompt)
        selection_output = self.model.generate_sentence(model_input)
        
        # 解析输出，提取选定的关系
        selected_relations = self.parser.parse_selected_relations(selection_output, related_relations)
        return selected_relations[:self.relation_k]
    
    def _score_frontier_entities(self, frontier_str: str, question: str, 
                               entities_in_context_str: str) -> List[Tuple[str, float]]:
        """对前沿实体进行评分，评估其与问题的相关性。

        Args:
            frontier_str: 前沿实体列表的字符串表示（用逗号分隔）
            question: 问题文本
            entities_in_context_str: 已探索实体的字符串表示

        Returns:
            按评分降序排序的实体和分数对列表
        """
        frontier = frontier_str.split(',')
        if not frontier:
            return []
        
        # 让语言模型评估实体相关性
        prompt = self.templates.ENTITY_RANKING.format(
            question=question,
            explored=entities_in_context_str,
            candidates="\n".join([f"- {entity}" for entity in frontier])
        )
        
        model_input = self.model.prepare_model_prompt(prompt)
        output = self.model.generate_sentence(model_input)
        
        # 解析输出，提取实体评分
        return self.parser.parse_entity_scores(output, frontier)
    
    def format_exploration_history(self, exploration_history: List[Dict[str, Any]]) -> str:
        """将探索历史格式化为可读文本。

        Args:
            exploration_history: 探索历史数据

        Returns:
            格式化后的探索历史文本
        """
        result = []
        for round_data in exploration_history:
            round_num = round_data["round"]
            result.append(f"Round {round_num}:")
            for expansion in round_data["expansions"]:
                entity = expansion["entity"]
                result.append(f"  Entity: {entity}")
                for relation_info in expansion["relations"]:
                    relation = relation_info["relation"]
                    outgoing = relation_info["outgoing_targets"]
                    incoming = relation_info["incoming_targets"]
                    if outgoing:
                        out_str = ", ".join(outgoing)
                        result.append(f"    {entity} --[{relation}]--> {out_str}")
                    if incoming:
                        in_str = ", ".join(incoming)
                        result.append(f"    {in_str} --[{relation}]--> {entity}")
            result.append("")
        return "\n".join(result)
    
    def check_if_can_answer(self, question: str, start_entities: List[str], 
                            exploration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查基于当前探索是否能回答问题。

        Args:
            question: 问题文本
            start_entities: 起始实体列表
            exploration_history: 探索历史数据

        Returns:
            包含决策、答案和推理路径的字典
        """
        try:
            formatted_history = self.format_exploration_history(exploration_history)
            entities_str = ", ".join(start_entities)
            
            # 让语言模型判断是否能回答问题
            prompt = self.templates.REASONING.format(
                question=question,
                entity=entities_str,
                num_rounds=len(exploration_history),
                exploration_history=formatted_history
            )
            
            model_input = self.model.prepare_model_prompt(prompt)
            reasoning_output = self.model.generate_sentence(model_input)
            
            # 解析输出
            return self.parser.parse_reasoning_output(reasoning_output)
            
        except Exception as e:
            logger.warning(f"Error in check_if_can_answer: {str(e)}")
            return {"can_answer": False}
    
    def generate_final_answer(self, question: str, start_entities: List[str], 
                             exploration_history: List[Dict[str, Any]]) -> Dict[str, str]:
        """生成最终答案。

        Args:
            question: 问题文本
            start_entities: 起始实体列表
            exploration_history: 探索历史数据

        Returns:
            包含答案和推理的字典
        """
        formatted_history = self.format_exploration_history(exploration_history)
        entities_str = ", ".join(start_entities)
        
        # 让语言模型生成最终答案
        prompt = self.templates.FINAL_ANSWER.format(
            question=question,
            entity=entities_str,
            full_exploration=formatted_history
        )
        
        model_input = self.model.prepare_model_prompt(prompt)
        final_output = self.model.generate_sentence(model_input)
        
        # 解析输出
        return self.parser.parse_final_answer(final_output)
    
    def explore_knowledge_graph(self, question: str, start_entities: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """探索知识图谱，使用更有针对性的方法。

        Args:
            question: 问题文本
            start_entities: 起始实体或实体列表

        Returns:
            探索历史数据
        """
        if isinstance(start_entities, str):
            start_entities = [start_entities]
        
        logger.info(f"Starting exploration with entities: {start_entities}")
        
        exploration_history = []
        entities_explored = {}
        entities_in_context = set()  # 跟踪已经在上下文中的实体
        
        # 初始化探索前沿
        frontier = [entity for entity in start_entities if entity and entity not in entities_explored]
        for entity in frontier:
            entities_explored[entity] = True
            entities_in_context.add(entity)
            
        # 主探索循环
        for round_num in range(self.max_rounds):
            logger.info(f"Round {round_num+1}: Exploring {len(frontier)} frontier entities")
            
            if not frontier:
                logger.info("No frontier entities to explore, ending exploration")
                break
                
            # 评估前沿实体相关性
            entities_context_str = ", ".join(entities_in_context)
            frontier_str = ",".join(frontier)
            frontier_scores = self.score_frontier_entities(frontier_str, question, entities_context_str)
            
            # 选择最相关的前K个实体
            top_frontier = frontier_scores[:min(5, len(frontier_scores))]
            logger.info(f"Selected {len(top_frontier)} most relevant entities from frontier")
            
            round_exploration = {"round": round_num + 1, "expansions": []}
            new_frontier = []
            
            # 探索所选实体
            exploration_history_str = self.format_exploration_history(exploration_history)
            
            for entity, relevance in top_frontier:
                # 选择关系时考虑已有上下文
                selected_relations = self.get_related_relations_with_context(
                    entity, question, entities_context_str, exploration_history_str
                )
                
                if not selected_relations:
                    logger.info(f"No relations found for entity: {entity}")
                    continue
                    
                logger.info(f"Selected {len(selected_relations)} relations for entity {entity}")
                entity_expansion = {"entity": entity, "relations": []}
                
                # 对每个选定的关系进行探索
                for relation in selected_relations:
                    try:
                        # 批量查询目标实体，提高效率
                        target_entities_out = self.kg.get_target_entities(entity, relation, "out")
                        target_entities_in = self.kg.get_target_entities(entity, relation, "in") 
                    except Exception as e:
                        logger.error(f"Error querying knowledge graph: {str(e)}")
                        continue
                    
                    # 记录关系信息
                    relation_info = {
                        "relation": relation,
                        "outgoing_targets": target_entities_out[:5],  # 限制数量避免过载
                        "incoming_targets": target_entities_in[:5]
                    }
                    entity_expansion["relations"].append(relation_info)
                    
                    # 更新实体上下文和探索前沿
                    for target in target_entities_out + target_entities_in:
                        if target:
                            entities_in_context.add(target)
                            if target not in entities_explored:
                                new_frontier.append(target)
                                entities_explored[target] = True
                
                # 如果找到了关系，添加到本轮探索结果
                if entity_expansion["relations"]:
                    round_exploration["expansions"].append(entity_expansion)
            
            # 添加本轮探索结果
            if round_exploration["expansions"]:
                exploration_history.append(round_exploration)
                
                # 检查是否已经能回答问题
                can_answer = self.check_if_can_answer(question, start_entities, exploration_history)
                if can_answer.get("can_answer", False):
                    logger.info(f"Found answer after round {round_num+1}, ending exploration")
                    
                    # 记录找到的答案
                    round_exploration["answer_found"] = {
                        "answer": can_answer.get("answer", ""),
                        "reasoning_path": can_answer.get("reasoning_path", "")
                    }
                    break
            else:
                logger.info("No new information found in this round")
                
            # 检查是否继续探索
            if not new_frontier:
                logger.info("No new entities to explore, ending exploration")
                break
                
            # 更新前沿队列
            frontier = new_frontier
        
        return exploration_history
    
    def process_question(self, data: Dict[str, Any], processed_ids: List[str]) -> Optional[Dict[str, Any]]:
        """处理单个问题，执行整个回答流程。

        Args:
            data: 包含问题和元数据的字典
            processed_ids: 已处理的问题ID列表

        Returns:
            包含问题、答案和推理过程的结果字典，如果问题已处理则返回None
        """
        question = data["question"]
        question_id = data["id"]
        
        # 跳过已处理的问题
        if question_id in processed_ids:
            return None
        
        # 获取起始实体
        start_entities = data.get("q_entity", data.get("entity", []))
        if not start_entities:
            logger.warning(f"No query entities found for question ID {question_id}")
            return None
            
        logger.info(f"Processing question {question_id}: {question}")
        
        try:
            # 探索知识图谱
            exploration_history = self.explore_knowledge_graph(question, start_entities)
            
            if not exploration_history:
                logger.warning(f"No exploration history for question ID {question_id}")
                return None
                
            # 生成最终答案
            final_result = self.generate_final_answer(question, start_entities, exploration_history)
            
            # 构建结果
            result = {
                "id": question_id,
                "question": question,
                "start_entities": start_entities,
                "prediction": final_result["answer"],
                "reasoning": final_result["reasoning"],
                "exploration_history": exploration_history,
                "formatted_exploration": self.format_exploration_history(exploration_history)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}", exc_info=True)
            return None