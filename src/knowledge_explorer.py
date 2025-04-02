import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import time
from src.knowledge_graph import KnowledgeGraph
from src.llm_output_parser import LLMOutputParser
from src.template import KnowledgeGraphTemplates

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class KnowledgeExplorer:
    """Knowledge graph explorer for answering questions."""
    
    def __init__(self, kg: KnowledgeGraph, model: Any, max_rounds: int = 3, 
                 relation_k: int = 5, use_cache: bool = True) -> None:
        """Initialize the knowledge explorer."""
        self.kg = kg
        self.model = model
        self.max_rounds = max_rounds
        self.relation_k = relation_k
        self.parser = LLMOutputParser()
        self.templates = KnowledgeGraphTemplates()
        
        # Apply caching if enabled
        if use_cache:
            self.get_related_relations = lru_cache(maxsize=1000)(self._get_related_relations)
        else:
            self.get_related_relations = self._get_related_relations
    
    def _get_related_relations(self, entity: str, question: str, 
                             context: str = "", history: str = "") -> List[str]:
        out_related_relations = self.kg.get_related_relations(entity, "out")
        in_related_relations = self.kg.get_related_relations(entity, "in")

        if not out_related_relations and not in_related_relations:
            logger.error(f"No relations found for entity '{entity}'")
            return []
            
        # 根据是否有上下文选择不同的模板
        template = (self.templates.RELATION_SELECTION_WITH_CONTEXT 
                   if context and history 
                   else self.templates.RELATION_SELECTION)
        
        # 准备prompt参数
        prompt_args = {
            "question": question,
            "entity": entity,
            "relations": "\n".join(f"- {rel}" for rel in related_relations),
            "relation_k": min(self.relation_k, len(related_relations))
        }
        
        # 如果有上下文信息，添加到参数中
        if context and history:
            prompt_args.update({
                "history": history,
                "context": context
            })
        
        prompt = template.format(**prompt_args)
        model_input = self.model.prepare_model_prompt(prompt)
        
        # 使用greedy策略进行关系选择
        selection_output = self.model.generate_sentence(
            model_input,
            temp_generation_mode="greedy"  # 快速生成
        )
        
        selected_relations = self.parser.parse_selected_relations(selection_output, related_relations)
        return selected_relations[:self.relation_k]
    
    def format_exploration_round(self, round_data: Dict[str, Any]) -> str:
        """Format only a single round of exploration."""
        result = []
        round_num = round_data["round"]
        result.append(f"Round {round_num}:")
        for expansion in round_data["expansions"]:
            entity = expansion["entity"]
            result.append(f"  Entity: {entity}")
            for relation_info in expansion["relations"]:
                relation = relation_info["relation"]
                targets = relation_info["targets"]
                if targets:
                    result.append(f"    {entity} --[{relation}]--> {', '.join(targets)}")
        result.append("")
        return "\n".join(result)
    
    def format_exploration_history(self, exploration_history: List[Dict[str, Any]]) -> str:
        """Format exploration history as readable text."""
        result = []
        for round_data in exploration_history:
            round_num = round_data["round"]
            result.append(f"Round {round_num}:")
            for expansion in round_data["expansions"]:
                entity = expansion["entity"]
                result.append(f"  Entity: {entity}")
                for relation_info in expansion["relations"]:
                    relation = relation_info["relation"]
                    targets = relation_info["targets"]
                    if targets:  # 只展示出边关系
                        result.append(f"    {entity} --[{relation}]--> {', '.join(targets)}")
            result.append("")
        return "\n".join(result)
    
    def check_if_can_answer(self, question: str, start_entities: List[str], 
                            exploration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if we can answer the question based on current exploration."""
        try:
            formatted_history = self.format_exploration_history(exploration_history)
            
            prompt = self.templates.REASONING.format(
                question=question,
                entity=", ".join(start_entities),
                num_rounds=len(exploration_history),
                exploration_history=formatted_history
            )
            
            model_input = self.model.prepare_model_prompt(prompt)
            # 使用beam search进行推理判断
            reasoning_output = self.model.generate_sentence(
                model_input,
                temp_generation_mode="beam",  # 使用beam search平衡速度和质量
                num_beams=4  # 可以调整beam数量
            )
            
            return self.parser.parse_reasoning_output(reasoning_output)
            
        except Exception as e:
            logger.warning(f"Error in check_if_can_answer: {str(e)}")
            return {"can_answer": False}
    
    def explore_knowledge_graph(self, question: str, start_entities: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Explore the knowledge graph to answer the question."""
        # Normalize input
        if isinstance(start_entities, str):
            start_entities = [start_entities]
        
        logger.info(f"Starting exploration with entities: {start_entities}")
        
        exploration_history = []
        entities_explored = set(start_entities)
        entities_in_context = set(start_entities)
        frontier = list(start_entities)
        
        # Main exploration loop
        for round_num in range(self.max_rounds):
            logger.info(f"Round {round_num+1}: Exploring {len(frontier)} frontier entities")
            
            if not frontier:
                logger.info("No frontier entities to explore, ending exploration")
                break
                
            round_exploration = {"round": round_num + 1, "expansions": []}
            new_frontier = []
            
            # Explore selected entities
            exploration_history_str = self.format_exploration_history(exploration_history)
            entities_context_str = ", ".join(entities_in_context)
            
            # 直接遍历所有frontier实体，不再进行评分和筛选
            for entity in frontier:
                # Get relevant relations considering context
                selected_relations = self.get_related_relations(
                    entity, question, entities_context_str, exploration_history_str
                )
                
                if not selected_relations:
                    continue
                    
                entity_expansion = {"entity": entity, "relations": []}
                
                # Explore each relation
                for relation in selected_relations:
                    try:
                        # 只获取出边目标实体
                        outgoing = self.kg.get_target_entities(entity, relation, "out")[:5]
                    except Exception as e:
                        logger.error(f"Error querying knowledge graph: {str(e)}")
                        continue
                    
                    # Record relation information
                    relation_info = {
                        "relation": relation,
                        "targets": outgoing  # 简化为只包含目标实体
                    }
                    entity_expansion["relations"].append(relation_info)
                    
                    # Update context and frontier
                    for target in outgoing:
                        if target:
                            entities_in_context.add(target)
                            if target not in entities_explored:
                                new_frontier.append(target)
                                entities_explored.add(target)
                
                if entity_expansion["relations"]:
                    round_exploration["expansions"].append(entity_expansion)
            
            # Add round results to history
            if round_exploration["expansions"]:
                exploration_history.append(round_exploration)
                
                # Check if we can answer the question
                can_answer = self.check_if_can_answer(question, start_entities, exploration_history)
                if can_answer.get("can_answer", False):
                    logger.info(f"Found answer after round {round_num+1}, ending exploration")
                    round_exploration["answer_found"] = {
                        "answer": can_answer.get("answer", ""),
                        "reasoning_path": can_answer.get("reasoning_path", "")
                    }
                    break
            else:
                logger.info("No new information found in this round")
                
            # Update frontier for next round
            frontier = new_frontier
            if not frontier:
                logger.info("No new entities to explore, ending exploration")
                break
        
        return exploration_history
    
    def process_question(self, data: Dict[str, Any], processed_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Process a single question and generate an answer."""
        question = data["question"]
        question_id = data["id"]
        
        # Skip already processed questions
        if question_id in processed_ids:
            return None
        
        # Get starting entities
        start_entities = data.get("q_entity", data.get("entity", []))
        if not start_entities:
            logger.warning(f"No query entities found for question ID {question_id}")
            return None
            
        logger.info(f"Processing question {question_id}: {question}")
        
        try:
            # Explore knowledge graph
            exploration_history = self.explore_knowledge_graph(question, start_entities)
            
            if not exploration_history:
                logger.warning(f"No exploration history for question ID {question_id}")
                return None
            
            # 获取最后一轮的答案（如果找到了答案）
            final_round = exploration_history[-1]
            answer_found = final_round.get("answer_found", {})
            
            # Return result
            return {
                "id": question_id,
                "question": question,
                "start_entities": start_entities,
                "prediction": answer_found.get("answer", ""),  # 使用探索过程中找到的答案
                "reasoning": answer_found.get("reasoning_path", ""),  # 使用探索过程中的推理路径
                "exploration_history": exploration_history,
                "formatted_exploration": self.format_exploration_history(exploration_history)
            }
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}", exc_info=True)
            return None
