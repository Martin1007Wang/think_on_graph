"""
知识图谱探索器模块，用于基于知识图谱回答问题。
"""
import logging
import re
import os
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from src.knowledge_graph import KnowledgeGraph
from src.llm_output_parser import LLMOutputParser
from src.template import KnowledgeGraphTemplates

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class KnowledgeExplorer:
    """知识图谱探索器，用于基于知识图谱回答问题。"""
    
    def __init__(
        self, 
        kg: KnowledgeGraph, 
        model: Any, 
        max_rounds: int = 3, 
        relation_k: int = 10, 
    ) -> None:
        """
        初始化知识图谱探索器。
        
        Args:
            kg: 知识图谱对象
            model: 语言模型对象
            max_rounds: 最大探索轮次
            relation_k: 每个实体选择的关系数量
        """
        self.kg = kg
        self.model = model
        self.max_rounds = max_rounds
        self.relation_k = relation_k
        self.parser = LLMOutputParser()
        self.templates = KnowledgeGraphTemplates()
    
    def format_exploration_round(self, round_data: Dict[str, Any]) -> str:
        """
        格式化单轮探索结果为可读文本。
        
        Args:
            round_data: 包含单轮探索信息的字典
            
        Returns:
            格式化的探索结果文本
        """
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
        """
        格式化完整探索历史为可读文本。
        
        Args:
            exploration_history: 探索历史列表
            
        Returns:
            格式化的完整探索历史文本
        """
        result = []
        for round_data in exploration_history:
            result.append(self.format_exploration_round(round_data))
        return "\n".join(result).rstrip()
    
    def _get_related_relations(
        self, 
        entity: str, 
        question: str, 
        context: str = "", 
        history: str = ""
    ) -> List[str]:
        """
        获取与实体相关的关系。
        
        Args:
            entity: 实体ID
            question: 问题文本
            context: 上下文信息
            history: 探索历史
            
        Returns:
            相关关系列表
        """
        # 获取与实体相关的出向关系
        out_related_relations = self.kg.get_related_relations(entity, "out")
        
        if not out_related_relations:
            logger.error(f"No relations found for entity '{entity}'")
            return []
        
        # 根据是否有上下文选择不同的模板
        template = (
            self.templates.RELATION_SELECTION_WITH_CONTEXT 
            if context and history 
            else self.templates.RELATION_SELECTION
        )
        
        # 准备prompt参数
        prompt_args = {
            "question": question,
            "entity": entity,
            "relations": "\n".join(f"[{rel}]" for rel in out_related_relations),
            "relation_k": min(self.relation_k, len(out_related_relations))
        }
        
        # 添加上下文信息（如果有）
        if context and history:
            prompt_args.update({
                "history": history,
                "context": context
            })
        
        # 生成prompt并准备模型输入
        prompt = template.format(**prompt_args)
        model_input = self.model.prepare_model_prompt(prompt)
        
        selection_output = self.model.generate_sentence(
            model_input,
            temp_generation_mode="beam"
        )
        
        # 解析选择的关系
        selected_relations = self.parser.parse_selected_relations(
            selection_output, 
            out_related_relations
        )
        
        return selected_relations[:self.relation_k]
    
    def _expand_entity(
        self, 
        entity: str, 
        question: str, 
        context: str, 
        history: str
    ) -> Dict[str, Any]:
        """
        展开单个实体，探索其相关关系和目标实体。
        
        Args:
            entity: 要展开的实体
            question: 问题文本
            context: 实体上下文
            history: 探索历史
            
        Returns:
            包含实体展开信息的字典
        """
        # 获取相关关系
        selected_relations = self._get_related_relations(
            entity, question, context, history
        )
        
        if not selected_relations:
            return None
        
        entity_expansion = {"entity": entity, "relations": []}
        
        # 探索每个关系
        for relation in selected_relations:
            try:
                # 获取关系目标实体（限制为前5个）
                outgoing = self.kg.get_target_entities(entity, relation, "out")[:5]
                
                if outgoing:
                    # 记录关系信息
                    relation_info = {
                        "relation": relation,
                        "targets": outgoing
                    }
                    entity_expansion["relations"].append(relation_info)
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {str(e)}")
                continue
        
        return entity_expansion if entity_expansion["relations"] else None
    
    def check_if_can_answer(
        self, 
        question: str, 
        start_entities: List[str], 
        exploration_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        检查是否能够基于当前探索回答问题。
        
        Args:
            question: 问题文本
            start_entities: 起始实体列表
            exploration_history: 探索历史
            
        Returns:
            包含答案信息的字典
        """
        try:
            # 格式化探索历史
            formatted_history = self.format_exploration_history(exploration_history)
            
            # 生成推理提示
            prompt = self.templates.REASONING.format(
                question=question,
                entity=", ".join(start_entities),
                num_rounds=len(exploration_history),
                exploration_history=formatted_history
            )
            
            # 生成推理输出
            model_input = self.model.prepare_model_prompt(prompt)
            reasoning_output = self.model.generate_sentence(
                model_input,
                temp_generation_mode="beam",
                num_beams=4
            )
            
            # 解析推理结果
            result = self.parser.parse_reasoning_output(reasoning_output)
            
            # 检查是否需要处理m编码
            if self._process_m_codes_if_needed(result, exploration_history):
                # 如果处理了m编码，重新生成答案
                formatted_history = self.format_exploration_history(exploration_history)
                prompt = self.templates.REASONING.format(
                    question=question,
                    entity=", ".join(start_entities),
                    num_rounds=len(exploration_history),
                    exploration_history=formatted_history
                )
                
                model_input = self.model.prepare_model_prompt(prompt)
                new_reasoning_output = self.model.generate_sentence(
                    model_input,
                    temp_generation_mode="beam",
                    num_beams=4
                )
                
                new_result = self.parser.parse_reasoning_output(new_reasoning_output)
                if "m." not in new_result.get("answer", ""):
                    logger.info("Successfully expanded m-codes to natural language entities")
                    result = new_result
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in check_if_can_answer: {str(e)}")
            return {"can_answer": False}
    
    def _process_m_codes_if_needed(
        self, 
        result: Dict[str, Any], 
        exploration_history: List[Dict[str, Any]]
    ) -> bool:
        """
        处理答案中的m编码（如果存在）。
        
        Args:
            result: 当前推理结果
            exploration_history: 探索历史
            
        Returns:
            如果处理了m编码则返回True，否则返回False
        """
        if not (result.get("can_answer", False) and "m." in result.get("answer", "")):
            return False
        
        # 从答案中提取m编码
        m_codes = re.findall(r'm\.[0-9a-z_]+', result.get("answer", ""))
        if not m_codes:
            return False
        
        logger.info("Found m-code in answer, performing additional exploration...")
        
        # 创建额外的探索轮次
        extra_round = {
            "round": len(exploration_history) + 1,
            "expansions": []
        }
        
        additional_exploration_added = False
        
        # 对每个m编码执行额外的一跳扩展
        for m_code in m_codes:
            # 获取与m编码相连的实体
            outgoing_relations = self.kg.get_related_relations(m_code, "out")
            if not outgoing_relations:
                continue
                
            entity_expansion = {"entity": m_code, "relations": []}
            
            for relation in outgoing_relations:
                targets = self.kg.get_target_entities(m_code, relation, "out")
                if targets:
                    # 记录这次额外扩展
                    relation_info = {
                        "relation": relation,
                        "targets": targets
                    }
                    entity_expansion["relations"].append(relation_info)
                    additional_exploration_added = True
            
            if entity_expansion["relations"]:
                extra_round["expansions"].append(entity_expansion)
        
        # 只有在有实际额外探索时才添加新轮次
        if additional_exploration_added and extra_round["expansions"]:
            # 将额外探索添加到探索历史中
            exploration_history.append(extra_round)
            logger.info(f"Added additional exploration round for m-codes: {m_codes}")
            return True
            
        return False
    
    def explore_knowledge_graph(
        self, 
        question: str, 
        start_entities: Union[str, List[str]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        探索知识图谱以回答问题。
        
        Args:
            question: 问题文本
            start_entities: 起始实体或实体列表
            
        Returns:
            (探索历史, 是否找到答案)的元组
        """
        # if isinstance(start_entities, str):
        #     start_entities = [start_entities]
            
        logger.info(f"Starting exploration with entities: {start_entities}")
        
        exploration_history: List[Dict[str, Any]] = []
        entities_explored: Set[str] = set(start_entities)
        entities_in_context: Set[str] = set(start_entities)
        frontier: List[str] = list(start_entities)
        answer_found = False
        answer_info = {}
        consecutive_unverified = 0  # 连续未验证通过的次数
        max_consecutive_unverified = 2  # 最大允许连续未验证通过的次数
        
        # 主探索循环
        for round_num in range(self.max_rounds):
            logger.info(f"Round {round_num+1}: Exploring {len(frontier)} frontier entities")
            
            # 准备本轮探索数据结构
            round_exploration = {"round": round_num + 1, "expansions": []}
            new_frontier = []
            
            # 生成探索历史和上下文字符串
            exploration_history_str = self.format_exploration_history(exploration_history)
            entities_context_str = ", ".join(entities_in_context)
            
            # 探索所有frontier实体
            for entity in frontier:
                # 展开单个实体
                entity_expansion = self._expand_entity(
                    entity, 
                    question, 
                    entities_context_str, 
                    exploration_history_str
                )
                
                if not entity_expansion:
                    continue
                    
                round_exploration["expansions"].append(entity_expansion)
                
                # 更新上下文和边界
                for relation_info in entity_expansion["relations"]:
                    for target in relation_info["targets"]:
                        if target:
                            entities_in_context.add(target)
                            if target not in entities_explored:
                                new_frontier.append(target)
                                entities_explored.add(target)
            
            # 如果本轮没有新发现，结束探索
            if not round_exploration["expansions"]:
                logger.info("No new information found in this round")
                break
                
            # 添加本轮结果到历史
            exploration_history.append(round_exploration)
            
            # 检查是否能回答问题
            can_answer = self.check_if_can_answer(question, start_entities, exploration_history)
            
            if can_answer.get("can_answer", False):
                # 检查答案格式和验证状态
                is_verified = can_answer.get("is_verified", False)
                has_sentence_structure = can_answer.get("has_sentence_structure", False)
                
                # 如果答案包含句子结构，记录但继续探索
                if has_sentence_structure:
                    logger.warning(
                        f"Answer contains sentence structure: '{can_answer.get('answer', '')}'. "
                        "Continuing exploration to find direct entity answers."
                    )
                    consecutive_unverified += 1
                    
                    # 如果已经到达最大轮次或连续未验证次数过多，使用一个格式化的答案
                    if round_num + 1 >= self.max_rounds or consecutive_unverified >= max_consecutive_unverified:
                        # 尝试从答案中提取实体
                        entities = self._extract_entities_from_sentence(can_answer.get("answer", ""))
                        if entities:
                            logger.info(f"Extracted entities from sentence: {entities}")
                            final_answer = ", ".join(entities)
                            answer_found = True
                            answer_info = {
                                "answer": final_answer,
                                "reasoning_path": can_answer.get("reasoning_path", ""),
                                "verification": "Extracted direct entities from sentence structure",
                                "is_verified": False,
                                "original_answer": can_answer.get("answer", "")
                            }
                            exploration_history[-1]["answer_found"] = answer_info
                            break
                elif is_verified:
                    logger.info(f"Found verified answer after round {round_num+1}, ending exploration")
                    answer_found = True
                    answer_info = {
                        "answer": can_answer.get("answer", ""),
                        "reasoning_path": can_answer.get("reasoning_path", ""),
                        "verification": can_answer.get("verification", ""),
                        "is_verified": True
                    }
                    exploration_history[-1]["answer_found"] = answer_info
                    break
                else:
                    # 答案未通过验证
                    logger.info(f"Found unverified answer after round {round_num+1}, continuing exploration")
                    consecutive_unverified += 1
                    
                    # 如果连续多次未通过验证，使用最后一次的答案
                    if consecutive_unverified >= max_consecutive_unverified:
                        logger.info(f"Reached max consecutive unverified answers ({max_consecutive_unverified}), using last answer")
                        answer_found = True
                        answer_info = {
                            "answer": can_answer.get("answer", ""),
                            "reasoning_path": can_answer.get("reasoning_path", ""),
                            "verification": can_answer.get("verification", ""),
                            "is_verified": False
                        }
                        exploration_history[-1]["answer_found"] = answer_info
                        break
            else:
                # 重置连续未验证计数
                consecutive_unverified = 0
            
            # 更新下一轮的frontier
            frontier = new_frontier
            if not frontier:
                logger.info("No new entities to explore, ending exploration")
                break
        
        return exploration_history, answer_found
    
    def get_fallback_answer(self, question: str) -> Dict[str, str]:
        """
        在没有足够信息时生成动态备用答案。
        
        Args:
            question: 问题文本
            
        Returns:
            包含动态生成的备用答案和推理的字典
        """
        try:
            # 使用模型生成一个更有帮助的回答
            prompt = self.templates.FALLBACK_ANSWER.format(question=question)
            model_input = self.model.prepare_model_prompt(prompt)
            fallback_output = self.model.generate_sentence(
                model_input,
                temp_generation_mode="beam",
                num_beams=4
            )
            
            # 解析动态生成的回答
            result = self.parser.parse_final_answer(fallback_output)
            fallback_answer = {
                "answer": result.get("answer", "I cannot answer this question based on the available knowledge graph."),
                "reasoning": result.get("reasoning", "The knowledge graph does not contain sufficient information to address this query.")
            }
        except Exception as e:
            logger.error(f"Error generating fallback answer: {str(e)}", exc_info=True)
            fallback_answer = {
                "answer": "I cannot answer this question based on the available knowledge graph.",
                "reasoning": "The knowledge graph does not contain sufficient information to address this query."
            }
        
        # 记录无法回答的问题
        self._save_unanswerable_question(question)
        
        return fallback_answer
    
    def _save_unanswerable_question(self, question: str) -> None:
        """
        保存无法回答的问题以供后续分析。
        
        Args:
            question: 无法回答的问题
        """
        log_dir = "logs/unanswerable_questions"
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用日期创建日志文件名
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"unanswerable_{date_str}.log")
        
        # 保存带时间戳的问题
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {question}\n")
        
        logger.info(f"Saved unanswerable question to {log_file}")
    
    def process_question(
        self, 
        data: Dict[str, Any], 
        processed_ids: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        处理单个问题并生成答案。
        
        Args:
            data: 包含问题信息的字典
            processed_ids: 已处理的问题ID列表
            
        Returns:
            包含处理结果的字典，如果问题已处理则返回None
        """
        # 提取问题、真实答案和ID
        question = data.get("question")
        ground_truth = data.get("answer")
        question_id = data.get("id", "unknown")
        
        # if not question:
        #     logger.error("No question found in data")
        #     return None
        
        # 跳过已处理的问题
        if question_id in processed_ids:
            logger.info(f"Question {question_id} already processed, skipping")
            return None
        
        # 获取起始实体
        start_entities = data.get("q_entity", data.get("entity", []))
        
        # 如果没有起始实体，使用备用答案
        if not start_entities:
            logger.error(f"No query entities found for question ID {question_id}")
            fallback = self.get_fallback_answer(question)
            return {
                "id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "start_entities": [],
                "prediction": fallback["answer"],
                "reasoning": fallback["reasoning"],
                "exploration_history": [],
                "answer_found_during_exploration": False,
                "fallback_used": True
            }
        
        logger.info(f"Processing question {question_id}: {question}")
        
        try:
            # 探索知识图谱
            exploration_history, answer_found = self.explore_knowledge_graph(
                question, start_entities
            )
            
            # 如果没有探索历史，使用备用答案
            if not exploration_history:
                logger.warning(f"No exploration history for question ID {question_id}")
                fallback = self.get_fallback_answer(question)
                final_answer = {
                    "answer": fallback["answer"],
                    "reasoning": fallback["reasoning"]
                }
                fallback_used = True
            else:
                fallback_used = False
                
                # 如果在探索过程中找到了答案，使用它
                if answer_found:
                    final_round = exploration_history[-1]
                    answer_info = final_round.get("answer_found", {})
                    answer_text = answer_info.get("answer", "")
                    
                    # 检查是否需要进一步清理答案
                    has_sentence_structure = any(indicator in answer_text.lower() for indicator in [
                        " was ", " is ", " were ", " are ", " has ", " had ", " will ", " would ", 
                        " could ", " can ", " may ", " might ", ". ", " because ", " since ", 
                        " therefore ", " thus ", " hence ", " as a result"
                    ])
                    
                    if has_sentence_structure and "original_answer" not in answer_info:
                        # 尝试从句子中提取实体
                        entities = self._extract_entities_from_sentence(answer_text)
                        if entities:
                            logger.info(f"Additional cleaning of answer with sentence structure: {entities}")
                            answer_text = ", ".join(entities)
                    
                    final_answer = {
                        "answer": answer_text,
                        "reasoning": answer_info.get("reasoning_path", ""),
                        "verification": answer_info.get("verification", ""),
                        "is_verified": answer_info.get("is_verified", False)
                    }
                    
                    # 如果有原始未清理的答案，保存它
                    if "original_answer" in answer_info:
                        final_answer["original_answer"] = answer_info["original_answer"]
                else:
                    # 有探索历史但未找到明确答案，使用check_if_can_answer重新尝试
                    logger.info("No explicit answer found during exploration, attempting one more reasoning attempt")
                    can_answer = self.check_if_can_answer(question, start_entities, exploration_history)
                    
                    if can_answer.get("can_answer", False):
                        is_verified = can_answer.get("is_verified", False)
                        has_sentence_structure = can_answer.get("has_sentence_structure", False)
                        answer_text = can_answer.get("answer", "")
                        
                        # 如果答案包含句子结构，尝试清理
                        if has_sentence_structure:
                            logger.info(f"Final answer contains sentence structure: {answer_text}")
                            entities = self._extract_entities_from_sentence(answer_text)
                            if entities:
                                logger.info(f"Extracted entities: {entities}")
                                answer_text = ", ".join(entities)
                                verification_note = " (Extracted direct entities from original answer)"
                            else:
                                verification_note = " (Note: This answer could not be fully verified with available evidence)"
                        elif not is_verified:
                            verification_note = " (Note: This answer could not be fully verified with available evidence)"
                        else:
                            verification_note = ""
                            
                        final_answer = {
                            "answer": answer_text,
                            "reasoning": can_answer.get("reasoning_path", ""),
                            "verification": can_answer.get("verification", "") + verification_note,
                            "is_verified": is_verified and not has_sentence_structure
                        }
                    else:
                        # 如果仍然无法回答，使用备用答案
                        logger.info("Still cannot answer after additional reasoning, using fallback")
                        fallback = self.get_fallback_answer(question)
                        final_answer = {
                            "answer": fallback["answer"],
                            "reasoning": fallback["reasoning"]
                        }
                        fallback_used = True
            
            # 准备最终结果
            result = {
                "id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "start_entities": start_entities,
                "prediction": final_answer.get("answer", ""),
                "exploration_history": exploration_history,
                "answer_found_during_exploration": answer_found,
                "fallback_used": fallback_used
            }
            
            # 只在需要时添加reasoning和verification
            reasoning = final_answer.get("reasoning")
            if reasoning and not any(
                round_data.get("answer_found", {}).get("reasoning_path") == reasoning
                for round_data in exploration_history
            ):
                result["reasoning"] = reasoning
                
            # 添加验证信息（如果有）
            verification = final_answer.get("verification")
            if verification:
                result["verification"] = verification
                result["is_verified"] = final_answer.get("is_verified", False)
            
            # 如果有原始未处理的答案，添加它以供参考
            if "original_answer" in final_answer:
                result["original_answer"] = final_answer["original_answer"]
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}", exc_info=True)
            # 返回错误信息并使用备用答案
            fallback = self.get_fallback_answer(question)
            return {
                "id": question_id,
                "question": question,
                "start_entities": start_entities,
                "prediction": fallback["answer"],
                "reasoning": f"Error occurred: {str(e)}. {fallback['reasoning']}",
                "error": str(e),
                "fallback_used": True
            }
    
    def _extract_entities_from_sentence(self, sentence: str) -> List[str]:
        """
        从包含句子结构的答案中提取实体。
        
        Args:
            sentence: 包含句子结构的答案文本
            
        Returns:
            提取出的实体列表
        """
        # 简单的分割和清理
        if "," in sentence:
            # 可能是列表中添加了上下文
            parts = [p.strip() for p in sentence.split(",")]
            # 尝试进一步清理描述性词语
            clean_parts = []
            for part in parts:
                # 移除常见的描述性短语
                for phrase in ["was a", "was the", "is a", "is the", "as a", "served as", "worked as"]:
                    if phrase in part.lower():
                        part = part.lower().split(phrase, 1)[1].strip()
                clean_parts.append(part.strip())
            return clean_parts
        elif " and " in sentence.lower():
            # 处理用"and"连接的实体
            parts = [p.strip() for p in sentence.split(" and ")]
            # 处理第一部分可能包含的描述
            if "was" in parts[0].lower() or "is" in parts[0].lower():
                first_part = parts[0].lower()
                for phrase in ["was a", "was the", "is a", "is the"]:
                    if phrase in first_part:
                        parts[0] = first_part.split(phrase, 1)[1].strip()
                        break
            return parts
        else:
            # 单个实体，可能带有描述
            for phrase in ["was a", "was the", "is a", "is the", "as a", "served as", "worked as"]:
                if phrase in sentence.lower():
                    return [sentence.lower().split(phrase, 1)[1].strip()]
            
            # 如果没有明确的描述词，尝试提取主语
            if " was " in sentence:
                return [sentence.split(" was ")[0].strip()]
            elif " is " in sentence:
                return [sentence.split(" is ")[0].strip()]
                
        # 如果上述方法都失败，返回原始句子
        return [sentence]