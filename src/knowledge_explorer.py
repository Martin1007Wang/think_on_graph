"""
知识图谱探索器模块，用于基于知识图谱回答问题。
"""
import logging
import re
import os
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import concurrent.futures
from collections import OrderedDict

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
    
    def __init__(self, kg: KnowledgeGraph, model: Any, max_rounds: int = 3, max_k_relations: int = 10) -> None:
        self.kg = kg
        self.model = model
        self.max_rounds = max_rounds
        self.max_k_relations = max_k_relations
        self.parser = LLMOutputParser()
        self.templates = KnowledgeGraphTemplates()
    
    def format_exploration_round(self, round_data: Dict[str, Any]) -> str:
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
        result = []
        for round_data in exploration_history:
            result.append(self.format_exploration_round(round_data))
        return "\n".join(result).rstrip()
    
    def _get_related_relations(self, entity: str, question: str, context: str = "", history: str = "") -> List[str]:
        out_related_relations = self.kg.get_related_relations(entity, "out")
        if not out_related_relations:
            logger.error(f"No relations found for entity '{entity}'")
            return []
        relation_dict = {f"REL_{i}": rel for i, rel in enumerate(out_related_relations)}
        relation_options = "\n".join(f"[{rel_id}] {rel}" for rel_id, rel in relation_dict.items())
        template = (
            self.templates.RELATION_SELECTION_WITH_CONTEXT 
            if context and history 
            else self.templates.RELATION_SELECTION
        )
        prompt_args = {
            "question": question,
            "entity": entity,
            "relations": relation_options,
            "relation_ids": ", ".join(relation_dict.keys()),
            "max_k_relations": min(self.max_k_relations, len(out_related_relations))
        }
        if context and history:
            prompt_args.update({"history": history, "context": context})
        prompt = template.format(**prompt_args)
        model_input = self.model.prepare_model_prompt(prompt)
        selection_output = self.model.generate_sentence(model_input,temp_generation_mode="beam")
        rel_ids = re.findall(r'REL_\d+', selection_output)
        selected_relations = list(OrderedDict.fromkeys(relation_dict[rel_id] for rel_id in rel_ids))
        return selected_relations
    
    def _expand_entity(self, entity: str, question: str, context: str, history: str) -> Dict[str, Any]:
        selected_relations = self._get_related_relations(entity, question, context, history)
        entity_expansion = {"entity": entity, "relations": []}
        
        for rel in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, rel, "out")
                if targets:
                    relation_info = {"relation": rel, "targets": targets}
                    entity_expansion["relations"].append(relation_info)
            except Exception as e:
                logger.error(f"Error retrieving targets for '{entity}' with relation '{rel}': {str(e)}")
        
        return entity_expansion if entity_expansion["relations"] else None
    
    def check_if_can_answer(self, question: str, start_entities: List[str], exploration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_history = self.format_exploration_history(exploration_history)
        try:
            prompt = self.templates.REASONING.format(
                question=question,
                entity=", ".join(start_entities),
                num_rounds=len(exploration_history),
                exploration_history=updated_history
            )
            model_input = self.model.prepare_model_prompt(prompt)
            reasoning_output = self.model.generate_sentence(
                model_input,
                temp_generation_mode="beam",
                num_beams=4
            )
            
            result = self.parser.parse_reasoning_output(reasoning_output)
            
            if self._process_m_codes_if_needed(result, exploration_history):
                updated_history = self.format_exploration_history(exploration_history)
                prompt = self.templates.REASONING.format(
                    question=question,
                    entity=", ".join(start_entities),
                    num_rounds=len(exploration_history),
                    exploration_history=updated_history
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
    
    def explore_knowledge_graph(self, question: str, start_entities: Union[str, List[str]]) -> Tuple[List[Dict[str, Any]], bool]:
        exploration_history: List[Dict[str, Any]] = []
        entities_explored: Set[str] = set(start_entities)
        entities_in_context: Set[str] = set(start_entities)
        frontier: List[str] = list(start_entities)
        answer_found = False
        
        for round_num in range(self.max_rounds):
            round_exploration = {"round": round_num + 1, "expansions": []}
            new_frontier = []
            entities_context = ", ".join(entities_in_context)
            entity_expansions = {}
            for entity in frontier:
                try:
                    expansion = self._expand_entity(entity, question, entities_context, self.format_exploration_history(exploration_history))
                    if expansion:
                        entity_expansions[entity] = expansion
                except Exception as e:
                    logger.error(f"Error expanding entity {entity}: {str(e)}")
                    entity_expansions[entity] = None
            for entity, expansion in entity_expansions.items():
                round_exploration["expansions"].append(expansion)
                for relation_info in expansion["relations"]:
                    for target in relation_info["targets"]:
                        entities_in_context.add(target)
                        if target not in entities_explored:
                            new_frontier.append(target)
                            entities_explored.add(target)
            
            exploration_history.append(round_exploration)
            can_answer = self.check_if_can_answer(question, start_entities, exploration_history)
            
            # 如果能回答问题，结束探索
            if can_answer.get("can_answer", False):
                logger.info(f"Found answer after round {round_num+1}, ending exploration")
                answer_found = True
                
                # 记录答案信息
                answer_info = {
                    "answer": can_answer.get("answer", ""),
                    "reasoning_path": can_answer.get("reasoning_path", ""),
                    "verification": can_answer.get("verification", ""),
                    "is_verified": can_answer.get("is_verified", True)  # 假设输出可信
                }
                exploration_history[-1]["answer_found"] = answer_info
                break
            
            # 更新下一轮的frontier
            frontier = new_frontier
        
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
        question = data.get("question")
        ground_truth = data.get("answer")
        question_id = data.get("id", "unknown")
        if question_id in processed_ids:
            logger.info(f"Question {question_id} already processed, skipping")
            return None
        start_entities = data.get("q_entity", data.get("entity", []))
        try:
            exploration_history, answer_found = self.explore_knowledge_graph(question, start_entities)
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
                    
                    final_answer = {
                        "answer": answer_text,
                        "reasoning": answer_info.get("reasoning_path", ""),
                        "verification": answer_info.get("verification", ""),
                        "is_verified": answer_info.get("is_verified", False)
                    }
                    
                   
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