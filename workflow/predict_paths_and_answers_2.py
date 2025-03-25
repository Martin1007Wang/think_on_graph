import os
import argparse
from tqdm import tqdm
import json
import logging
from functools import partial
from multiprocessing import Pool
from datasets import load_dataset
from src.llms import get_registed_model
from src.utils.qa_utils import eval_path_result_w_ans
from src.template import Template
from src.knowledge_graph import KnowledgeGraph

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置日志
def setup_logging(name="kg_qa", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# 实体和关系检索提示模板
RELATION_SELECTION_PROMPT = """
You are a knowledge graph reasoning expert. Given a question and a topic entity, your task is to select the most relevant relations to explore.

# Question: 
{question}

# Topic entity: 
{entity}

# Available relations from this entity:
{relations}

Select the top {k} most relevant relations that would help answer this question. For each relation, explain why it is relevant.
Format your response as:
1. [relation_name] - [brief explanation]
2. [relation_name] - [brief explanation]
...

Only list relations that are directly relevant to the question. Focus on relations that would lead to answer entities.
"""

REASONING_PROMPT = """
You are a knowledge graph reasoning expert. Given a question and information gathered from a knowledge graph, determine if you can answer the question.

# Question: 
{question}

# Starting entities: 
{entity}

# Knowledge graph exploration (over {num_rounds} rounds):
{exploration_history}

Can you answer the question based on the information above? If yes, provide the answer with the reasoning path. If no, explain what additional information you need.

If you can answer the question:
Answer: [your answer]
Reasoning path: [specify the exact path of relations and entities that led to this answer]

If you cannot answer yet:
Status: Need more information
Missing information: [specify what additional relations or entities you need to explore]
"""

FINAL_ANSWER_PROMPT = """
You are a knowledge graph reasoning expert. Based on the exploration of the knowledge graph, please provide a final answer to the question.

# Question: 
{question}

# Starting entities: 
{entity}

# Complete exploration:
{full_exploration}

# Final answer:
"""

def get_output_file(path, force=False):
    """获取输出文件和已处理列表"""
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def parse_selected_relations(llm_output, available_relations):
    """解析LLM输出，提取选择的关系"""
    selected_relations = []
    
    # 拆分输出到行
    lines = llm_output.strip().split('\n')
    
    # 查找编号行（如 "1. relation_name - explanation"）
    for line in lines:
        line = line.strip()
        # 跳过空行
        if not line:
            continue
            
        # 匹配编号格式
        if line[0].isdigit() and '. ' in line:
            # 提取关系名称 (在 '. ' 之后和 ' - ' 之前)
            parts = line.split('. ', 1)[1].split(' - ', 1)
            if len(parts) > 0:
                relation = parts[0].strip()
                # 验证关系名称确实在可用关系中
                matching_relations = [r for r in available_relations if r.lower() == relation.lower()]
                if matching_relations:
                    selected_relations.append(matching_relations[0])  # 使用原始格式
                    
    # 如果没有找到有效关系，尝试直接从文本中提取
    if not selected_relations:
        for rel in available_relations:
            if rel in llm_output:
                selected_relations.append(rel)
                
    return selected_relations

def get_related_relations(kg, entity, question, model, k=5):
    """获取实体相关的关系并让模型选择最相关的k个"""
    # 从Neo4j直接获取与该实体相关的所有关系
    related_relations = kg.get_related_relations(entity)
    if not related_relations:
        logger.warning(f"No relations found for entity '{entity}'")
        return []
        
    # 如果关系数量小于等于k，直接返回所有关系
    if len(related_relations) <= k:
        return related_relations
        
    # 构建提示，让模型选择最相关的关系
    prompt = RELATION_SELECTION_PROMPT.format(
        question=question,
        entity=entity,
        relations="\n".join(f"- {rel}" for rel in related_relations),
        k=min(k, len(related_relations))
    )
    
    # 准备模型输入并生成选择
    model_input = model.prepare_model_prompt(prompt)
    selection_output = model.generate_sentence(model_input)
    
    # 解析输出，获取选择的关系
    selected_relations = parse_selected_relations(selection_output, related_relations)
    
    # 限制结果数量
    return selected_relations[:k]

def explore_knowledge_graph(kg, question, start_entities, model, max_rounds=3, k=5):
    """探索知识图谱进行多轮推理，支持多个起始实体
    
    参数:
        kg: 知识图谱连接器
        question: 问题文本
        start_entities: 起始实体列表 (List[str])
        model: 语言模型
        max_rounds: 最大探索轮数
        k: 每个实体选择的关系数量
    """
    # 确保start_entities是列表
    if isinstance(start_entities, str):
        start_entities = [start_entities]
    
    logger.info(f"Starting exploration with entities: {start_entities}")
    
    exploration_history = []
    entities_explored = {}  # 跟踪已探索的实体
    frontier = []  # 当前探索前沿
    
    # 初始化已探索实体和前沿
    for entity in start_entities:
        if entity and entity not in entities_explored:
            entities_explored[entity] = True
            frontier.append(entity)
    
    for round_num in range(max_rounds):
        logger.info(f"Round {round_num+1}: Exploring {len(frontier)} frontier entities")
        
        round_exploration = {
            "round": round_num + 1,
            "expansions": []
        }
        
        new_frontier = []  # 新的探索前沿
        
        # 对每个前沿实体进行扩展
        for entity in frontier:
            # 获取实体相关的关系
            selected_relations = get_related_relations(kg, entity, question, model, k)
            
            if not selected_relations:
                logger.info(f"No relations found for entity: {entity}")
                continue
                
            logger.info(f"Selected {len(selected_relations)} relations for entity {entity}")
            
            entity_expansion = {
                "entity": entity,
                "relations": []
            }
            
            # 对每个选定的关系，获取目标实体
            for relation in selected_relations:
                # 获取该关系连接的目标实体
                target_entities_out = kg.get_target_entities(entity, relation, "out")
                target_entities_in = kg.get_target_entities(entity, relation, "in")
                
                relation_info = {
                    "relation": relation,
                    "outgoing_targets": target_entities_out[:5],  # 限制返回数量
                    "incoming_targets": target_entities_in[:5]    # 限制返回数量
                }
                
                entity_expansion["relations"].append(relation_info)
                
                # 将新实体添加到下一轮探索前沿
                for target in target_entities_out + target_entities_in:
                    if target and target not in entities_explored:
                        new_frontier.append(target)
                        entities_explored[target] = True
            
            # 只有当实体有关系时才添加到扩展中
            if entity_expansion["relations"]:
                round_exploration["expansions"].append(entity_expansion)
        
        # 只有当有扩展时才添加到历史记录中
        if round_exploration["expansions"]:
            exploration_history.append(round_exploration)
        
        # 如果没有新的前沿实体，结束探索
        if not new_frontier:
            logger.info("No new entities to explore, ending exploration")
            break
            
        frontier = new_frontier
        
        # 检查是否可以根据当前信息回答问题
        if round_num < max_rounds - 1:  # 除了最后一轮，每轮都检查
            can_answer = check_if_can_answer(question, start_entities, exploration_history, model)
            if can_answer["can_answer"]:
                logger.info("Found answer, ending exploration")
                break
    
    return exploration_history

def format_exploration_history(exploration_history):
    """将探索历史格式化为可读文本"""
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
        
        result.append("")  # Empty line between rounds
    
    return "\n".join(result)

def check_if_can_answer(question, start_entities, exploration_history, model):
    """检查是否可以根据当前信息回答问题"""
    try:
        formatted_history = format_exploration_history(exploration_history)
        
        # 将实体列表格式化为字符串
        entities_str = ", ".join(start_entities)
        
        prompt = REASONING_PROMPT.format(
            question=question,
            entity=entities_str,
            num_rounds=len(exploration_history),
            exploration_history=formatted_history
        )
        
        model_input = model.prepare_model_prompt(prompt)
        reasoning_output = model.generate_sentence(model_input)
        
        # 解析输出
        can_answer = "answer:" in reasoning_output.lower() and "reasoning path:" in reasoning_output.lower()
        
        if can_answer:
            # 提取答案和推理路径
            answer_text = reasoning_output.lower().split("answer:", 1)[1].split("\n", 1)[0].strip()
            path_text = ""
            if "reasoning path:" in reasoning_output.lower():
                path_text = reasoning_output.lower().split("reasoning path:", 1)[1].strip()
                
            return {
                "can_answer": True,
                "answer": answer_text,
                "reasoning_path": path_text
            }
        else:
            return {
                "can_answer": False
            }
    except Exception as e:
        logger.warning(f"Error in check_if_can_answer: {str(e)}")
        return {"can_answer": False}

def generate_final_answer(question, start_entities, exploration_history, model):
    """生成最终答案"""
    formatted_history = format_exploration_history(exploration_history)
    
    # 将实体列表格式化为字符串
    entities_str = ", ".join(start_entities)
    
    prompt = FINAL_ANSWER_PROMPT.format(
        question=question,
        entity=entities_str,
        full_exploration=formatted_history
    )
    
    model_input = model.prepare_model_prompt(prompt)
    final_answer = model.generate_sentence(model_input)
    
    return final_answer

def iterative_reasoning(data, processed_list, model, kg, args):
    """使用迭代推理方法回答问题"""
    question = data["question"]
    id = data["id"]
    
    if id in processed_list:
        return None
    
    # 获取起始实体列表
    if "q_entity" in data:
        start_entities = data["q_entity"]
    elif "entity" in data:
        start_entities = data.get("entity", [])
    else:
        start_entities = []
    
    # 确保start_entities是非空列表
    if not start_entities:
        logger.warning(f"No query entities found for question ID {id}")
        return None
        
    logger.info(f"Processing question {id} with entities: {start_entities}")
    
    try:
        # 多轮探索知识图谱
        exploration_history = explore_knowledge_graph(
            kg, 
            question, 
            start_entities, 
            model, 
            max_rounds=args.max_rounds, 
            k=args.top_k_relations
        )
        
        if not exploration_history:
            logger.warning(f"No exploration history for question ID {id}")
            return None
            
        # 生成最终答案
        final_answer = generate_final_answer(question, start_entities, exploration_history, model)
        
        # 构建结果
        result = {
            "id": id,
            "question": question,
            "start_entities": start_entities,
            "prediction": final_answer,
            "exploration_history": exploration_history,
            "formatted_exploration": format_exploration_history(exploration_history)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing question {id}: {str(e)}", exc_info=True)
        return None

def main(args):
    # 初始化知识图谱连接
    kg = KnowledgeGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        model_name=args.embedding_model
    )
    
    try:
        # 加载数据集
        input_file = os.path.join(args.data_path, args.d)
        dataset = load_dataset(input_file, split=args.split)
        
        # 构建输出路径
        post_fix = f"{args.prefix}iterative-rounds{args.max_rounds}-topk{args.top_k_relations}"
        data_name = args.d + "_undirected" if args.undirected else args.d
        output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
        
        logger.info(f"Save results to: {output_dir}")
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 初始化模型
        LLM = get_registed_model(args.model_name)
        model = LLM(args)
        
        logger.info("Preparing pipeline for inference...")
        model.prepare_for_inference()
        
        # 保存参数
        with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
        # 获取输出文件和已处理列表
        fout, processed_list = get_output_file(os.path.join(output_dir, 'predictions.jsonl'), force=args.force)
        
        # 单进程或多进程预测
        if args.n > 1:
            with Pool(args.n) as p:
                for res in tqdm(
                    p.imap(
                        partial(
                            iterative_reasoning,
                            processed_list=processed_list,
                            model=model,
                            kg=kg,
                            args=args
                        ),
                        dataset,
                    ),
                    total=len(dataset),
                ):
                    if res is not None:
                        if args.debug:
                            print(json.dumps(res))
                        fout.write(json.dumps(res) + "\n")
                        fout.flush()
        else:
            for data in tqdm(dataset):
                res = iterative_reasoning(
                    data, 
                    processed_list, 
                    model, 
                    kg,
                    args
                )
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
                else:
                    logger.warning(f"None result for: {data.get('id', 'unknown')}")
                    
        fout.close()
                
        # 评估结果
        eval_path_result_w_ans(os.path.join(output_dir, 'predictions.jsonl'))
        
    finally:
        # 关闭Neo4j连接
        kg.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集参数
    parser.add_argument('--data_path', type=str, default='rmanluo')
    parser.add_argument('--d', '-d', type=str, default='RoG-webqsp')
    parser.add_argument('--split', type=str, default='test[:100]')
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, help="model_name for save results", default='gcr-Llama-2-7b-chat-hf')
    parser.add_argument('--model_path', type=str, help="Path to the model weights", default=None)
    
    # 推理参数
    parser.add_argument('--max_rounds', type=int, default=3, help="Maximum number of reasoning rounds")
    parser.add_argument('--top_k_relations', type=int, default=5, help="Number of top relations to select per entity")
    
    # Neo4j参数
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang')
    parser.add_argument('--embedding_model', type=str, default='msmarco-distilbert-base-tas-b', 
                         help="Sentence transformer model for computing embeddings")
    
    # 实用参数
    parser.add_argument('--force', action='store_true', help="force to overwrite the results")
    parser.add_argument('--n', type=int, default=1, help="number of processes")
    parser.add_argument('--undirected', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--debug', action='store_true', help="print debug information")
    parser.add_argument('--prefix', type=str, default="")
    
    args, _ = parser.parse_known_args()
    
    # 获取模型注册信息并添加模型特定参数
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    
    args = parser.parse_args()
    
    main(args)