from typing import Dict, Any, Optional

class KnowledgeGraphTemplates:
    # === 关系选择与实体排序模板 ===
    
    RELATION_SELECTION = """
    You are a knowledge graph reasoning expert. Given a question and a topic entity, your task is to select the most relevant relations to explore from the provided list.

    # Question: 
    {question}

    # Topic entity: 
    {entity}

    # Available relations from this entity (select only from these):
    {relations}

    Select exactly {relation_k} relations that are most relevant to answering the question. Your response must follow this exact format, with no additional text outside the numbered list:
    1. [relation_name] - [brief explanation of relevance]
    2. [relation_name] - [brief explanation of relevance]
    ...
    {relation_k}. [relation_name] - [brief explanation of relevance]

    - Only choose relations from the provided list.
    - If fewer than {relation_k} relations are relevant, repeat the most relevant relation to fill the list.
    - Do not include any introductory text, conclusions, or extra lines beyond the {relation_k} numbered items.
    """

    ENTITY_RANKING = """
    You are a knowledge graph reasoning expe rt. Given a question and a set of already explored entities, rank the candidate entities by their relevance to answering the question.
    
    # Question: 
    {question}
    
    # Already explored entities: 
    {explored}
    
    # Candidate entities to evaluate:
    {candidates}
    
    For each candidate entity, assign a relevance score from 1-10 (10 being most relevant) based on:
    1. Direct relevance to the question
    2. Potential to connect to relevant information
    3. Uniqueness compared to already explored entities
    
    Format your response as:
    [Entity]: [Score] - [Brief justification]
    """

    RELATION_SELECTION_WITH_CONTEXT = """
    You are a knowledge graph reasoning expert. Given a question, a topic entity, and the exploration history so far, select the most promising relations to explore next.
    
    # Question: 
    {question}
    
    # Current entity to expand: 
    {entity}
    
    # Exploration history so far:
    {history}
    
    # Available relations from this entity (select only from these):
    {relations}
    
    Select exactly {relation_k} relations that are most likely to lead to the answer. Your response must follow this exact format:
    1. [relation_name] - [brief explanation of relevance]
    2. [relation_name] - [brief explanation of relevance]
    ...
    
    Consider:
    - Which relations might connect to information needed to answer the question
    - Avoid relations that would lead to already explored paths
    - Prioritize relations that fill gaps in the current knowledge
    """

    # === 推理与决策模板 ===

    REASONING = """
    You are a knowledge graph reasoning expert. Given a question and information gathered from a knowledge graph, determine if you can answer the question.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Knowledge graph exploration (over {num_rounds} rounds):
    {exploration_history}

    Based on the information above, can you answer the question? Respond in this exact format, with no additional text outside the specified sections:

    [Decision: Yes/No]
    [Answer: your answer if Yes, otherwise leave blank]
    [Missing information: specify what additional relations or entities are needed if No, otherwise leave blank]
    """

    FINAL_ANSWER = """
    You are a knowledge graph reasoning expert. Based on the exploration of the knowledge graph, provide a final answer to the question.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Complete exploration:
    {full_exploration}

    Provide your final answer in this exact format, with no additional text outside the specified sections:
    [Final Answer: your concise answer to the question]
    [Reasoning: brief explanation of how the answer was derived from the exploration]
    """

    # === 零样本和引导提示模板 ===
    
    ZERO_SHOT_PROMPT = """
    You are a knowledge graph reasoning expert. Analyze reasoning paths connecting the topic entity to potential answers.

    # Question: 
    {question}

    # Topic entity: 
    {entity}

    Consider: 
    1) Semantic alignment with question intent
    2) Path efficiency 
    3) Relation relevance
    4) Factual correctness

    Your task is to identify the most promising reasoning paths that could lead to the answer.
    """
    
    # 语义路径模板：强调语义相关性，分析路径优势
    SEMANTIC_PATH_TEMPLATE = (
        "# Semantic Reasoning Path:\n{reasoning_path}\n\n"
        "# Path Analysis:\nHIGH SEMANTIC RELEVANCE - Strong alignment with question intent.\n"
        "Strengths: Semantic alignment, relevant relations, intuitive reasoning\n\n"
        "# Answer:\n{answer}"
    )

    # 最短路径模板：强调路径效率，突出简洁性
    SHORTEST_PATH_TEMPLATE = (
        "# Shortest Reasoning Path:\n{reasoning_path}\n\n"
        "# Path Analysis:\nMAXIMUM EFFICIENCY - Most direct connection in the graph.\n"
        "Strengths: Direct path, minimal hops, structurally optimal\n\n"
        "# Answer:\n{answer}"
    )

    # 负面路径模板：指出路径缺陷，分析无关性
    NEGATIVE_PATH_TEMPLATE = (
        "# Problematic Reasoning Path:\n{reasoning_path}\n\n"
        "# Path Analysis:\nLOW RELEVANCE - Poor alignment with question intent.\n"
        "Issues: Poor semantic alignment, irrelevant relations, illogical reasoning\n\n"
        "# Answer:\n{answer}"
    )


    # === 路径分析与评估模板 ===

    PATH_EVALUATION = """
    You are a knowledge graph reasoning expert. Evaluate the following reasoning path for answering the question.
    
    # Question:
    {question}
    
    # Path:
    {path}
    
    Provide a comprehensive assessment using the following metrics:
    
    [Semantic Relevance: 1-10] - How well the path aligns with the question's meaning
    [Path Efficiency: 1-10] - How direct and concise the path is
    [Factual Accuracy: 1-10] - How factually correct the path relationships are
    [Overall Quality: 1-10] - Overall assessment considering all factors
    
    [Analysis: detailed explanation of strengths and weaknesses]
    [Conclusion: whether this path effectively answers the question]
    """

    PATH_COMPARISON = """
    You are a knowledge graph reasoning expert. Compare multiple reasoning paths for answering the question.
    
    # Question:
    {question}
    
    # Path 1:
    {path1}
    
    # Path 2:
    {path2}
    
    Evaluate each path on:
    - Semantic relevance to the question
    - Path efficiency and directness
    - Factual accuracy of relationships
    - Overall quality for answering the question
    
    Then recommend the superior path with justification.
    """

    # === 路径生成模板 ===

    PATH_GENERATION = """
    You are a knowledge graph reasoning expert. Generate reasoning paths from the topic entities to answer the question.

    # Question: 
    {question}

    # Topic entities: 
    {entities}

    Generate {num_paths} distinct reasoning paths that could lead to the answer. For each path:
    1. Start from one of the topic entities
    2. Follow relevant relations in the knowledge graph
    3. Reach a potential answer entity
    
    Format each path as:
    [Path {i}]: entity1 --[relation1]--> entity2 --[relation2]--> ... --[relationN]--> answerEntity
    
    Prioritize paths that are semantically relevant to the question and factually accurate.
    """

    # === 多选题专用模板 ===

    MCQ_PATH_GENERATION = """
    You are a knowledge graph reasoning expert. Generate reasoning paths to determine the correct answer choice.

    # Question: 
    {question}

    # Topic entities: 
    {entities}

    # Answer Choices:
    {choices}

    Generate reasoning paths from the topic entities that could lead to each potential answer choice.
    Then analyze which path best answers the question.
    
    For each choice, provide:
    [Choice]: the answer option
    [Path]: the reasoning path from topic entity to this choice
    [Analysis]: assessment of path quality and relevance
    
    Conclude with the most likely correct answer based on path quality.
    """

    def __init__(self):
        """初始化模板类，添加模板注册表和格式化方法"""
        # 模板注册表，方便按名称获取模板
        self._template_registry = {
            "relation_selection": self.RELATION_SELECTION,
            "entity_ranking": self.ENTITY_RANKING,
            "relation_selection_context": self.RELATION_SELECTION_WITH_CONTEXT,
            "reasoning": self.REASONING,
            "final_answer": self.FINAL_ANSWER,
            "zero_shot": self.ZERO_SHOT_PROMPT,
            "path_evaluation": self.PATH_EVALUATION,
            "path_comparison": self.PATH_COMPARISON,
            "path_generation": self.PATH_GENERATION,
            "mcq_path_generation": self.MCQ_PATH_GENERATION
        }
    
    def get_template(self, template_name: str) -> Optional[str]:
        """根据名称获取指定的模板。
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板字符串，如果不存在则返回None
        """
        return self._template_registry.get(template_name.lower())
    
    def format_template(self, template_name: str, **kwargs) -> Optional[str]:
        """根据提供的参数格式化指定的模板。
        
        Args:
            template_name: 模板名称
            **kwargs: 格式化参数
            
        Returns:
            格式化后的模板字符串，如果模板不存在则返回None
        """
        template = self.get_template(template_name)
        if template is None:
            return None
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template '{template_name}': {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}")