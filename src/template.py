from typing import Dict, Any, Optional, List, Union, Protocol
from enum import Enum
from dataclasses import dataclass

class TemplateCategory(Enum):
    """枚举定义不同类别的提示模板"""
    RELATION = "relation"
    REASONING = "reasoning"
    PATH = "path"
    EVALUATION = "evaluation"
    ZERO_SHOT = "zero_shot"
    MCQ = "mcq"

@dataclass
class PromptTemplate:
    """提示模板数据类，存储模板及其元数据"""
    name: str
    template: str
    category: TemplateCategory
    required_params: List[str]
    description: str = ""

class PromptFormatter(Protocol):
    """定义格式化提示的协议接口"""
    def __call__(self, template: str, **kwargs) -> str:
        """格式化提示模板的方法"""
        ...

class KnowledgeGraphTemplates:
    """知识图谱推理提示模板管理类"""
    
    # 核心提示模板定义
    RELATION_SELECTION = """
    You are a knowledge graph reasoning expert. Given a question and a topic entity, your task is to select the most relevant relations to explore from the provided list.

    # Question: 
    {question}

    # Topic entity: 
    {entity}

    # Available relations from this entity (select only from these):
    {relations}

    Select up to {max_k_relations} relation IDs that are most relevant to answering the question.
    Your response should ONLY contain the relation IDs (e.g., REL_0, REL_1) from the list above.
    Valid relation IDs: {relation_ids}
    Your selection (IDs only):
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
    
    Select up to {max_k_relations} relation IDs that are most relevant to answering the question.
    Your response should ONLY contain the relation IDs (e.g., REL_0, REL_1) from the list above.
    Valid relation IDs: {relation_ids}
    Your selection (IDs only):
    """
    
    ENTITY_RANKING = """
    You are a knowledge graph reasoning expert. Given a question and a set of already explored entities, rank the candidate entities by their relevance to answering the question.
    
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

    # 推理与决策模板
    REASONING = """
    You are a knowledge graph reasoning expert. Given a question and information from a knowledge graph, determine if you can answer strictly based on the provided triples.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Knowledge graph exploration (over {num_rounds} rounds):
    {exploration_history}

    CRITICAL RULES:
    1. ONLY use EXPLICIT triples shown as Entity--[Relation]-->Target from the exploration.
    2. NO inference, synthesis, or combining unconnected information.
    3. Answer MUST ONLY contain exact entity names from triples, comma-separated.
    4. NEVER create sentences or narratives - ONLY list entity names.
    5. If multiple valid answers exist, list ALL without modification.
    
    Respond in this exact format:
    Decision: Yes/No
    Reasoning path: if Yes, step-by-step entity-relation-entity chains
    Preliminary Answer: ONLY entity names that directly answer the question, comma-separated
    Verification: Yes/No - whether the answer follows directly from reasoning path without assumptions
    Final Answer: After verification, ONLY entity names, comma-separated, or state cannot answer
    """

    # 零样本和引导提示模板
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
    
    # 路径模板，使用元组定义以提高可读性
    SEMANTIC_PATH_TEMPLATE = (
        "# Semantic Reasoning Path:\n{reasoning_path}\n\n"
        "# Path Analysis:\nHIGH SEMANTIC RELEVANCE - Strong alignment with question intent.\n"
        "Strengths: Semantic alignment, relevant relations, intuitive reasoning\n\n"
        "# Answer:\n{answer}"
    )

    SHORTEST_PATH_TEMPLATE = (
        "# Shortest Reasoning Path:\n{reasoning_path}\n\n"
        "# Path Analysis:\nMAXIMUM EFFICIENCY - Most direct connection in the graph.\n"
        "Strengths: Direct path, minimal hops, structurally optimal\n\n"
        "# Answer:\n{answer}"
    )

    NEGATIVE_PATH_TEMPLATE = (
        "# Problematic Reasoning Path:\n{reasoning_path}\n\n"
        "# Path Analysis:\nLOW RELEVANCE - Poor alignment with question intent.\n"
        "Issues: Poor semantic alignment, irrelevant relations, illogical reasoning\n\n"
        "# Answer:\n{answer}"
    )

    # 路径分析与评估模板
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

    # 路径生成模板
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

    # 多选题专用模板
    MCQ_PATH_GENERATION = """
    You are a knowledge graph reasoning expert. Generate reasoning paths to determine the correct answer choice.

    # Question: 
    {question}

    # Topic entities: 
    {entities}

    # Answer Choices:
    {choices}
    
    For each answer choice, generate a possible reasoning path from the topic entities that would lead to that choice. Format as:

    [Choice A]: entity1 --[relation1]--> entity2 --[relation2]--> ... entityN
    [Analysis]: Brief explanation of why this path is or isn't valid

    Repeat for all choices, then conclude with:
    [Most likely answer]: Indicate which choice has the most supportable path
    [Reasoning]: Justification for your selection
    """

    # 最终答案模板 
    FINAL_ANSWER = """
    You are a knowledge graph reasoning expert. Given a question and the full exploration history of a knowledge graph, provide your best answer.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Full knowledge graph exploration:
    {full_exploration}

    Based on the exploration above, provide your best answer to the question. 
    Answer the question directly and concisely, using only information from the exploration.
    If the exploration doesn't contain enough information to answer confidently, explain what's missing.

    Respond in this exact format:
    Answer: your answer based strictly on information in the exploration
    Reasoning: step-by-step explanation of how you arrived at the answer
    """
    
    # 备用答案模板
    FALLBACK_ANSWER = """
    You are a knowledge graph reasoning expert. The following question couldn't be answered with the available knowledge graph information.

    # Question: 
    {question}

    Provide a helpful response explaining why this type of question might be difficult to answer with a knowledge graph.
    Consider what information would be needed to answer it properly.

    Respond in this exact format:
    Answer: concise statement that the question cannot be answered with available information
    Reasoning: explanation of what information would be needed to answer properly
    """

    def __init__(self):
        """初始化模板类，构建模板注册表"""
        # 创建更结构化的模板注册表
        self._templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """初始化并注册所有模板"""
        templates = [
            PromptTemplate(
                name="relation_selection",
                template=self.RELATION_SELECTION,
                category=TemplateCategory.RELATION,
                required_params=["question", "entity", "relations", "max_k_relations"],
                description="选择最相关的关系进行探索"
            ),
            PromptTemplate(
                name="relation_selection_context",
                template=self.RELATION_SELECTION_WITH_CONTEXT,
                category=TemplateCategory.RELATION,
                required_params=["question", "entity", "history", "relations", "max_k_relations"],
                description="基于历史上下文选择最相关的关系"
            ),
            PromptTemplate(
                name="entity_ranking",
                template=self.ENTITY_RANKING,
                category=TemplateCategory.RELATION,
                required_params=["question", "explored", "candidates"],
                description="对候选实体进行相关性排序"
            ),
            PromptTemplate(
                name="reasoning",
                template=self.REASONING,
                category=TemplateCategory.REASONING,
                required_params=["question", "entity", "exploration_history", "num_rounds"],
                description="基于知识图谱探索进行推理判断"
            ),
            PromptTemplate(
                name="final_answer",
                template=self.FINAL_ANSWER,
                category=TemplateCategory.REASONING,
                required_params=["question", "entity", "full_exploration"],
                description="提供最终答案和推理路径"
            ),
            PromptTemplate(
                name="zero_shot",
                template=self.ZERO_SHOT_PROMPT,
                category=TemplateCategory.ZERO_SHOT,
                required_params=["question", "entity"],
                description="零样本推理路径分析"
            ),
            PromptTemplate(
                name="semantic_path",
                template=self.SEMANTIC_PATH_TEMPLATE,
                category=TemplateCategory.PATH,
                required_params=["reasoning_path", "answer"],
                description="语义相关的推理路径模板"
            ),
            PromptTemplate(
                name="shortest_path",
                template=self.SHORTEST_PATH_TEMPLATE,
                category=TemplateCategory.PATH,
                required_params=["reasoning_path", "answer"],
                description="最短推理路径模板"
            ),
            PromptTemplate(
                name="negative_path",
                template=self.NEGATIVE_PATH_TEMPLATE,
                category=TemplateCategory.PATH,
                required_params=["reasoning_path", "answer"],
                description="无效推理路径模板"
            ),
            PromptTemplate(
                name="path_evaluation",
                template=self.PATH_EVALUATION,
                category=TemplateCategory.EVALUATION,
                required_params=["question", "path"],
                description="评估推理路径质量"
            ),
            PromptTemplate(
                name="path_comparison",
                template=self.PATH_COMPARISON,
                category=TemplateCategory.EVALUATION,
                required_params=["question", "path1", "path2"],
                description="比较多条推理路径"
            ),
            PromptTemplate(
                name="path_generation",
                template=self.PATH_GENERATION,
                category=TemplateCategory.PATH,
                required_params=["question", "entities", "num_paths"],
                description="生成推理路径"
            ),
            PromptTemplate(
                name="mcq_path_generation",
                template=self.MCQ_PATH_GENERATION,
                category=TemplateCategory.MCQ,
                required_params=["question", "entities", "choices"],
                description="多选题推理路径生成"
            ),
        ]
        
        # 注册所有模板
        for template in templates:
            self._templates[template.name] = template
    
    def get_template(self, template_name: str) -> Optional[str]:
        """根据名称获取指定的模板内容。
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板字符串，如果不存在则返回None
        """
        template = self._templates.get(template_name.lower())
        return template.template if template else None
    
    def get_template_info(self, template_name: str) -> Optional[PromptTemplate]:
        """获取模板的完整信息，包括必要参数和描述。
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板信息对象，如果不存在则返回None
        """
        return self._templates.get(template_name.lower())
    
    def list_templates(self, category: Optional[TemplateCategory] = None) -> List[str]:
        """列出所有可用模板或特定类别的模板。
        
        Args:
            category: 可选的模板类别过滤器
            
        Returns:
            模板名称列表
        """
        if category:
            return [name for name, template in self._templates.items() 
                   if template.category == category]
        return list(self._templates.keys())
    
    def format_template(self, template_name: str, **kwargs) -> Optional[str]:
        """根据提供的参数格式化指定模板。
        
        Args:
            template_name: 模板名称
            **kwargs: 格式化参数
            
        Returns:
            格式化后的模板字符串，如果模板不存在则返回None
            
        Raises:
            ValueError: 缺少必要参数或格式化错误
        """
        template_info = self._templates.get(template_name.lower())
        if not template_info:
            return None
        
        # 检查必要参数
        missing_params = [param for param in template_info.required_params 
                          if param not in kwargs]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for template '{template_name}': {', '.join(missing_params)}"
            )
        
        try:
            return template_info.template.format(**kwargs)
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}")
    
    def format_template_safely(self, template_name: str, 
                              formatter: Optional[PromptFormatter] = None, 
                              **kwargs) -> Optional[str]:
        """使用自定义格式化器或默认格式化方法安全地格式化模板。
        
        Args:
            template_name: 模板名称
            formatter: 可选的自定义格式化器
            **kwargs: 格式化参数
            
        Returns:
            格式化后的模板字符串，如果格式化失败则返回None
        """
        try:
            if formatter:
                template = self.get_template(template_name)
                if template:
                    return formatter(template, **kwargs)
            return self.format_template(template_name, **kwargs)
        except Exception:
            return None
    
    # 添加类方法便于静态访问模板
    @classmethod
    def get_static_template(cls, template_name: str) -> Optional[str]:
        """静态方法获取模板内容，便于不实例化直接访问。
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板字符串，不存在则返回None
        """
        template_attr = template_name.upper()
        if hasattr(cls, template_attr):
            return getattr(cls, template_attr)
        return None