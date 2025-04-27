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
    
    RELATION_SELECTION = """
    You are a knowledge graph exploration strategist. Given a question and a topic entity, select relevant relations to explore.

    # Question: 
    {question}

    # Topic entity: 
    {entity}

    # Available relations from this entity (select only from these):
    {relations}

    Select **up to {max_k_relations}** relation IDs that seem **most promising or potentially relevant** for answering the question. Consider that the answer might require exploring multiple steps.
    Your response should ONLY contain the relation IDs (e.g., REL_0, REL_1) from the list above.
    Valid relation IDs: {relation_ids}
    Your selection (IDs only, up to {max_k_relations}):
    """

    RELATION_SELECTION_WITH_CONTEXT = """
    You are a knowledge graph exploration strategist. Given a question, a topic entity, the exploration history so far, and the current context entities, select relevant relations to explore next.

    # Question:
    {question}

    # Topic entity to expand from:
    {entity}

    # Exploration History So Far:
    {history} 
    # Current Context Entities:
    {context}

    # Available relations from '{entity}' (select only from these):
    {relations}

    Identify relation IDs that seem **promising or potentially relevant** for finding the answer, considering the question and the information **already uncovered in the history**. Select **up to {max_k_relations}** IDs. Prioritize relations that open **new avenues** not yet explored.
    Your response should ONLY contain the chosen relation IDs (e.g., REL_0, REL_1) separated by commas or newlines.
    Valid relation IDs: {relation_ids}

    Your selection (IDs only, up to {max_k_relations}):
    """
    
    ENTITY_SELECTION = """
    You are a knowledge graph exploration strategist. Given a question and a list of candidate entities discovered, select entities to explore further in the next round.

    # Question: 
    {question}

    # Candidate entities to evaluate (select only from these):
    {entities} 
    # (Optional Context - Add if feasible) Exploration History So Far:
    # {exploration_history} 

    Select **up to {max_k_entities}** entity IDs that seem **most promising or potentially relevant** for finding the answer. Prioritize entities that could lead down **new or diverse paths** compared to what has already been explored (if history provided). Don't discard entities too early if they seem related to the question's topic.
    Your response should ONLY contain the entity IDs (e.g., ENT_0, ENT_1) from the list above.
    Valid entity IDs: {entity_ids}
    Your selection (IDs only, up to {max_k_entities}):
    """

    # 推理与决策模板
    REASONING = """
    You are a knowledge graph reasoning expert. Given a question and information from a knowledge graph exploration, determine if you can confidently answer based *primarily* on the provided triples and provide the result in JSON format.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Knowledge graph exploration (over {num_rounds} rounds):
    {exploration_history}

    CRITICAL RULES:
    1. Base your reasoning *as much as possible* on the EXPLICIT Entity--[Relation]-->Target triples shown in the exploration history. Avoid making large inferential leaps.
    2. DO NOT synthesize information *significantly* beyond what's present in the triples.
    3. Prefer directly connected paths. Only combine information from different paths if strongly supported by the context within the history.
    4. The `answer_entities` list MUST ONLY contain exact entity names found as Targets in the reasoning path *within the history*.
    5. If a confident answer cannot be constructed *primarily* from the history, set `can_answer` to false.

    Respond ONLY with a valid JSON object adhering to this exact structure:
    ```json
    {{
      "can_answer": boolean, // True if the question can be reasonably answered based *primarily* on the exploration_history, False otherwise.
      "reasoning_path": "string", // If True: Provide the step-by-step entity--[relation]-->entity chain from the history that leads to the answer. If False, explain why the history provides insufficient evidence.
      "answer_entities": ["string", ...], // If True: Provide a list of the EXACT entity names that form the answer, extracted directly from the reasoning path targets. Must NOT contain descriptive text. If False, provide an empty list [].
      "answer_sentence": "string", // If True: Provide a brief natural language sentence summarizing the answer based *only* on the answer_entities. If False, state that the answer cannot be determined.
      "verification": "string", // Briefly explain the confidence level, pointing to the key supporting triple(s) in the history. If cannot answer, explain confidence is low due to lack of direct evidence.
      "is_verified": boolean // True if `answer_entities` are strongly supported by the `reasoning_path` from the history, False otherwise.
    }}
    ```
    Ensure the output is a single, valid JSON object and nothing else.
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
    You are a knowledge graph reasoning expert. Given a question and the full exploration history of a knowledge graph, provide your best answer in JSON format.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Full knowledge graph exploration:
    {full_exploration}

    CRITICAL RULES:
    1. Base your reasoning ONLY on the EXPLICIT Entity--[Relation]-->Target triples shown in the exploration history.
    2. DO NOT infer or synthesize information not directly present in the triples.
    3. The `answer_entities` list MUST ONLY contain exact entity names found as Targets in the reasoning path.
    4. If you cannot answer based SOLELY on the history, set `can_answer` to false.

    Based strictly on the exploration history provided, determine if you can answer the question and formulate your response as a valid JSON object.

    Respond ONLY with a valid JSON object adhering to this exact structure:
    ```json
    {{
      "can_answer": boolean, // True if the question can be answered based *only* on the full_exploration, False otherwise.
      "reasoning_path": "string", // If can_answer is True, provide the step-by-step entity--[relation]-->entity chain from the history that leads to the answer. If False, explain why the history is insufficient.
      "answer_entities": ["string", ...], // If can_answer is True, provide a list of the EXACT entity names that form the answer, extracted directly from the reasoning path targets. Must NOT contain descriptive text. If False, provide an empty list [].
      "answer_sentence": "string", // If can_answer is True, provide a brief natural language sentence summarizing the answer based *only* on the answer_entities. If False, state that the answer cannot be determined.
      "verification": "string", // Briefly explain the confidence level or point to the specific triples in the history that directly support the answer_entities. If cannot answer, explain confidence is low.
      "is_verified": boolean // True if `answer_entities` are directly supported by the `reasoning_path` without ambiguity or external knowledge, False otherwise.
    }}
    ```
    Ensure the output is a single, valid JSON object and nothing else.
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
                name="fallback_answer",
                template=self.FALLBACK_ANSWER,
                category=TemplateCategory.REASONING,
                required_params=["question"],
                description="备用答案模板"
            ),
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
                name="entity_selection",
                template=self.ENTITY_SELECTION,
                category=TemplateCategory.RELATION,
                required_params=["question", "entities", "max_k_entities", "entity_ids"],
                description="对候选实体进行相关性选择"
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