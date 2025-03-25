class Template:
    # === 通用提示模板 ===
    
    # 零样本提示：引导模型作为知识图谱专家分析推理路径，包含多维度评估指导
    ZERO_SHOT_PROMPT = (
        "As a knowledge graph expert, analyze reasoning paths connecting the topic entity to potential answers.\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entity: \n{entity}\n\n"
        "Consider: 1) Semantic alignment with question intent, 2) Path efficiency, "
        "3) Relation relevance, 4) Factual correctness"
    )

    # === 关系选择模板 ===

    # 关系选择模板：展示候选关系并解释选择理由
    RELATION_SELECTION_TEMPLATE = (
        "# Candidate Relations:\n{selected_relations}\n\n"
        "# Selection Analysis:\n"
        "These relations for {entity} are promising because they {explanation}"
    )

    # === 路径分析模板 ===

    # 基础路径模板：简洁展示推理路径和答案
    PATH_TEMPLATE = (
        "# Reasoning Path:\n{reasoning_path}\n\n"
        "# Analysis:\nPath connects topic entity to answer through relevant relations.\n\n"
        "# Answer:\n{answer}"
    )

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

    # === 路径比较与评估模板 ===

    # 路径评估模板：多维度评分单条路径，提供结论
    PATH_EVALUATION_TEMPLATE = (
        "# Path Evaluation\n\n"
        "## Path:\n{path}\n\n"
        "## Assessment:\n"
        "- Semantic: {semantic_score}/10 - {semantic_explanation}\n"
        "- Efficiency: {efficiency_score}/10 - {efficiency_explanation}\n"
        "- Factual: {factual_score}/10 - {factual_explanation}\n"
        "- Overall: {overall_score}/10\n\n"
        "## Conclusion:\n{conclusion}"
    )

    # === 路径生成模板 ===

    # 零样本路径生成提示：引导生成推理路径，无示例
    PATH_GENERATION_ZERO_SHOT = (
        "Generate reasoning paths in the KG from topic entities to answer the question.\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entities: \n{entities}"
    )

    # 多选题零样本路径生成提示：生成路径并结合选项
    PATH_GENERATION_MCQ_ZERO_SHOT = (
        "Generate reasoning paths in the KG from topic entities to answer the question.\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entities: \n{entities}\n\n"
        "# Answer Choices:\n{choices}"
    )

    # 少样本路径生成提示：提供示例引导生成推理路径
    PATH_GENERATION_FEW_SHOT = (
        "Generate reasoning paths in the KG from topic entities to answer the question.\n\n"
        "Examples:\n{examples}\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entities: \n{entities}"
    )