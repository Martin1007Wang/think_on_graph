class Template:
    # === 通用提示模板 ===
    
    # 零样本提示：引导模型作为知识图谱专家分析推理路径，包含多维度评估指导
    ZERO_SHOT_PROMPT = (
        "You are a knowledge graph reasoning expert. Given a question and a topic entity, "
        "your task is to analyze reasoning paths in the knowledge graph that connect the "
        "topic entity to potential answer entities.\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entity: \n{entity}\n\n"
        "When evaluating reasoning paths, consider multiple dimensions:\n"
        "1. Semantic alignment - How well does the path align with the question's intent?\n"
        "2. Path efficiency - Is the path direct or does it contain unnecessary detours?\n"
        "3. Relation relevance - Are the relations in the path relevant to answering the question?\n"
        "4. Factual correctness - Does the path lead to an accurate answer?"
    )

    # === 关系选择模板 ===

    # 关系选择模板：展示候选关系并解释选择理由
    RELATION_SELECTION_TEMPLATE = (
        "# Candidate Relations:\n"
        "{selected_relations}\n\n"
        "# Selection Analysis:\n"
        "I've analyzed the relations connected to {entity} and identified these as most promising "
        "because they {explanation}"
    )

    # === 路径分析模板 ===

    # 基础路径模板：简洁展示推理路径和答案
    PATH_TEMPLATE = (
        "# Reasoning Path:\n"
        "{reasoning_path}\n\n"
        "# Analysis:\n"
        "This path connects the topic entity to the answer through a sequence of relevant relations.\n\n"
        "# Answer:\n"
        "{answer}"
    )

    # 语义路径模板：强调语义相关性，分析路径优势
    SEMANTIC_PATH_TEMPLATE = (
        "# Semantic Reasoning Path:\n"
        "{reasoning_path}\n\n"
        "# Path Analysis:\n"
        "This path has HIGH SEMANTIC RELEVANCE to the question. Each relation in this path was selected "
        "based on its semantic alignment with the question intent, ensuring the reasoning process "
        "closely follows the question's requirements.\n\n"
        "Strengths:\n"
        "- Strong semantic alignment with the question's intent\n"
        "- Relations chosen based on relevance to the query\n"
        "- Follows intuitive reasoning that humans would use\n\n"
        "# Answer:\n"
        "{answer}"
    )

    # 最短路径模板：强调路径效率，突出简洁性
    SHORTEST_PATH_TEMPLATE = (
        "# Shortest Reasoning Path:\n"
        "{reasoning_path}\n\n"
        "# Path Analysis:\n"
        "This path has MAXIMUM EFFICIENCY in the knowledge graph. It represents the most direct "
        "connection between the topic entity and the answer entity in the graph structure.\n\n"
        "Strengths:\n"
        "- Most direct path in the graph structure\n"
        "- Minimal number of hops between entities\n"
        "- Structurally optimal in the knowledge graph\n\n"
        "# Answer:\n"
        "{answer}"
    )

    # 负面路径模板：指出路径缺陷，分析无关性
    NEGATIVE_PATH_TEMPLATE = (
        "# Problematic Reasoning Path:\n"
        "{reasoning_path}\n\n"
        "# Path Analysis:\n"
        "This path has LOW RELEVANCE to the question. Although it connects entities in the knowledge "
        "graph, the relations do not align well with the question's intent.\n\n"
        "Issues:\n"
        "- Poor semantic alignment with the question\n"
        "- Contains irrelevant or tangential relations\n"
        "- Does not follow logical reasoning for the given question\n\n"
        "# Answer:\n"
        "{answer}"
    )

    # === 路径比较与评估模板 ===

    # 路径评估模板：多维度评分单条路径，提供结论
    PATH_EVALUATION_TEMPLATE = (
        "# Comprehensive Path Evaluation\n\n"
        "## Path:\n"
        "{path}\n\n"
        "## Multi-dimensional Assessment:\n"
        "- Semantic Relevance: {semantic_score}/10 - {semantic_explanation}\n"
        "- Structural Efficiency: {efficiency_score}/10 - {efficiency_explanation}\n"
        "- Factual Support: {factual_score}/10 - {factual_explanation}\n"
        "- Overall Quality: {overall_score}/10\n\n"
        "## Conclusion:\n"
        "{conclusion}"
    )

    # === 路径生成模板 ===

    # 零样本路径生成提示：引导生成推理路径，无示例
    PATH_GENERATION_ZERO_SHOT = (
        "Reasoning path is a sequence of triples in the KG that connects the topic entities in the question "
        "to answer entities. Given a question, please generate some reasoning paths in the KG starting from "
        "the topic entities to answer the question.\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entities: \n{entities}"
    )

    # 多选题零样本路径生成提示：生成路径并结合选项
    PATH_GENERATION_MCQ_ZERO_SHOT = (
        "Reasoning path is a sequence of triples in the KG that connects the topic entities in the question "
        "to answer entities. Given a question, please generate some reasoning paths in the KG starting from "
        "the topic entities to answer the question.\n\n"
        "# Question: \n{question}\n\n"
        "# Topic entities: \n{entities}\n\n"
        "# Answer Choices:\n"
        "{choices}"
    )

    # 少样本路径生成提示：提供示例引导生成推理路径
    PATH_GENERATION_FEW_SHOT = (
        "Reasoning path is a sequence of triples in the KG that connects the topic entities in the question "
        "to answer entities. Given a question, please generate some reasoning paths in the KG starting from "
        "the topic entities to answer the question.\n\n"
        "Here are some examples:\n"
        "{examples}\n\n"
        "Now, please generate reasoning paths for the following question:\n"
        "# Question: \n{question}\n\n"
        "# Topic entities: \n{entities}"
    )