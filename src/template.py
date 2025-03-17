# 提示模板
class Template:
    ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.
                        # Question: 
                        {question}
                        # Topic entity: 
                        {entity}"""

    RELATION_SELECTION_TEMPLATE = """# Selected Relations (ordered by relevance):
                        {selected_relations}
                        # Explanation:
                        These relations are most relevant because they {explanation}"""

    PATH_TEMPLATE = """# Reasoning Path:
                        {reasoning_path}
                        # Answer:
                        {answer}"""