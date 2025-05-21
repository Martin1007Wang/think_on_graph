from typing import Dict, Any, Optional, List, Union, Protocol, ClassVar, Type
from enum import Enum
from dataclasses import dataclass, field
import functools
import re

class TemplateCategory(Enum):
    RELATION = "relation"
    REASONING = "reasoning"
    PATH = "path"
    EVALUATION = "evaluation"
    ZERO_SHOT = "zero_shot"
    MCQ = "mcq"

@dataclass
class PromptTemplate:
    name: str
    template: str
    category: TemplateCategory
    required_params: List[str]
    description: str = ""
    param_pattern: re.Pattern = field(init=False, repr=False)
    
    def __post_init__(self):
        self.param_pattern = re.compile(r'\{([^{}]*)\}')
        
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        return [param for param in self.required_params if param not in params]
    
    def get_all_params(self) -> List[str]:
        return list(set(self.param_pattern.findall(self.template)))

class PromptFormatter(Protocol):
    def __call__(self, template: str, **kwargs) -> str:
        """Format a prompt template"""
        ...

class KnowledgeGraphTemplates:
    RELATION_SELECTION: ClassVar[str] = """**Role:** KG Strategist
**Objective:** Identify paths to answer: "{question}"
**Current Entity:** "{entity}"

**Task:** From 'Available Relations' below, select **up to {max_selection_count}** distinct relations.
Choose relations MOST LIKELY to lead to relevant information for the Objective.
**Available Relations for "{entity}":**
    ```
    {relations}
    ```
**Output Requirements (CRITICAL):**
* Respond ONLY with the **exact, complete lines** selected from 'Available Relations'.
* One selected relation per line.
* NO other text, explanations, or comments.

**Your Selection:**"""

    RELATION_SELECTION_WITH_HISTORY: ClassVar[str] = """**Role:** KG Strategist
**Objective:** Identify paths to answer: "{question}"
**Current Entity:** "{entity}"
**Past Steps & Findings:** "{history}"
**Task:** From 'Available Relations' below, select **up to {max_selection_count}** distinct relations.
Choose relations MOST LIKELY to lead to relevant information for the Objective.
**Available Relations for "{entity}":**
    ```
    {relations}
    ```
**Output Requirements (CRITICAL):**
* Respond ONLY with the **exact, complete lines** selected from 'Available Relations'.
* One selected relation per line.
* NO other text, explanations, or comments.

**Your Selection:**"""
    
    PATH_SELECTION: ClassVar[str] = """**Role:** Knowledge Graph Exploration Result Evaluator

**Objective:** Evaluate the findings from the latest exploration step and select the most relevant ones (direct relations or paths via intermediate entities) that are most likely to help answer the 'Question'.

**Context:**
* Question: `{question}`
* Entity Expanded in this Step: `{entity}`
* Exploration History:`{history}`
* Composite Paths:
    ```
    {paths}
    ```
**Task:**
1.  Review the context and the 'Composite Paths' list.
2.  Evaluate the 'Composite Paths' list. Choose the {max_selection_count} most promising lines from the list that represent the best paths to explore.
3.  **CRITICAL: Output Requirements:**
    * Respond ONLY with the **exact, complete lines** you selected from the 'Composite Paths' list.
    * List one selected line per line in your response.
    * **Example Output Format (if selecting two lines):**
        ```
        PATH_3: {entity}-[relationX]->EntityA-[relationY]->TargetEntityB
        PATH_7: {entity}-[relationX]->EntityC-[relationZ]->TargetEntityD
        ```
    * ABSOLUTELY NO other text, explanations, commentary, or modifications to the selected lines.

**Your Selection:**"""

    CVT_ATTRIBUTE_SELECTION_TEMPLATE: ClassVar[str] = """**Role:** Knowledge Graph Analyst

**Objective:** Evaluate individual attributes (relation -> target pairs) extracted from potentially relevant CVT instances and select the specific attributes most likely to contribute to answering the user's question.

**Context:**
* Question: `{question}`
* Exploration Step: CVT instances reached from entity '{source_entity}' via relation '{source_relation}'.
* Exploration History:`{history}`

**Candidate Attributes:**
Below is a list of attributes extracted from the CVT instances. Each attribute is identified by an ID (e.g., [ATTR_0]) and shows the source CVT (in parentheses), the relation (attribute name), and its target value(s).
```
{attributes}
```

**Task:**
1. Carefully review the 'Question' and the 'Candidate Attributes'. Understand the specific information needed (e.g., person for "who", date for "when", title for "what position") and any constraints (e.g., year '2011', role 'governor').
2. Analyze each attribute line, considering its relation and the CVT it came from.
3. Evaluate the relevance of each attribute for answering the 'Question' within the given exploration context. Check if the attribute's relation or value matches the question's intent and constraints (like date or title, if available in the attribute or CVT context).
4. Identify and select up to **{max_selection_count}** attribute lines that represent the **most relevant information** found to answer the question.

**CRITICAL: Output Requirements:**
* Respond ONLY with the **exact, complete lines** you selected from the 'Candidate Attributes' list.
* List one selected line per line in your response.
* **Example Output Format (if selecting two lines):**
    ```
    ATTR_0: EntityA-[relationY]->TargetEntityB
    ATTR_7: EntityC-[relationZ]->TargetEntityD
    ```
* ABSOLUTELY NO other text, explanations, commentary, or modifications to the selected lines.

**Your Selection:**"""


    REASONING: ClassVar[str] = """**Task:** Extract descriptive answer entities for the Question strictly from the object part (after '->') of relevant triples in the Exploration History. Format the output as a JSON list.

**Inputs:**
* Question: `{question}`
* Starting Entities: `{entity}`
* Exploration History (Triples): `{exploration_history}`

**Instructions:**
1.  **Strict Adherence:** Base answer *solely* on provided triples; no external knowledge or inference.
2.  **Find All Answers:** Identify ALL answer entities in the history (use exact names). **IMPORTANT: If the object part of a triple (after `->`) contains multiple values separated by commas (e.g., `Val1, Val2, Val3`), you MUST treat each value as a distinct entity and include each one separately in the `answer_entities` array.**
3.  **Trace All Paths:** Identify and include ALL reasoning paths that lead to the identified answer entities. Path *content* format: `Entity1--[relation1]-->Entity2`. If one path string leads to multiple comma-separated answers, include that single path string once in `reasoning_paths`.
4.  **JSON Output:**
    * Respond with a *single*, *strictly valid* JSON object.
    * **CRITICAL JSON RULES:**
        * Adhere precisely to the structure below.
        * **ALL** string values (keys, values, array elements) **MUST** use double quotes (`"`).
        * No trailing commas.
    * **JSON Structure:**
        ```json
        {{
          "can_answer": <boolean>, // True if answer found in history
          "reasoning_paths": [ // Array of all reasoning path strings leading to answers
                               // Each element MUST be a string like: "EntityA--[relX]-->Answer1[, Answer2...]"
            "EntityA--[relationX]-->Answer1, Answer2",
            "EntityB--[relationY]-->Answer3"
            // ... include all paths
          ],
          "answer_entities": [ // Array of all unique answer entity names
                               // Each element MUST be a string like: "AnswerEntity1"
                               // **If a triple yields comma-separated values like "Val1, Val2", list them as ["Val1", "Val2"].**
            "AnswerEntity1",
            "AnswerEntity2"
            // ... include all answers extracted according to Instruction #2
          ],
          "analysis": "<string> // Explain how paths yield answers, based only on history triples."
        }}
        ```"""
    

    FALLBACK_ANSWER: ClassVar[str] = """You are a knowledge graph reasoning expert. Your task is to ALWAYS provide an answer to the question using the exploration history, even if evidence is limited or inconclusive.

Question:
{question}

Exploration History:
{exploration_history}

INSTRUCTIONS:
1. You MUST provide an answer even with limited evidence. Make your best guess based on available information.
2. IMPORTANT: You MUST set can_answer to true in ALL cases. Your job is to ALWAYS provide the best possible answer.
3. If evidence is weak, acknowledge this in your analysis but still provide your best answer.

Respond with a single JSON object:
```json
{{
  "can_answer": true, // ALWAYS set to true - you MUST provide an answer
  "reasoning_path": "string", // Step-by-step entity--[relation]-->entity chain, even if partial or best guess
  "answer_entities": ["string", ...], // Best candidate target entities based on available information
  "analysis": "string" // Concise answer statement with confidence level and justification
}}
```"""

    ZERO_SHOT_PROMPT: ClassVar[str] = """You are a knowledge graph reasoning expert. Analyze reasoning paths connecting the topic entity to potential answers.
Question:
{question}

Topic entity:
{entity}

Your task is to identify the most promising reasoning paths that could lead to the answer."""

    SEMANTIC_PATH_TEMPLATE: ClassVar[str] = """# Semantic Reasoning Path: {reasoning_path}

Path Analysis:
HIGH SEMANTIC RELEVANCE - Strong alignment with question intent. Strengths: Semantic alignment, relevant relations, intuitive reasoning

Answer: {answer}"""

    SHORTEST_PATH_TEMPLATE: ClassVar[str] = """# Shortest Reasoning Path: {reasoning_path}

Path Analysis:
MAXIMUM EFFICIENCY - Most direct connection in the graph. Strengths: Direct path, minimal hops, structurally optimal

Answer:{answer}"""

    NEGATIVE_PATH_TEMPLATE: ClassVar[str] = """# Problematic Reasoning Path: {reasoning_path}

Path Analysis:
LOW RELEVANCE - Poor alignment with question intent. Issues: Poor semantic alignment, irrelevant relations, illogical reasoning

Answer:
{answer}"""

    POSITIVE_PATH_TEMPLATE: ClassVar[str] = """# Reasoning Path: {reasoning_path}

Path Analysis:
HIGH RELEVANCE - Strong alignment with question intent. Strengths: Direct semantic connection, relevant relations, logical reasoning flow

Answer:
{answer}"""

    PATH_EVALUATION: ClassVar[str] = """You are a knowledge graph reasoning expert. Evaluate the following reasoning path for answering the question.

Question:
{question}

Path:
{path}

Provide a comprehensive assessment using the following metrics:

[Semantic Relevance: 1-10] - How well the path aligns with the question's meaning [Path Efficiency: 1-10] - How direct and concise the path is [Factual Accuracy: 1-10] - How factually correct the path relationships are [Overall Quality: 1-10] - Overall assessment considering all factors

[Analysis: detailed explanation of strengths and weaknesses] [Conclusion: whether this path effectively answers the question]"""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        if not hasattr(self.__class__, '_initialized'):
            self._initialize_templates()
            self.__class__._initialized = True

    @classmethod
    def _get_template_definitions(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "fallback_answer",
                "template": cls.FALLBACK_ANSWER,
                "category": TemplateCategory.REASONING,
                "required_params": ["question", "exploration_history"],
                "description": "Fallback answer template"
            },
            {
                "name": "relation_selection",
                "template": cls.RELATION_SELECTION,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "entity", "relations", "max_selection_count"],
                "description": "Select most relevant relations for exploration"
            },
            {
                "name": "relation_selection_with_history",
                "template": cls.RELATION_SELECTION_WITH_HISTORY,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "entity", "history", "relations", "max_selection_count"],
                "description": "Select relations based on history"
            },
            {
                "name": "path_selection",
                "template": cls.PATH_SELECTION,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "paths", "max_selection_count"],
                "description": "Select relevant paths"
            },
            {
                "name": "cvt_attribute_selection",
                "template": cls.CVT_ATTRIBUTE_SELECTION_TEMPLATE,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "source_entity", "source_relation", "history", "attributes", "max_selection_count"],
                "description": "Select specific attributes from CVT instances most relevant to answering the question."
            },
            # {
            #     "name": "targeted_relation_selection",
            #     "template": cls.TARGETED_RELATION_SELECTION,
            #     "category": TemplateCategory.RELATION,
            #     "required_params": ["question", "reasoning", "node", "relations", "relation_ids"],
            #     "description": "Select specific relation from intermediate node"
            # },
            # {
            #     "name": "entity_selection",
            #     "template": cls.ENTITY_SELECTION,
            #     "category": TemplateCategory.RELATION,
            #     "required_params": ["question", "entities", "max_selection_count", "entity_ids"],
            #     "description": "Select relevant candidate entities"
            # },
            {
                "name": "reasoning",
                "template": cls.REASONING,
                "category": TemplateCategory.REASONING,
                "required_params": ["question", "entity", "exploration_history"],
                "description": "Reasoning based on knowledge graph exploration"
            },
            # {
            #     "name": "enhanced_reasoning",
            #     "template": cls.ENHANCED_REASONING,
            #     "category": TemplateCategory.REASONING,
            #     "required_params": ["question", "entity", "exploration_history"],
            #     "description": "Enhanced reasoning with focus on intermediate nodes"
            # },
            # {
            #     "name": "final_answer",
            #     "template": cls.FINAL_ANSWER,
            #     "category": TemplateCategory.REASONING,
            #     "required_params": ["question", "entity", "full_exploration"],
            #     "description": "Final answer with reasoning path"
            # },
            {
                "name": "zero_shot",
                "template": cls.ZERO_SHOT_PROMPT,
                "category": TemplateCategory.ZERO_SHOT,
                "required_params": ["question", "entity"],
                "description": "Zero-shot reasoning path analysis"
            },
            {
                "name": "semantic_path",
                "template": cls.SEMANTIC_PATH_TEMPLATE,
                "category": TemplateCategory.PATH,
                "required_params": ["reasoning_path", "answer"],
                "description": "Semantically relevant reasoning path template"
            },
            {
                "name": "shortest_path",
                "template": cls.SHORTEST_PATH_TEMPLATE,
                "category": TemplateCategory.PATH,
                "required_params": ["reasoning_path", "answer"],
                "description": "Shortest reasoning path template"
            },
            {
                "name": "negative_path",
                "template": cls.NEGATIVE_PATH_TEMPLATE,
                "category": TemplateCategory.PATH,
                "required_params": ["reasoning_path", "answer"],
                "description": "Invalid reasoning path template"
            },
            {
                "name": "positive_path",
                "template": cls.POSITIVE_PATH_TEMPLATE,
                "category": TemplateCategory.PATH,
                "required_params": ["reasoning_path", "answer"],
                "description": "Valid reasoning path template"
            },
            {
                "name": "path_evaluation",
                "template": cls.PATH_EVALUATION,
                "category": TemplateCategory.EVALUATION,
                "required_params": ["question", "path"],
                "description": "Evaluate reasoning path quality"
            },
            # {
            #     "name": "path_comparison",
            #     "template": cls.PATH_COMPARISON,
            #     "category": TemplateCategory.EVALUATION,
            #     "required_params": ["question", "path1", "path2"],
            #     "description": "Compare multiple reasoning paths"
            # },
            # {
            #     "name": "path_generation",
            #     "template": cls.PATH_GENERATION,
            #     "category": TemplateCategory.PATH,
            #     "required_params": ["question", "entities", "num_paths"],
            #     "description": "Generate reasoning paths"
            # },
            # {
            #     "name": "mcq_path_generation",
            #     "template": cls.MCQ_PATH_GENERATION,
            #     "category": TemplateCategory.MCQ,
            #     "required_params": ["question", "entities", "choices"],
            #     "description": "Multiple choice question reasoning path generation"
            # },
            # {
            #     "name": "frontier_prioritization",
            #     "template": cls.FRONTIER_PRIORITIZATION,
            #     "category": TemplateCategory.RELATION,
            #     "required_params": ["question", "entities", "current_round", "max_selection_count"],
            #     "description": "Prioritize frontier entities for exploration"
            # },
        ]

    def _initialize_templates(self) -> None:
        for template_def in self.__class__._get_template_definitions():
            template = PromptTemplate(
                name=template_def["name"],
                template=template_def["template"],
                category=template_def["category"],
                required_params=template_def["required_params"],
                description=template_def["description"]
            )
            self._templates[template.name] = template

    def get_template(self, template_name: str) -> Optional[str]:
        template = self._templates.get(template_name.lower())
        return template.template if template else None

    def get_template_info(self, template_name: str) -> Optional[PromptTemplate]:
        return self._templates.get(template_name.lower())

    def list_templates(self, category: Optional[TemplateCategory] = None) -> List[str]:
        if category:
            return [name for name, template in self._templates.items() 
                if template.category == category]
        return list(self._templates.keys())
    
    def format_template(self, template_name: str, **kwargs) -> Optional[str]:
        template_info = self._templates.get(template_name.lower())
        if not template_info:
            return None
        
        missing_params = template_info.validate_params(kwargs)
        if missing_params:
            raise ValueError(
                f"Missing required parameters for template '{template_name}': {', '.join(missing_params)}"
            )
        
        try:
            return template_info.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing format parameter in template '{template_name}': {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}")

    @classmethod
    def get_static_template(cls, template_name: str) -> Optional[str]:
        template_attr = template_name.upper()
        if hasattr(cls, template_attr):
            return getattr(cls, template_attr)
        return None

    def __getitem__(self, template_name: str) -> Optional[str]:
        return self.get_template(template_name)

