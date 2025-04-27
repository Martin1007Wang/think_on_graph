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
    RELATION_SELECTION: ClassVar[str] = """You are a knowledge graph exploration strategist. Given a question and a topic entity, select relevant relations to explore.

# Question: 
{question}

# Topic entity: 
{entity}

# Available relations from this entity:
{relations}

Select up to {max_k_relations} relation IDs that seem most promising or potentially relevant for answering the question. Consider that the answer might require exploring multiple steps.
Your response should ONLY contain the relation IDs (e.g., REL_0, REL_1) from the list above.
Your selection (IDs only, up to {max_k_relations}):"""

    RELATION_SELECTION_WITH_CONTEXT: ClassVar[str] = """You are a knowledge graph exploration strategist. Given a question, a topic entity, the exploration history so far, select relevant relations to explore next.

# Question:
{question}

# Topic entity to expand from:
{entity}

# Exploration History So Far:
{history} 

# Available relations from this entity:
{relations}

Select up to {max_k_relations} relation IDs that seem most promising or potentially relevant for answering the question. Consider that the answer might require exploring multiple steps.
Your response should ONLY contain the relation IDs (e.g., REL_0, REL_1) from the list above.
Your selection (IDs only, up to {max_k_relations}):"""
    
    INTERMEDIATE_PATH_SELECTION: ClassVar[str] = """You are a knowledge graph exploration strategist. To answer the question: "{question}"

We've reached the following point in our exploration:
{source_entity} -> {source_relation} -> some intermediate entity

Now we need to select the most promising next paths to follow from this intermediate entity.
Select up to {max_paths} paths that are most likely to lead to the answer:

{path_options}

Only respond with the IDs of the selected paths (e.g., PATH_0, PATH_1, PATH_2). Separate multiple selections with commas.
Your selection (IDs only, up to {max_paths}):"""
    
    ENTITY_SELECTION: ClassVar[str] = """You are a knowledge graph exploration strategist. Given a question and a list of candidate entities discovered, select entities to explore further in the next round.

# Question: 
{question}

# Candidate entities to evaluate (select only from these):
{entities} 

Select up to {max_k_entities} entity IDs that seem most promising or potentially relevant for finding the answer. Prioritize entities that could lead down new or diverse paths compared to what has already been explored (if history provided). Don't discard entities too early if they seem related to the question's topic.
Your response should ONLY contain the entity IDs (e.g., ENT_0, ENT_1) from the list above.
Your selection (IDs only, up to {max_k_entities}):"""

    REASONING: ClassVar[str] = """You are a knowledge graph reasoning expert. Answer questions using ONLY the provided exploration history triples.

# Question:
{question}

# Starting entities:
{entity}

# Exploration History:
{exploration_history}

INSTRUCTIONS:
1. Use ONLY information from the exploration history. No external knowledge.
2. Include ONLY exact entity names from the history in your answer_entities.
3. If insufficient evidence exists, set can_answer to false.
4. IMPORTANT: Make sure your JSON is valid - all string values must be properly quoted.
5. For reasoning_path, use a properly quoted string showing the entity-relation-entity chain.

Respond with a single JSON object in this exact format (with proper quoting):
```json
{{
  "can_answer": true or false,
  "reasoning_path": "Entity1--[relation]-->Entity2--[relation]-->Entity3",
  "answer_entities": ["Entity Name 1", "Entity Name 2"],
  "analysis": "Your concise answer or explanation here."
}}
```"""

    FINAL_ANSWER: ClassVar[str] = """You are a knowledge graph reasoning expert. Answer the question using the provided exploration history, providing your best answer even when evidence is limited.

Question:
{question}

Starting entities:
{entity}

Full knowledge graph exploration:
{full_exploration}

INSTRUCTIONS:
1. Use ONLY information from the exploration history. No external knowledge.
2. Include ONLY exact entity names from the history in your answer_entities.
3. Even with limited evidence, attempt to provide the most reasonable answer based on available triples.
4. Set can_answer to false ONLY if absolutely no relevant information exists.
5. If you cannot answer, provide a concise explanation of why.

Respond with a single JSON object:
```json
{{
  "can_answer": boolean, // True in most cases, False only if completely impossible to answer
  "reasoning_path": "string", // Step-by-step entity--[relation]-->entity chain, even if partial
  "answer_entities": ["string", ...], // Best candidate target entities or empty list if truly impossible
  "analysis": "string" // Concise answer statement with confidence level or explanation of impossibility
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
            # {
            #     "name": "fallback_answer",
            #     "template": cls.FALLBACK_ANSWER,
            #     "category": TemplateCategory.REASONING,
            #     "required_params": ["question"],
            #     "description": "Fallback answer template"
            # },
            {
                "name": "relation_selection",
                "template": cls.RELATION_SELECTION,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "entity", "relations", "max_k_relations"],
                "description": "Select most relevant relations for exploration"
            },
            {
                "name": "relation_selection_context",
                "template": cls.RELATION_SELECTION_WITH_CONTEXT,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "entity", "history", "context", "relations", "max_k_relations"],
                "description": "Select relations based on historical context"
            },
            {
                "name": "intermediate_path_selection",
                "template": cls.INTERMEDIATE_PATH_SELECTION,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "source_entity","source_relation", "path_options", "max_paths"],
                "description": "Select relevant paths from intermediate nodes"
            },
            # {
            #     "name": "targeted_relation_selection",
            #     "template": cls.TARGETED_RELATION_SELECTION,
            #     "category": TemplateCategory.RELATION,
            #     "required_params": ["question", "reasoning", "node", "relations", "relation_ids"],
            #     "description": "Select specific relation from intermediate node"
            # },
            {
                "name": "entity_selection",
                "template": cls.ENTITY_SELECTION,
                "category": TemplateCategory.RELATION,
                "required_params": ["question", "entities", "max_k_entities", "entity_ids"],
                "description": "Select relevant candidate entities"
            },
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
            {
                "name": "final_answer",
                "template": cls.FINAL_ANSWER,
                "category": TemplateCategory.REASONING,
                "required_params": ["question", "entity", "full_exploration"],
                "description": "Final answer with reasoning path"
            },
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
            #     "required_params": ["question", "entities", "current_round", "max_entities"],
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

