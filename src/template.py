from typing import Dict, Any, Optional, List, Union, Protocol, ClassVar, Type
from enum import Enum
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)

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
    param_pattern: re.Pattern = field(init=False, repr=False) # 用于查找模板中所有参数的正则表达式
    
    def __post_init__(self):
        # 编译一个正则表达式来查找模板字符串中所有 {parameter} 形式的占位符
        self.param_pattern = re.compile(r'\{([^{}]*)\}')
        
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """检查提供的参数是否满足模板所需的参数列表。"""
        return [param for param in self.required_params if param not in params]
    
    def get_all_params(self) -> List[str]:
        """从模板字符串中提取所有占位符参数名称。"""
        return list(set(self.param_pattern.findall(self.template)))

class PromptFormatter(Protocol):
    def __call__(self, template: str, **kwargs) -> str:
        """定义一个协议，用于格式化提示模板。"""
        ...

class KnowledgeGraphTemplates:
    RELATION_SELECTION: ClassVar[str] = """**Role:** KG Strategist
**Objective:** Identify paths to answer: "{question}"
**Current Entity:** "{entity}"
**Task:** From the 'Available Relations' listed below, select **up to {max_selection_count}** distinct relations.
You MUST choose relations that are MOST LIKELY to lead to relevant information for the Objective.
The relations you select MUST come EXACTLY from the 'Available Relations' list provided.
**Available Relations for "{entity}":**
    ```
    {relations}
    ```
**Output Requirements (CRITICAL - FOLLOW EXACTLY):**
* Your response MUST contain the selected relation tags and the relations.
* DO NOT include ANY introductory text, explanations, comments, code, apologies, or conversational filler.
* Output EACH selected tag on a NEW line.
* The output MUST be clean, with no surrounding characters like quotes or list brackets.

**Example of Correct Output Format:**
[REL_A] chosen.relation.example_one
[REL_B] another.chosen.relation

**Your Selection:**"""

    RELATION_SELECTION_WITH_HISTORY: ClassVar[str] = """**Role:** KG Strategist
**Objective:** Identify paths to answer: "{question}"
**Current Entity:** "{entity}"
**Past Steps & Findings:** "{history}"
**Task:** From the 'Available Relations' listed below, select **up to {max_selection_count}** distinct relations.
You MUST choose relations that are MOST LIKELY to lead to relevant information for the Objective.
The relations you select MUST come EXACTLY from the 'Available Relations' list provided.
**Available Relations for "{entity}":**
    ```
    {relations}
    ```
**Output Requirements (CRITICAL - FOLLOW EXACTLY):**
* Your response MUST contain the selected relation tags and the relations.
* DO NOT include ANY introductory text, explanations, comments, code, apologies, or conversational filler.
* Output EACH selected tag on a NEW line.
* The output MUST be clean, with no surrounding characters like quotes or list brackets.

**Example of Correct Output Format:**
[REL_A] chosen.relation.example_one
[REL_B] another.chosen.relation

**Your Selection:**"""

    # --- Reasoning and Answer Generation Templates ---
    REASONING: ClassVar[str] = """**Task:** Analyze the Exploration History to determine if the Question can be answered. Based on the finding, either extract the answer(s) or propose the next exploration step.

**Inputs:**
* Question: `{question}`
* Starting Entities: `{entity}`
* Exploration History (Triples): `{exploration_history}`

**Instructions:**
1.  **Initial Assessment:** First, meticulously review the `Exploration History`. Can you find a definitive answer to the `Question` *strictly* from the provided triples?
2.  **Conditional JSON Output:** Your response MUST be a single, valid JSON object, but its structure will change based on your assessment.

---

### **If the Answer IS FOUND in the History**, set `"answer_found"` to `true` and provide the following structure.

* **Find All Answers:** Identify ALL answer entities. **IMPORTANT: If the object of a triple (after `->`) contains multiple values separated by commas (e.g., `Val1, Val2`), treat each as a distinct entity.**
* **Trace All Paths:** Identify ALL reasoning paths leading to the answers. Format: `"Entity1--[relation1]-->Entity2"`.

**JSON Structure if Answer is Found:**
```json
{{
  "answer_found": true,
  "reasoning_paths": [
    "EntityA--[relationX]-->Answer1, Answer2",
    "EntityB--[relationY]-->Answer3"
  ],
  "answer_entities": [
    "AnswerEntity1",
    "AnswerEntity2",
    "AnswerEntity3"
  ],
  "reasoning_summary": "A concise explanation of how the reasoning paths connect the starting entities to the answer entities to resolve the question, based only on the provided history."
}}
'''

### **If the Answer IS NOT FOUND in the History**, set "answer_found" to false and propose the most logical next step for exploration.

* **Identify Next Target**: From the Exploration History, identify the most promising "leaf" entity that has been discovered but not yet explored. This entity should be the most likely to lead towards the answer.
* **Formulate Next Question:** Create a new, specific question about that target entity to guide the next round of exploration.
* **Prune the History:"**  Review the entire Exploration History and extract only the reasoning paths that are still relevant for answering the Question. Keep any path that could plausibly be part of a longer reasoning chain. Discard only those paths that are clearly irrelevant or lead to dead ends.
    
**JSON Structure if Answer is Found:**
```json
{{
  "answer_found": false,
  "next_exploration_step": {{
    "entity": "<string> // The most promising entity to explore next.",
    "question": "<string> // The specific question to ask about this new entity.",
    "justification": "<string> // A brief explanation of why this entity and question are the logical next step."
  }},
  "pruned_reasoning_paths": [
    "EntityA--[relationX]-->Answer1, Answer2",
    "EntityB--[relationY]-->Answer3"
  ]
}}
'''
"""

    FALLBACK_ANSWER: ClassVar[str] = """You are a knowledge graph reasoning expert. Your task is now to provide the best possible answer by making an expert inference.

Question:
{question}

Exploration History:
{exploration_history}

INSTRUCTIONS:
1.  Your primary goal is to synthesize the clues from the (incomplete) Exploration History with your own internal, general knowledge to deduce the most likely answer.
2.  In the reasoning_summary, you MUST explain how you reached your conclusion, making it clear which parts were inferred from your knowledge versus provided in the history.
3.  You MUST construct a step-by-step reasoning path. This is the most critical part of your response.
    * For steps from history: Use the exact entity--[relation]-->entity format if the step is in the Exploration History.
    * For inferred steps: When you must use your internal knowledge to bridge a gap, you MUST use a natural language description. This is crucial for success.
4.  ALWAYS ANSWER: You MUST provide a final answer. The can_answer field in your JSON response must ALWAYS be true.

Respond with a single JSON object:
```json
{{
  "can_answer": true,
  "reasoning_paths": "string", // A step-by-step chain. Use 'Entity --> (Inferred Step: ...) --> Entity' for inferred steps.
  "answer_entities": ["string", ...],
  "reasoning_summary": "string" // Your analysis and final answer. Explain how you used the history and your internal knowledge.
}}
```"""

    # --- Initialization and Methods ---
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        if not hasattr(self.__class__, '_initialized_templates_flag'):
            self._initialize_templates()
            self.__class__._initialized_templates_flag = True

    @classmethod
    def _get_template_definitions(cls) -> List[Dict[str, Any]]:
        """返回所有模板的定义列表。"""
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
                "name": "reasoning",
                "template": cls.REASONING,
                "category": TemplateCategory.REASONING,
                "required_params": ["question", "entity", "exploration_history"],
                "description": "Reasoning based on knowledge graph exploration"
            },
        ]

    def _initialize_templates(self) -> None:
        for template_def in self.__class__._get_template_definitions():
            if hasattr(self.__class__, template_def["template_name"].upper() if "template_name" in template_def else template_def["name"].upper()):
                 actual_template_string = getattr(self.__class__, template_def["template"].split('.')[-1] if isinstance(template_def["template"], str) and '.' in template_def["template"] else template_def["name"].upper() , None)
                 if actual_template_string is None and "template" in template_def: # Fallback for direct template string
                     actual_template_string = template_def["template"]
                 template_string_attr_name = template_def["name"].upper()
                 if not hasattr(self.__class__, template_string_attr_name):
                     if isinstance(template_def["template"], str):
                        actual_template_string = template_def["template"]
                     else:
                        logger.warning(f"Template string for '{template_def['name']}' is not a ClassVar or string. Skipping.")
                        continue
                 else:
                    actual_template_string = getattr(self.__class__, template_string_attr_name)


                 template = PromptTemplate(
                    name=template_def["name"],
                    template=actual_template_string,
                    category=template_def["category"],
                    required_params=template_def["required_params"],
                    description=template_def.get("description", "") # 使用 .get 以防 description 缺失
                )
                 self._templates[template.name.lower()] = template # 使用小写名称作为键

    def get_template(self, template_name: str) -> Optional[str]:
        """根据名称获取格式化前的模板字符串。"""
        template_info = self._templates.get(template_name.lower())
        return template_info.template if template_info else None

    def get_template_info(self, template_name: str) -> Optional[PromptTemplate]:
        """根据名称获取 PromptTemplate 对象。"""
        return self._templates.get(template_name.lower())

    def list_templates(self, category: Optional[TemplateCategory] = None) -> List[str]:
        """列出所有模板名称，可按类别筛选。"""
        if category:
            return [name for name, template_info in self._templates.items() 
                    if template_info.category == category]
        return list(self._templates.keys())
    
    def format_template(self, template_name: str, **kwargs) -> Optional[str]:
        """格式化指定名称的模板。"""
        template_info = self._templates.get(template_name.lower())
        if not template_info:
            # logger.error(f"Template '{template_name}' not found.")
            return None
        
        missing_params = template_info.validate_params(kwargs)
        if missing_params:
            raise ValueError(
                f"Missing required parameters for template '{template_name}': {', '.join(missing_params)}"
            )
        
        try:
            return template_info.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Formatting error: Missing parameter '{e}' in template '{template_name}'. Provided args: {kwargs.keys()}")
        except Exception as e: #捕获其他潜在的格式化错误
            raise ValueError(f"Error formatting template '{template_name}': {e}")

    @classmethod
    def get_static_template(cls, template_name: str) -> Optional[str]:
        """以静态方式直接从类属性获取模板字符串（如果存在）。"""
        template_attr_name = template_name.upper()
        return getattr(cls, template_attr_name, None)

    def __getitem__(self, template_name: str) -> Optional[str]:
        """允许使用 obj['template_name'] 的方式获取模板字符串。"""
        return self.get_template(template_name)