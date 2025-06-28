from typing import Dict, Any, Optional, List, ClassVar
from enum import Enum
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)

# --- Data-only classes, no major changes needed ---
class TemplateCategory(Enum):
    RELATION = "relation"
    REASONING = "reasoning"
    PATH = "path"
    EVALUATION = "evaluation"
    ZERO_SHOT = "zero_shot"
    MCQ = "mcq"

@dataclass
class PromptTemplate:
    """A simple data container for a single prompt template."""
    name: str
    template: str
    category: TemplateCategory
    required_params: List[str]
    description: str = ""

# =============================================================================
#  The Refactored KnowledgeGraphTemplates Class
# =============================================================================

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
    # ... (include all your other template strings here) ...
    FALLBACK_ANSWER: ClassVar[str] = """You are a knowledge graph reasoning expert. The standard search based on the provided history was inconclusive. Your task is now to provide the best possible answer by making an expert inference.

Question:
{question}

Exploration History (use this as potentially incomplete context):
{exploration_history}

INSTRUCTIONS:
1.  Your primary goal is to synthesize the clues from the (incomplete) `Exploration History` with your own **internal, general knowledge** to deduce the most likely answer.
2.  In the `reasoning_summary`, you must explain how you reached your conclusion, making it clear which parts were inferred.
3.  You MUST always construct a step-by-step `entity--[relation]-->entity` path.
    * If a reasoning step (a relation) is directly supported by the `Exploration History`, state it normally.
        * Example: `The Matrix--[director]-->Lana Wachowski`
    * If you must infer a reasoning step using your internal knowledge, you MUST **annotate it by adding the prefix `inferred:`** to the relation name.
        * Example: `Lana Wachowski--[inferred: attended college]--> inffered: Bard College`
4.  **ALWAYS ANSWER:** You MUST provide a final answer. The `can_answer` field in your JSON response must ALWAYS be `true`.

Respond with a single JSON object:
```json
{{
  "can_answer": true,
  "reasoning_path": "string", // A step-by-step chain. Inferred steps MUST be annotated, like `entity--[inferred: relation]-->entity`.
  "answer_entities": ["string", ...],
  "reasoning_summary": "string" // Your analysis and final answer. For example: "While the history confirmed Lana Wachowski as the director, my general knowledge indicates she attended Bard College."
}}
```"""
    
    _DEFINITIONS: ClassVar[List[Dict[str, Any]]] = [
        {
            "name": "relation_selection",
            "template": RELATION_SELECTION,
            "category": TemplateCategory.RELATION,
            "required_params": ["question", "entity", "relations", "max_selection_count"],
            "description": "Select most relevant relations for exploration."
        },
        {
            "name": "relation_selection_with_history",
            "template": RELATION_SELECTION_WITH_HISTORY,
            "category": TemplateCategory.RELATION,
            "required_params": ["question", "entity", "history", "relations", "max_selection_count"],
            "description": "Select relations based on history."
        },
        {
            "name": "reasoning",
            "template": REASONING,
            "category": TemplateCategory.REASONING,
            "required_params": ["question", "entity", "exploration_history"],
            "description": "Reasoning based on knowledge graph exploration."
        },
        {
            "name": "fallback_answer",
            "template": FALLBACK_ANSWER,
            "category": TemplateCategory.REASONING,
            "required_params": ["question", "exploration_history"],
            "description": "Fallback answer template when standard search fails."
        },
    ]
    
    _templates: Dict[str, PromptTemplate] = {
        definition["name"].lower(): PromptTemplate(**definition)
        for definition in _DEFINITIONS
    }

    def __init__(self):
        pass

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        return self._templates.get(template_name.lower())

    def list_templates(self, category: Optional[TemplateCategory] = None) -> List[str]:
        if category:
            return [
                name for name, template_obj in self._templates.items()
                if template_obj.category == category
            ]
        return list(self._templates.keys())

    def format_template(self, template_name: str, **kwargs) -> str:
        template_obj = self.get_template(template_name)
        if not template_obj:
            raise ValueError(f"Template '{template_name}' not found. Available templates: {self.list_templates()}")

        required = set(template_obj.required_params)
        provided = set(kwargs.keys())
        missing = required - provided
        
        if missing:
            raise ValueError(f"Missing required parameters for template '{template_name}': {sorted(list(missing))}")

        try:
            return template_obj.template.format(**kwargs)
        except KeyError as e:
            # This helps debug cases where the template has a placeholder not listed in required_params
            raise ValueError(f"Formatting error in template '{template_name}': placeholder {e} has no value.")

    def __getitem__(self, template_name: str) -> str:
        """Allows getting the raw template string using dictionary-style access, e.g., templates['reasoning']"""
        template_obj = self.get_template(template_name)
        if template_obj:
            return template_obj.template
        raise KeyError(f"Template '{template_name}' not found.")