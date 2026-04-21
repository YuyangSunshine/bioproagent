"""Public prompt templates (generic, privacy-preserving)."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


ASK_PROMPT = ChatPromptTemplate.from_template(
    """
You are a biological automation assistant.
Given the user query and optional document, decide whether clarification is needed.
If details are sufficient, output exactly: no
Otherwise output up to 3 concise clarification questions.

User Query:
{query}

Reference Document:
{document}
"""
)


SUMMARY_EXP_PROMPT = """
Summarize user intent into a concise and executable experiment specification.
Use first-person narrative and include only user-provided facts.

Initial Query:
{query}

Clarification Questions:
{question}

User Answers:
{answer}
"""


SUMMARY_PROMPT = """
Generate a clear step-by-step scientific protocol draft in English.
Prioritize user intent and available evidence.
Avoid proprietary device assumptions.

User Requirements:
{query}

Experiment Info:
{exp_info}

Retrieved Information:
{online_info}

Local Information:
{local_info}

Reference Document:
{document}
"""


REVISE_JSON_PROMPT = """
Revise the workflow JSON based on user requirements.
Output only JSON enclosed by <exp_flow> ... </exp_flow>.

Current Workflow:
{exp_flow}

User Request:
{query}
"""


REVISE_TEXT_PROMPT = """
Revise the protocol text according to revision suggestions.
Output only the full revised protocol text.

Protocol:
{protocol}

Revision Suggestions:
{suggestion}
"""


ALIGNMENT_PROMPT = """
Convert the scientific protocol into automation-oriented steps.
Use tags:
- [AUTO] for platform-capable operations
- [EXTERNAL] for out-of-platform equipment
- [MANUAL] for human operations

Do not assume any private device catalog.
Users must configure mapping rules in their own deployment.

Experiment Info:
{exp_info}

Protocol Draft:
{protocol}

Reference Document:
{document}
"""


def build_alignment_prompt(exp_info: str, protocol: str, document: str = '') -> str:
    return ALIGNMENT_PROMPT.format(
        exp_info=exp_info or 'None',
        protocol=protocol or 'None',
        document=document or 'None',
    )


PAINT_PROMPT = """
Generate executable workflow JSON from aligned protocol.
Use a generic schema with fields: nodes, consumables, connections.
Do not rely on private hardware templates.

Aligned Protocol:
{protocol}

Experiment Info:
{exp_info}

Rectify Suggestion:
{suggestion}

Output only JSON enclosed by <exp_flow> ... </exp_flow>.
"""


def build_paint_prompt(protocol: str, exp_info: str = '', suggestion: str = '[N/A]') -> str:
    return PAINT_PROMPT.format(
        protocol=protocol or 'None',
        exp_info=exp_info or 'None',
        suggestion=suggestion or '[N/A]',
    )
