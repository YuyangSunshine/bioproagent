# src/prompts/planner.py

"""
ProAgent planner prompt utilities.

This module explicitly mirrors the paper memory notation:
- Phi(M_work): projection of working memory into lightweight previews
- tau_t from M_episodic: recent trajectory summary
- Retrieved(M_long): retrieved long-term memory snippets
"""


def _phi_project_working_memory(mem_work: dict) -> str:
    """Phi(M_work): project key-value artifacts into lightweight (k, preview(v))."""
    if not mem_work:
        return "Empty M_work"

    lines = []
    for key, value in mem_work.items():
        value_str = str(value).replace("\n", " ")
        preview = value_str[:80] + "..." if len(value_str) > 80 else value_str
        lines.append(f"(${key}, {preview})")
    return "\n".join(lines)


def _summarize_tau(mem_episodic: list) -> str:
    """Summarize tau_t from episodic memory as recent (action, observation) tuples."""
    if not mem_episodic:
        return "Empty M_episodic"

    traj_lines = []
    for item in mem_episodic[-8:]:
        status_icon = "✓" if item["status"] == "success" else "✗"
        traj_lines.append(f"{status_icon} ({item['tool']}, {item['summary'][:80]})")
    return "\n".join(traj_lines)


def build_planner_prompt(
    user_input: str,
    experiment_context: str,
    mem_episodic: list,
    mem_work: dict,
    mem_long: str,
    tools_description: str,
    session_id: str,
    current_state: str = "INIT",
    state_guidance: str = ""
) -> str:
    """Build Planner Prompt"""
    tau_summary = _summarize_tau(mem_episodic)
    phi_mem_work = _phi_project_working_memory(mem_work)
    retrieved_mem_long = mem_long if mem_long else "No relevant M_long retrieved."

    return f"""You are BioProAgent，a bioexperiment planning agent. Your task is to convert user requests into executable tool call sequences.

## Scientific Principles
- Reproducibility: Steps are clear and reproducible
- Control Principle: Negative control (NTC), Positive control (PTC), Blank control
- Variable Control: Change only one variable at a time

## Current State: {current_state}
## Suggested Actions: {state_guidance}

## User Request
{user_input}

## Experiment Context
{experiment_context if experiment_context != "Unknown" else "Waiting to clarify"}

## Retrieved(M_long)
{retrieved_mem_long}

## tau_t from M_episodic
{tau_summary}

## Phi(M_work) = (k, preview(v))
{phi_mem_work}

## Available Tools
{tools_description}

## Output Format Requirements (Strictly Follow)

Output must be a JSON array, each element must contain:
- "tool_name": tool name (string)
- "args": argument object
- "tool_name": tool name (string)
- "args": argument object

OutputExample:
```json
[
  {{"tool_name": "clarify_experiment_scope", "args": {{"query": "User question", "session_id": "{session_id}"}}}},
  {{"tool_name": "retrieve_knowledge", "args": {{"query": "Retrieval content", "session_id": "{session_id}"}}}}
]

### FSM State→Action Mapping
- CLARIFY_INTENT → [clarify_experiment_scope]
- DESIGN_DRAFT → [retrieve_knowledge, generate_scientific_draft]
- VERIFY_DRAFT → [reflect_on_protocol]
- RECTIFY_DRAFT → [modify_protocol, reflect_on_protocol]
- DESIGN_CODE → [align_draft_to_automation, generate_machine_code, validate_machine_code, add_memory]
- RECTIFY_CODE → [fix_machine_code, validate_machine_code]
- SUCCESS → []

## Important Instructions
- Follow scientific principles strictly.
- Current state is **{current_state}**
- Suggested actions: {state_guidance}
- Please plan multiple steps according to the suggestions

Please output a JSON array (no explanation): """
