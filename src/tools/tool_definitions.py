# src/tools/definitions.py

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
from pathlib import Path
from langchain_core.tools import tool

# import llm
from src.core.llms import (
    fast_llm,
    general_llm,
    quality_llm,
    mem,
)
from src.prompts.prompts import (
    ASK_PROMPT, SUMMARY_EXP_PROMPT, SUMMARY_PROMPT, 
    REVISE_JSON_PROMPT, REVISE_TEXT_PROMPT,
)
from src.capabilities.retrieval.knowledge_sources import default_retriever, search_pubmed, Protocol_search, Web_search

from src.capabilities.verification.engine import (
    unified_reflector,
    validate_machine_code as validator_check,
    fix_machine_code_core as fixer_fix,
    ProtocolValidator,
)
from src.capabilities.automation.parameter_processing import parameter_builder
from src.prompts.prompts import build_alignment_prompt, build_paint_prompt


_TOOL_DESCRIPTIONS_PATH = Path(__file__).with_name("tool_descriptions.json")


def _load_tool_descriptions() -> str:
    if not _TOOL_DESCRIPTIONS_PATH.exists():
        return ""
    try:
        with _TOOL_DESCRIPTIONS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return ""
        return data.get("detailed", "") or data.get("concise", "")
    except Exception:
        return ""



def _extract_simplified_flow(content: str) -> Optional[Dict]:
    """Parse experiment flow JSON"""
    try:
        # 1. First attempt to extract content within <exp_flow> tags
        match = re.search(r'<exp_flow>(.*?)</exp_flow>', content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = content.strip()
        
        # 2. Remove markdown code blocks
        if json_str.startswith("```"):
            json_str = re.sub(r'^```json?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
        
        # 3. Parse JSON
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        print(f" JSON parsing failed: {e}")
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                return json.loads(json_str)
        except:
            pass
        return None
    except Exception as e:
        print(f" parsing failed: {e}")
        return None


# --- Tool Definitions ---
@tool
def chat_response(query: str, session_id: str) -> str:
    """
    Call this tool to handle small talk and non-task-related conversations
    parameters: query: str, session_id: str
    """
    prompt = f"""You are ProAgent, a friendly and professional biological experiment planning assistant.
    The user is currently making small talk or asking a casual question鈥攑lease respond naturally.
    If the topic can be directed toward experiment-related matters, you may appropriately ask if they need help planning an experiment.

    User: {query}

    Response (concise and friendly):"""
    
    response = fast_llm.invoke(prompt)
    return response.content.strip()


@tool
def ask_user_confirmation(question: str) -> str:
    """
    Call this tool when user confirmation is needed for the next step, the user will only answer yes or no, without detailed information.
    It will pause the process, present the question to the user, and receive the user's answer (Yes/No).
    parameters: question: str
    """

    print(f"\n[Interaction Step] ProAgent requests confirmation: {question}")
    print("   (The Agent is waiting for your input. Please enter 'yes' to continue, or 'no' to skip)")
    
    user_answer = input("   >> Your choice (y/n): ").strip().lower()
    
    if user_answer in ["y", "yes", "ok"]:
        return "User_Confirmed: YES"
    else:
        return "User_Confirmed: NO"


@tool
def clarify_experiment_scope(query: str, doc_content: Optional[str] = None) -> str:
    """
    Call this tool to clarify the specific requirements of the experiment with the user before starting complex tasks.
    It will generate follow-up questions, obtain the user's answers, and summarize a complete set of experiment information (exp_info) and search keywords.

    [Note]: This tool will pause execution and wait for user input. The Planner should use it as the first step (if necessary).
    parameters: query: str, doc_content: Optional[str] = None
    """
    print("--- Tool: Clarify Experiment Scope ---")

    prompt = ASK_PROMPT.format(query=query, document=doc_content or "")
    question = fast_llm.invoke(prompt).content
    
    if question.lower().strip() == "no":
        print("The experiment information has been fully completed; no further follow-up is required.")
        exp_info = query
    else:
        print(f"AI follow-up question: {question}")
        answer = input("\n Your answer? ").strip()
        prompt = SUMMARY_EXP_PROMPT.format(query=query, question=question, answer=answer)
        exp_info = fast_llm.invoke(prompt).content
        print(f"Summarized experiment information: {exp_info}")

    prompt = """
    You are a researcher in automated biological experiments.
    Based on the [experiment-related information], generate no more than 3 English keywords for searching, separated only by commas.

    ## Experiment-related Information
    {exp_info}
    """
    keywords_str = fast_llm.invoke(prompt.format(exp_info=exp_info)).content.strip()
    output = {"exp_info": exp_info, "keywords": keywords_str}
    output_str = json.dumps(output, ensure_ascii=False, indent=2)
    return output_str

@tool
def retrieve_knowledge(query: str, keywords: str, session_id: str) -> str:
    """
    Call this tool to retrieve information in parallel from all available sources (local storage, the web, literature, and memory) to support protocol generation.
    parameters: query: str, keywords: str, session_id: str
    """
    print("--- Tool: Retrieve Knowledge ---")
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_mem = executor.submit(mem.search, query=query, user_id=session_id)
        future_local = executor.submit(default_retriever.invoke, query)
        future_pubmed = executor.submit(search_pubmed, keywords, max_results=1)
        future_protocols = executor.submit(Protocol_search, keywords)
        future_web = executor.submit(Web_search, query)
        
        print("... Waiting for all parallel retrievals to complete ...")
        
        try:
            mem_results = future_mem.result()
            if mem_results and mem_results.get("results"):
                results["memory"] = "\n".join(f"- {m['memory']}" for m in mem_results["results"])
            else:
                results["memory"] = "no relevant memories"
        except Exception as e:
            print(f"[!] Parallel retrieval of 'memory' failed: {e}")
            results["memory"] = f"Error: {e}"

        try:
            results["local_db"] = future_local.result()
        except Exception as e:
            print(f"[!] Parallel retrieval of 'local_db' failed: {e}")
            results["local_db"] = f"Error: {e}"
            
        try:
            results["pubmed"] = future_pubmed.result()
        except Exception as e:
            print(f"[!] Parallel retrieval of 'pubmed' failed: {e}")
            results["pubmed"] = f"Error: {e}"

        try:
            results["protocols_io"] = future_protocols.result()
        except Exception as e:
            print(f"[!] Parallel retrieval of 'protocols_io' failed: {e}")
            results["protocols_io"] = f"Error: {e}"

        try:
            results["web_search"] = future_web.result()
        except Exception as e:
            print(f"[!] Parallel retrieval of 'web_search' failed: {e}")
            results["web_search"] = f"Error: {e}"

    print("--- All parallel retrievals have been completed ---")

    output = f"""
    [From Memory]: {results.get("memory")}
    [From Local DB]: {results.get("local_db")}
    [From PubMed]: {results.get("pubmed")}
    [From Protocols.io]: {results.get("protocols_io")}
    [From Web]: {results.get("web_search")}
    """
    return output

@tool
def generate_scientific_draft(query: str, exp_info: str, knowledge: str, doc_content: Optional[str] = None) -> str:
    """
    Call this tool to generate a universal, scientifically rigorous experimental protocol draft (plain-text SOP) based on user requirements and knowledge.
    This stage does not consider automated equipment; only scientific logic is focused on.
    parameters: query: str, exp_info: str, knowledge: str, doc_content: Optional[str] = None
    """
    print("--- Tool: Generate Scientific Draft ---")

    print("Integrating information...")
    summary_prompt_formatted = SUMMARY_PROMPT.format(
        query=query, 
        exp_info=exp_info, 
        online_info=knowledge,
        local_info="", 
        document=doc_content or ""
    )
    protocol_summary = quality_llm.invoke(summary_prompt_formatted).content
    print("--- Scientific draft generation completed ---")
    return protocol_summary

@tool
def reflect_on_protocol(query: str, protocol_text: str, task_type="GEN_SCIENTIFIC") -> dict:
    """
    Call this tool to review the generated [Protocol Text Draft] (Reflector).
    Supports differentiated inspection strategies for multiple task types.
    parameters: query: str, protocol_text: str, task_type="GEN_SCIENTIFIC"
    """
    print(f"--- Tool: Reflect on Protocol [Type: {task_type}] ---")

    full_response = unified_reflector(query, protocol_text, task_type)

    final_conclusion = ""

    if task_type == "GEN_SCIENTIFIC":
        match = re.search(r'<conclusion>(.*?)</conclusion>', full_response, re.DOTALL | re.IGNORECASE)
        if match:
            final_conclusion = match.group(1).strip()
        else:
            final_conclusion = full_response.strip()
    else:
        final_conclusion = full_response.strip()

    clean_status = final_conclusion.replace(".", "").strip().upper()

    pass_indicators = ["DONE", "PASS", "OK"]
    is_passed = any(indicator in clean_status for indicator in pass_indicators)
    
    if not is_passed and len(final_conclusion) < 20 and "suggestion" not in final_conclusion:
        is_passed = clean_status in ["DONE", "PASS", "OK"]
    
    if is_passed:
        return {"status": "pass", "message": "PASS"}
    else:
        return {
            "status": "fail",
            "errors": [
                {
                    "type": "ScientificError",
                    "code": "REFLECTOR_CRITIQUE",
                    "message": final_conclusion
                }
            ]
        }


@tool
def modify_protocol(protocol: str, request: dict) -> str:
    """
    Call this tool to revise the protocol per [Modification Suggestions] (from Reflector/Validator).
    parameters: protocol: str, request: dict
    `request` is a dict with 'status' and 'errors' list.
    """
    print("--- Tool: Modify Protocol ---")

    protocol_clean = protocol.strip()
    if protocol_clean.startswith("```"):
        lines = protocol_clean.split('\n')
        if len(lines) >= 2:
            protocol_clean = '\n'.join(lines[1:-1]).strip()
    error_string = ""

    if isinstance(request, str):
        if "not generated" in request:
             print("[Skip] Modification request indicates missing dependency.")
             return protocol
        try:
            request = json.loads(request)
        except:
            error_string = request
    
    if isinstance(request, dict):
        errors = request.get("errors", [])
        if not errors and "message" in request:
            errors = [request["message"]]
        if isinstance(errors, list):
            error_parts = []
            for e in errors:
                msg = str(e.get('message', e)) if isinstance(e, dict) else str(e)
                if msg and len(msg) > 5 and "not generated" not in msg:
                    error_parts.append(msg)
            error_string = "\n".join(error_parts)
        else:
            error_string = str(errors)
    else:
        error_string = str(request)
    
    if not error_string or len(error_string.strip()) < 5 or "No explicit modification" in error_string:
        print(f"   [Modification] No valid suggestions found. Skipping revision.")
        return protocol

    print(f"   [Modification Suggestions]: {error_string[:100]}...")
    
    if protocol_clean.startswith("{") or protocol_clean.startswith("<exp_flow>"):
        tmpl = REVISE_JSON_PROMPT
    else:
        tmpl = REVISE_TEXT_PROMPT
        
    prompt = tmpl.format(
        protocol=protocol,
        exp_flow=protocol,
        query="[N/A]", 
        suggestion=error_string
    )
    return quality_llm.invoke(prompt).content

@tool
def align_draft_to_automation(draft: str, exp_info: str, doc_content: Optional[str] = None) -> str:
    """
    Call this tool to convert the scientific protocol into detailed step-by-step text aligned with laboratory automation equipment.

    This tool labels each step with an execution type:
    - [AUTO]: Executable on the current automation platform
    - [EXTERNAL]: Requires external equipment (available in the laboratory but not on the automation platform)
    - [MANUAL]: Requires manual operation

    parameters: draft: str, exp_info: str, doc_content: Optional[str] = None
    Output: A standardized, line-by-line descriptive string, serving as input for JSON canvas generation in the next step.
    """
    print("--- Tool: Protocol to Equipment Alignment ---")
    draft = str(draft) if draft else ""
    exp_info = str(exp_info) if exp_info else ""
    doc_content = str(doc_content) if doc_content else ""
    
    prompt = build_alignment_prompt(
        exp_info=exp_info,
        protocol=draft,
        document=doc_content
    )

    aligned_protocol = quality_llm.invoke(prompt).content
    
    print("--- Equipment Alignment Completed ---")
    print(f"  - Execution Type Label: [AUTO]/[EXTERNAL]/[MANUAL]")
    
    return aligned_protocol


@tool
def generate_machine_code(aligned_protocol: str, exp_info: str = "", suggestion: str = "[N/A]", session_id: str = "") -> str:
    """
    Call this tool to generate Machine Code via a Two-Stage Generation
    parameters: aligned_protocol: str, exp_info: str = "", suggestion: str = "[N/A]", session_id: str = ""
    """
    print("--- Tool: Generate Machine Code ---")
    prompt = build_paint_prompt(
        protocol=aligned_protocol,
        exp_info=exp_info,
        suggestion=suggestion
    )

    response = quality_llm.invoke(prompt)
    raw_output = response.content.strip()

    simplified_flow = _extract_simplified_flow(raw_output)

    if not simplified_flow:
        print(" LLM output parsing failed, attempting to use raw output directly")
        if "<exp_flow>" in raw_output:
            return raw_output
        return f"<exp_flow>\n{raw_output}\n</exp_flow>"

    print("--- Stage 2: Parameter Population ---")
    
    try:
        full_flow = parameter_builder.build_experiment_flow(simplified_flow)
        
        auto_count = sum(1 for n in full_flow.get("nodes", []) if n.get("resourceId", -1) != -1)
        manual_count = len(full_flow.get("nodes", [])) - auto_count
        
        print(f"  - Automated Nodes: {auto_count}")
        print(f"  - Manual/External Nodes: {manual_count}")
        
    except Exception as e:
        print(f" Parameter population failed: {e}")
        full_flow = simplified_flow

    result = f"<exp_flow>\n{json.dumps(full_flow, indent=2, ensure_ascii=False)}\n</exp_flow>"
    print("--- Machine Code Generation Completed ---")
    return result


@tool
def validate_machine_code(exp_flow_json: str) -> dict:
    """
    Call this tool to perform technical validation on the machine-generated JSON.

    Validation items:
    1. Correctness of JSON format
    2. Validity of resourceId for automated nodes
    3. Completeness of the connection graph (no isolated nodes, no cycles)
    4. Completeness of parameters for external/manual nodes

    Return:
    - status: "success" / "fail"
    - manual_steps: List of node IDs requiring human intervention
    - external_steps: List of node IDs requiring external equipment
    - errors: List of error messages (if failed)

    parameters: exp_flow_json: str
    """
    print("--- Tool: Technical Validation ---")
    is_valid, message, violations = validator_check(exp_flow_json, verbose=True, check_rules=True)
    
    if not is_valid:
        error_code = "VALIDATION_ERROR"
        if "ResourceError" in message:
            error_code = "ResourceError"
        elif "ConnectionError" in message:
            error_code = "ConnectionError"
        elif "SchemaError" in message:
            error_code = "SchemaError"
        elif "RuleError" in message:
            error_code = "RuleError"
        
        return {
            "status": "fail",
            "errors": [{
                "type": "TechnicalError",
                "code": error_code,
                "message": message
            }],
            "rule_violations": [
                {"rule_id": v.rule_id, "severity": v.severity.value, "message": v.message}
                for v in violations
            ] if violations else []
        }
    
    try:
        validator = ProtocolValidator()
        data, _ = validator.extract_json(exp_flow_json)
        
        if data:
            manual_steps = []
            external_steps = []
            
            for node in data.get("nodes", []):
                rid = node.get("resourceId")
                node_id = node.get("templateNodeId")
                
                if rid == -1:
                    params = node.get("parameters", {})
                    op_type = params.get("operationType", "manual")
                    
                    if op_type == "external":
                        external_steps.append({
                            "nodeId": node_id,
                            "deviceName": params.get("deviceName", "External Equipment")
                        })
                    else:
                        manual_steps.append({
                            "nodeId": node_id,
                            "description": params.get("description", params.get("notes", ""))
                        })
            
            nodes = data.get("nodes", [])
            return {
                "status": "success",
                "manual_steps": manual_steps,
                "external_steps": external_steps,
                "rule_warnings": [
                    {"rule_id": v.rule_id, "message": v.message}
                    for v in violations if v.severity.value == "warn"
                ] if violations else [],
                "summary": {
                    "total_nodes": len(nodes),
                    "auto_nodes": len(nodes) - len(manual_steps) - len(external_steps),
                    "external_nodes": len(external_steps),
                    "manual_nodes": len(manual_steps)
                }
            }
    except Exception as e:
        print(f" Post-processing Error: {e}")
    
    return {"status": "success"}
        
@tool
def fix_machine_code(machine_code: str, errors: str = "", session_id: str = "") -> str:
    """
    Call this tool to repair machine code

    Repair content:
    1. ResourceId errors (invalid IDs mapped to valid IDs or -1)
    2. Missing required params
    3. Format issues (time format, etc.)
    4. Param completion for external/manual nodes
    parameters: machine_code: str, errors: str = "", session_id: str = ""
    """
    print("--- Tool: Intelligent Machine Code Repair ---")
    is_valid, message, fixed_json = fixer_fix(machine_code)
    
    if is_valid:
        print(f"[Success] {message}")
    else:
        print(f"[Error] {message}")
    
    return fixed_json

        
@tool
def add_memory(content: str, role: str, session_id: str) -> str:
    """
    Call this tool at the end of the task to store key information (e.g., the final protocol) in long-term memory (Memo).
    The Planner should save the final product (JSON or SOP) after all validations are passed.
    parameters: content: str, role: str, session_id: str
    """
    print(f"--- Tool: Add Memory (add_memory) [Role: {role}] ---")

    try:
        add_messages = [{"role": role, "content": content}]
        mem.add(add_messages, user_id=session_id)
        return "Memory added successfully"
    except Exception as e:
        return f"Failed to add memory: {str(e)}"

tool_list = [
    chat_response,
    clarify_experiment_scope,
    retrieve_knowledge,
    generate_scientific_draft,
    ask_user_confirmation,
    align_draft_to_automation,
    reflect_on_protocol,
    modify_protocol,
    generate_machine_code,
    validate_machine_code,
    fix_machine_code,
    add_memory
]


def get_tools_description() -> str:
    """Generate tool description"""
    curated = _load_tool_descriptions()
    if curated:
        return curated

    desc = ""
    for tool in tool_list:
        desc += f"{tool.name}: {tool.description}\n"
    return desc
