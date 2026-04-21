# main_evaluation.py

"""
ProAgent - State Machine-Based Adaptive Biological Experiment Planning Agent
Core Improvements:
1. State machine-driven planning cycle
2. Semantic data references (e.g., $draft, $knowledge)
3. Cross-Loop memory persistence
4. Comprehensive error handling
"""

import json
import uuid
import threading
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.render import render_text_description

from src.core.llms import mem, general_llm
from config import settings
from src.tools.tool_definitions import tool_list, get_tools_description
from src.core.planner_transfer_matrix import AdaptivePlanner, PlannerContext, ExecutionRecord, AgentState


class ExecutionStatus(Enum):
    """Execution Status Enumeration"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    BLOCKED = "blocked"


@dataclass
class SessionState:
    """
    Session State Management

    Key Design Points:
    - mem_episodic: Complete execution history, preserved across Loops
    - mem_work: Semantic data storage, referenced via $key
    - chat_history: Conversation history, used for LLM context
    """
    session_id: str
    chat_history: List = field(default_factory=list)
    mem_episodic: List[ExecutionRecord] = field(default_factory=list)
    mem_work: Dict[str, Any] = field(default_factory=dict)
    global_exp_context: str = "Unknown"
    current_draft: Optional[str] = None
    is_user_confirmed: bool = False
    current_state: AgentState = AgentState.INIT
    state_history: List[str] = field(default_factory=list)
    tool_attempt_counts: Dict[str, int] = field(default_factory=dict)
    draft_history: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0



class ProAgent:
    """
    ProAgent - Adaptive Biological Experiment Planning Agent

    Core Features:
    1. State machine-driven planning
    2. Multi-step execution per planning cycle
    3. Semantic data references
    4. Comprehensive memory management
    """
    
    def __init__(self, eval_mode=False, ablation_config=None, llm=general_llm, planner_config=None):
        print(">>> Initializing ProAgent ProAgent...")
        print(f">>> RuleEngine limits profile: {settings.RULE_PROFILE}")
        self.eval_mode = eval_mode
        self.ablation_config = ablation_config or {}
        self.planner_config = planner_config or {}
        
        # Core Components
        self.mem = mem
        self.llm = llm
        self.available_tools = {t.name: t for t in tool_list}
        
        # Runtime Configuration
        self.max_loops = 8
        self.max_consecutive_failures = 3
        self.max_tool_retries = 3
        self.oscillation_detection_window = 4
        
        # --- [Ablation Logic] Dynamically Remove Tools ---
        if self.ablation_config.get("no_rag"):
            print("馃毇 [Ablation] Disable RAG (Retrieve Knowledge)")
            if "retrieve_knowledge" in self.available_tools:
                del self.available_tools["retrieve_knowledge"]
                
        if self.ablation_config.get("no_reflect"):
            print("馃毇 [Ablation] Disable Reflection/Validation")
            for t in ["reflect_on_protocol", "modify_protocol", "validate_machine_code", "fix_machine_code"]:
                if t in self.available_tools:
                    del self.available_tools[t]
                    
        if self.ablation_config.get("no_clarify"):
            print("馃毇 [Ablation] Disable Clarification")
            if "clarify_experiment_scope" in self.available_tools:
                del self.available_tools["clarify_experiment_scope"]
        
        
        # Semantic Key Mapping
        self.tools_description = get_tools_description()
        
        # Initialize Planner
        self.planner = AdaptivePlanner(self.tools_description, llm=self.llm)
        self.planner.require_confirmation_before_code = bool(
            self.planner_config.get("require_confirmation_before_code", False)
        )
        
        if self.ablation_config.get("no_rag"):
            self.planner.no_rag = True
        if self.ablation_config.get("no_reflect"):
            self.planner.force_skip_reflection = True
            self.planner.force_skip_validator = True
        if self.ablation_config.get("no_clarify"):
            self.planner.force_skip_clarification = True
        
        self.tool_output_keys = {
            "clarify_experiment_scope": "exp_info",
            "retrieve_knowledge": "knowledge",
            "generate_scientific_draft": "draft",
            "reflect_on_protocol": "ks_verification",
            "modify_protocol": "draft",
            "align_draft_to_automation": "aligned_protocol",
            "generate_machine_code": "machine_code",
            "validate_machine_code": "kp_verification",
            "ask_user_confirmation": "user_decision",
            "add_memory": "memory_stored",
            "fix_machine_code": "machine_code",
            "chat_response": "chat_response",
        }
        
        self.output_dir = "outputs"
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.result_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        print("鉁?ProAgent initialization completed")

    def set_confirmation_gate_mode(self, enabled: bool):
        """Enable or disable draft->code confirmation gate (used by evaluation profiles)."""
        self.planner.require_confirmation_before_code = bool(enabled)
    
    # ==================== Session Management ====================
    
    def _create_session(self) -> SessionState:
        """Create a new conversation session"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        print(f"馃摑 New session: {session_id}")
        return SessionState(session_id=session_id)
    
    # ==================== Memory Management ====================
    
    def _mem_long_persist_async(self, session_id: str, messages: List[Dict]):
        """Asynchronously persist long-term memory (M_long) to Memo."""
        if self.eval_mode:  # Skip directly in evaluation mode
            return

        def _task():
            try:
                self.mem.add(messages, user_id=session_id)
            except Exception as e:
                print(f"鈿狅笍 [Memory] Write failed: {e}")

        threading.Thread(target=_task, daemon=True).start()
    
    def _mem_long_retrieve(self, session_id: str, query: str) -> str:
        """Retrieve long-term memory (M_long) for current query."""
        try:
            results = self.mem.search(query, user_id=session_id)

            if isinstance(results, dict):
                data = results.get("results", [])
            elif isinstance(results, list):
                data = results
            else:
                data = []

            if not data:
                return "No relevant historical memory."

            memory_texts = []
            for item in data:
                if isinstance(item, dict) and "memory" in item:
                    memory_texts.append(f"- {item['memory']}")

            return "\n".join(memory_texts) if memory_texts else "No relevant historical memory."

        except Exception as e:
            print(f"鈿狅笍 [Memory] Retrieval failed: {e}")
            return "The memory system is temporarily unavailable."

    
    # ==================== Parameter Parsing ====================
    
    def _psi_resolve_action(
        self, 
        args: Dict, 
        mem_work: Dict[str, Any],
        local_step_outputs: Dict[int, Any]
    ) -> Tuple[Dict, List[str]]:
        """
        Resolve parameter references
        
        Supported reference formats:
        - $draft: Reference current draft
        - $knowledge: Reference retrieval results
        - $ks_verification: Reference scientific verification results
        - $kp_verification: Reference physical verification results
        - $step_0, $step_1: Reference step outputs in current planning
        - $exp_info: Reference experiment information
        """
        resolved = {}
        missing_refs = []
        
        for key, value in args.items():
            if not isinstance(value, str) or not value.startswith("$"):
                resolved[key] = value
                continue
            
            ref_key = value[1:]
            resolved_value = None
            
            # 1. Step reference
            if ref_key.startswith("step_"):
                try:
                    step_idx = int(ref_key.split("_")[1])
                    if step_idx in local_step_outputs:
                        resolved_value = local_step_outputs[step_idx]
                except (ValueError, IndexError):
                    pass
            
            # 2. Semantic reference
            if resolved_value is None:
                resolved_value = mem_work.get(ref_key)
                
                # Alias handling
                if resolved_value is None and ref_key == "current_draft":
                    resolved_value = mem_work.get("draft")
            
            # 3. Handle missing references
            if resolved_value is None:
                missing_refs.append(value)

                resolved[key] = f"[Data {value} not yet generated; please execute relevant steps first]"
            else:
                resolved[key] = resolved_value
        
        return resolved, missing_refs

    
    def _validate_and_coerce_args(self, tool_name: str, args: Dict) -> Dict:
        """
        Validate and correct parameter types
        
        Perform type conversion according to the actual Schema requirements of each tool
        """
        coerced = dict(args)

        # Define parameter type requirements for each tool
        # str: Requires string
        # dict: Requires dictionary
        # any: No conversion

        TOOL_PARAM_TYPES = {
            "clarify_experiment_scope": {
                "query": "str",
                "doc_content": "str",
            },
            "retrieve_knowledge": {
                "query": "str",
                "keywords": "str",
            },
            "generate_scientific_draft": {
                "query": "str",
                "exp_info": "str",
                "knowledge": "str",
                "doc_content": "str",
            },
            "reflect_on_protocol": {
                "protocol_text": "str",
                "query": "str",
                "task_type": "str",
            },
            "modify_protocol": {
                "protocol": "str",
                "request": "dict",
            },
            "ask_user_confirmation": {
                "question": "str",
            },
            "align_draft_to_automation": {
                "draft": "str",
                "exp_info": "str",
                "doc_content": "str",
            },
            "generate_machine_code": {
                "aligned_protocol": "str",
                "exp_info": "str",
                "suggestion": "str",
            },
            "validate_machine_code": {
                "exp_flow_json": "str",
            },
            "fix_machine_code": {
                "machine_code": "str",
                "errors": "str",
            },
            "add_memory": {
                "content": "str",
                "role": "str",
            },
        }
        
        param_types = TOOL_PARAM_TYPES.get(tool_name, {})
        
        for param_name, expected_type in param_types.items():
            if param_name not in coerced:
                continue

            value = coerced[param_name]

            if expected_type == "str":
                if not isinstance(value, str):
                    if isinstance(value, dict):
                        coerced[param_name] = json.dumps(value, ensure_ascii=False, indent=2)
                    else:
                        coerced[param_name] = str(value)
                    print(f"鈿狅笍 Type conversion: {param_name} ({type(value).__name__} 鈫?str)")

            elif expected_type == "dict":
                if not isinstance(value, dict):
                    if isinstance(value, str):
                        try:
                            coerced[param_name] = json.loads(value)
                            print(f"鈿狅笍 Type conversion: {param_name} (str 鈫?dict)")
                        except json.JSONDecodeError:
                            coerced[param_name] = {"message": value}
                            print(f"鈿狅笍 Type conversion: {param_name} (str 鈫?dict wrapper)")
                    else:
                        coerced[param_name] = {"value": str(value)}
                        print(f"鈿狅笍 Type conversion: {param_name} ({type(value).__name__} 鈫?dict)")

        return coerced

    # ==================== Tool Execution ====================
    
    def _execute_step(
        self, 
        step_idx: int,
        step_data: Dict, 
        state: SessionState,
        local_step_outputs: Dict[int, Any]
    ) -> ExecutionRecord:
        """Execute Single Node"""
        tool_name = step_data.get("tool_name", "unknown")
        raw_args = step_data.get("args", {})
        
        print(f"\n>> Step {step_idx}: {tool_name}")
        
        if tool_name not in self.available_tools:
            print(f"鉂?Tool does not exist: {tool_name}")
            return ExecutionRecord(
                tool=tool_name,
                status="failed",
                summary=f"Tool '{tool_name}' not found",
                output=None
            )
        
        try:
            # 1. Resolve parameter references
            args, missing_refs = self._psi_resolve_action(
                raw_args, state.mem_work, local_step_outputs
            )
            
            # 2. Check if critical parameters are missing
            if missing_refs:
                critical_params = {
                    "reflect_on_protocol": ["protocol_text"],
                    "modify_protocol": ["protocol"],
                    "validate_machine_code": ["exp_flow_json"],
                    "fix_machine_code": ["machine_code"],
                    "generate_machine_code": ["aligned_protocol"],
                }
                
                critical_missing = []
                for ref in missing_refs:
                    ref_clean = ref.replace("$", "")
                    param_list = critical_params.get(tool_name, [])
                    # Check if any parameter references this missing ref
                    for param_name in param_list:
                        if raw_args.get(param_name) == ref:
                            critical_missing.append(ref)
                            break
                
                if critical_missing:
                    print(f"鈿狅笍 Missing critical data: {critical_missing}")
                    return ExecutionRecord(
                        tool=tool_name,
                        status="blocked",
                        summary=f"Missing critical data: {critical_missing}",
                        output={"error": "missing_dependency", "refs": critical_missing}
                    )
                else:
                    print(f"鈿狅笍 Non-critical data missing: {missing_refs}, continuing execution")
            
            # 3. Type validation and correction
            args = self._validate_and_coerce_args(tool_name, args)


            # 4. Auto-fill session_id
            if "session_id" not in args:
                args["session_id"] = state.session_id


            # 5. Print arguments
            self._print_args(args)


            # 6. Execute tool
            output = self.available_tools[tool_name].invoke(args)


            # 7. Save to current planning outputs
            local_step_outputs[step_idx] = output


            # 8. Update global available data
            self._update_mem_work(tool_name, output, state)


            # 9. Generate summary
            summary = self._generate_output_summary(output)
            print(f"鉁?Result: {summary}")


            # 10. Check tool return status
            if isinstance(output, dict) and output.get("status") == "fail":
                return ExecutionRecord(
                    tool=tool_name,
                    status="failed",
                    summary=output.get("message", output.get("errors", [{}])[0].get("message", "Tool returned fail")),
                    output=output
                )
                
            return ExecutionRecord(
                tool=tool_name,
                status="success",
                summary=summary,
                output=output
            )
            
        except Exception as e:
            import traceback
            print(f"鉂?Execution failed: {e}")
            traceback.print_exc()
            return ExecutionRecord(
                tool=tool_name,
                status="failed",
                summary=str(e),
                output=None
            )
            
    def _format_chat_history(self, history: list) -> str:
        """Format chat history"""
        if not history:
            return ""
        lines = []
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content[:100]}")
        return "\n".join(lines)


    def _print_args(self, args: Dict):
        """Print arguments (truncate long content)"""
        print("  args: {")
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 100:
                v_str = v_str[:100] + "..."
            print(f"    {k}: {v_str}")
        print("  }")


    def _generate_output_summary(self, output: Any) -> str:
        """Generate output summary"""
        if output is None:
            return "(no output)"

        if isinstance(output, dict):
            # Extract key information
            if "status" in output:
                status = output.get("status", "unknown")
                errors = output.get("errors", [])
                if errors:
                    return f"status={status}, errors={len(errors)}"
                return f"status={status}"
            return json.dumps(output, ensure_ascii=False)[:150] + "..."

        output_str = str(output)
        return output_str[:150] + "..." if len(output_str) > 150 else output_str
    
    def _update_mem_work(self, tool_name: str, output: Any, state: SessionState):
        """Update available data"""
        # Use semantic tool name
        key = self.tool_output_keys.get(tool_name, f"tool_{tool_name}")
        state.mem_work[key] = output

        # Print storage confirmation
        output_preview = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
        print(f"馃捑 Stored: {key} = {output_preview}")


        # Special handling
        if tool_name == "clarify_experiment_scope":
            # Parse experiment information
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                    exp_info = parsed.get("exp_info", output)
                except:
                    exp_info = output
            elif isinstance(output, dict):
                exp_info = output.get("exp_info", str(output))
            else:
                exp_info = str(output)

            state.global_exp_context = exp_info[:500]
            state.mem_work["exp_info"] = exp_info  # Ensure storage
            print("馃挕 Experiment background updated")


        elif tool_name in ["generate_scientific_draft", "modify_protocol"]:
            # Update draft
            draft_content = str(output) if output else ""

            # Save history (keep up to 5 versions)
            if state.current_draft:
                state.draft_history.append({
                    "version": len(state.draft_history) + 1,
                    "content": state.current_draft,
                    "source_tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                })
                if len(state.draft_history) > 5:
                    state.draft_history.pop(0)
            
            state.current_draft = draft_content[:5000]
            state.mem_work["draft"] = draft_content
            state.mem_work["current_draft"] = draft_content
            print(f"馃摑 Draft updated (version {len(state.draft_history) + 1})")
        
        elif tool_name == "retrieve_knowledge":
            # Ensure knowledge is stored correctly
            state.mem_work["knowledge"] = output
            print("馃摎 Knowledge stored")


        elif tool_name == "reflect_on_protocol":
            # Store Layer-1 verification (K_s) snapshot and sigma_sci in {-1, 1}
            state.mem_work["ks_verification"] = output
            if isinstance(output, dict):
                status = str(output.get("status", "")).lower()
                state.mem_work["sigma_sci"] = 1 if status in {"pass", "done", "success"} else -1
            else:
                out_str = str(output).lower()
                state.mem_work["sigma_sci"] = 1 if "pass" in out_str or "done" in out_str or "yes" in out_str else -1
            print("馃攳 K_s verification stored")


        elif tool_name == "validate_machine_code":
            # Store Layer-2 verification (K_p) snapshot and sigma_phys in {-1, 1}
            state.mem_work["kp_verification"] = output
            if isinstance(output, dict):
                status = str(output.get("status", "")).lower()
                is_valid = output.get("is_valid")
                passed = status == "success" or is_valid is True
                state.mem_work["sigma_phys"] = 1 if passed else -1
            else:
                out_str = str(output).lower()
                state.mem_work["sigma_phys"] = 1 if "success" in out_str or "valid" in out_str else -1
            print("📐 K_p verification stored")

        elif tool_name == "ask_user_confirmation":
            # Parse user confirmation
            if isinstance(output, dict):
                state.is_user_confirmed = output.get("confirmed", False)
            elif isinstance(output, str):
                lower = output.lower()
                state.is_user_confirmed = ("yes" in lower) or ("y" == lower.strip())
            else:
                state.is_user_confirmed = False
            print(f"馃懁 User confirmation: {state.is_user_confirmed}")


        elif tool_name == "chat_response":
            # Save chat to memory
            self._mem_long_persist_async(
                state.session_id,
                [{"role": "assistant", "content": str(output)}]
            )

    
    def _detect_oscillation(self, state: SessionState, current_state: AgentState) -> bool:
        """Detect state oscillation"""
        state.state_history.append(current_state.value)

        # Only look at the recent window
        recent = state.state_history[-self.oscillation_detection_window:]

        if len(recent) >= self.oscillation_detection_window:
            # Detect A-B-A-B pattern
            unique_states = set(recent)
            if len(unique_states) <= 2:
                print(f"鈿狅笍 State oscillation detected: {recent}")
                return True

        return False


    def _check_tool_retry_limit(self, state: SessionState, tool_name: str) -> bool:
        """Check tool retry count limit"""
        count = state.tool_attempt_counts.get(tool_name, 0)
        if count >= self.max_tool_retries:
            print(f"鈿狅笍 Tool {tool_name} has retried {count} times, skipping")
            return True
        return False
    
    # ==================== Termination Conditions ====================
    
    def _should_terminate(
        self, 
        tool_name: str, 
        output: Any, 
        state: SessionState,
        current_state: AgentState
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine whether to terminate the loop
        
        Returns:
            (Whether to terminate, Termination reason)
        """
        # 1. Explicit termination tools
        if tool_name == "add_memory":
            return True, "task_completed"

        if current_state == AgentState.CHATTING and tool_name == "chat_response":
            return True, "chat_completed"

        # 2. Validation passed with no errors
        if tool_name in ["reflect_on_protocol", "validate_machine_code"]:
            if isinstance(output, dict):
                status = output.get("status", "")
                errors = output.get("errors", [])

                # Validation passed
                if status == "success" and not errors:
                    # For reflect_on_protocol: should ask user if they want to continue
                    if tool_name == "reflect_on_protocol":
                        return False, None  # Continue execution, let planner decide next step
                    # For validate_machine_code: task completed
                    return True, "code_validated"

        # 3. User declines to continue
        if tool_name == "ask_user_confirmation":
            if isinstance(output, dict) and not output.get("confirmed", True):
                return True, "user_declined"
            elif isinstance(output, str) and ("no" in output.lower()):
                return True, "user_declined"

        # 4. State judgment
        if current_state == AgentState.SUCCESS:
            return True, "state_completed"

        return False, None
    
    # ==================== Final Response ====================
    
    def _build_final_response(self, state: SessionState, reason: str) -> str:
        """Build final response"""
        if reason == "task_completed":
            if state.current_draft:
                return f"鉁?Task completed! \n\n**Experiment Plan:**\n{state.current_draft[:3000]}"
            return "鉁?Task completed!"

        elif reason == "chat_response":
            return state.mem_work.get("chat_response", "Hello! How can I assist you?")

        elif reason == "chat_completed":
            chat_output = state.mem_work.get("tool_chat_response", "")
            return chat_output if chat_output else "Hello! How can I assist you?"

        elif reason == "code_validated":
            machine_code = state.mem_work.get("machine_code", "")
            if machine_code:
                return f"鉁?Machine code generated and validated successfully! \n\n```json\n{str(machine_code)[:2000]}\n```"
            return "鉁?Machine code validated successfully!"

        elif reason == "user_declined":
            if state.current_draft:
                return f"Okay, paused. Here's the current plan鈥攍et me know if you want to continue: \n\n{state.current_draft[:2000]}"
            return "Okay, paused. Let me know if you want to continue."

        elif reason == "validation_passed":
            return f"鉁?Plan validated successfully! \n\n{state.current_draft[:2000]}" if state.current_draft else "Plan completed."

        elif reason == "state_completed":
            return self._build_final_response(state, "task_completed")

        else:
            # Default: return current progress
            if state.current_draft:
                return f"Current progress: \n\n{state.current_draft[:2000]}"
            return "Task in progress鈥攑lease continue describing your request."
    
    # ==================== Main Loop ====================
    
    def process_query(self, user_query: str, state: SessionState) -> str:
        """
        Process a single user query
        
        1. New task detection
        2. Oscillation detection
        3. Tool retry limit
        4. Critical parameter missing detection
        5. Empty plan handling optimization
        """

        if self.ablation_config.get("no_rag"):
            user_query += " (CONSTRAINT: External search is DISABLED. You MUST generate the draft directly using your internal knowledge. DO NOT try to use retrieve_knowledge.)"
        
        if self.ablation_config.get("no_clarify"):
            user_query += " (Assume all experiment details are clear. Do not ask for clarification.)"
        # ========================================
        
        # Persist user message
        self._mem_long_persist_async(
            state.session_id, 
            [{"role": "user", "content": user_query}]
        )
        
        # Intent recognition
        chat_history_str = self._format_chat_history(state.chat_history[-6:])
        intent_type, chat_response = self.planner.classify_intent(user_query, chat_history_str)
        
        print(f"馃幆 Intent recognized: {intent_type}")
        
        if intent_type == "chat":
            state.chat_history.append(HumanMessage(content=user_query))
            state.chat_history.append(AIMessage(content=chat_response))
            self._mem_long_persist_async(
                state.session_id,
                [{"role": "assistant", "content": chat_response}]
            )
            return chat_response
        
        
        # Retrieve long-term memory
        mem_long = self._mem_long_retrieve(state.session_id, user_query)


        # Initialize loop variables
        consecutive_failures = 0
        final_answer = None

        print(f"\n馃搳 Available data: {list(state.mem_work.keys())}")


        # === Plan-Execute Loop ===
        for loop_i in range(self.max_loops):
            print(f"\n{'-'*50}")
            print(f"馃攧 [Loop {loop_i + 1}/{self.max_loops}]")
            print(f"{'-'*50}")


            # 1. Build Planner context
            ctx = PlannerContext(
                session_id=state.session_id,
                user_input=user_query,
                experiment_context=state.global_exp_context,
                current_state=state.current_state,
                mem_episodic=state.mem_episodic,
                mem_work=state.mem_work,
                mem_long=mem_long
            )


            # 2. Generate plan
            plan, current_state, step_tokens = self.planner.generate_plan(ctx)
            state.current_state = current_state
            state.total_tokens += step_tokens
            print(f"馃搳 Current accumulated tokens: {state.total_tokens}")


            # 3. 馃攳 Oscillation detection
            if self._detect_oscillation(state, current_state):
                final_answer = self._build_recovery_response(state, "oscillation_detected")
                break


            # 4. Handle empty plan
            if not plan:
                if current_state == AgentState.SUCCESS:
                    final_answer = self._build_final_response(state, "state_completed")
                    break
                
                consecutive_failures += 1
                print(f"鈿狅笍 Empty plan (consecutive failures: {consecutive_failures})")
                
                if consecutive_failures >= self.max_consecutive_failures:
                    final_answer = self._build_recovery_response(state, "max_retries_exceeded")
                    break
                continue
            
            consecutive_failures = 0
            
            # 5. Print plan
            plan_tools = [s.get("tool_name", "?") for s in plan]
            print(f"馃搵 Plan ({len(plan)} steps): {plan_tools}")


            # 6. Execute plan
            local_step_outputs = {}
            loop_should_break = False

            for step_idx, step in enumerate(plan):
                tool_name = step.get("tool_name", "unknown")


                # 馃攳 Check tool retry limit
                if self._check_tool_retry_limit(state, tool_name):
                    print(f"鈴笍 Skip {tool_name} (retry limit exceeded)")
                    continue


                result = self._execute_step(step_idx, step, state, local_step_outputs)


                # Add to execution history
                state.mem_episodic.append(result)


                # Handle execution result
                if result.status == "failed":
                    print(f"鉂?Step {step_idx} failed: {result.summary}")
                    # self._increment_tool_attempt(state, tool_name)


                    # 馃攳 Special handling: If validation fails, try rolling back
                    if tool_name == "modify_protocol" and state.draft_history:
                        print("馃攧 Modification failed, considering rollback...")


                    loop_should_break = True
                    break


                elif result.status == "blocked":
                    print(f"馃毇 Step {step_idx} blocked: {result.summary}")
                    loop_should_break = True
                    break


                # Check termination conditions
                should_stop, reason = self._should_terminate(
                    tool_name,
                    result.output,
                    state,
                    current_state
                )
                
                if should_stop:
                    print(f"馃弫 Terminate: {reason}")
                    final_answer = self._build_final_response(state, reason)
                    loop_should_break = True
                    break
            
            # Check if there's a final answer
            if final_answer:
                break


            # Handle loop interrupts (failure/blocking)
            if loop_should_break:
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    final_answer = self._build_recovery_response(state, "max_retries_exceeded")
                    break
                # Otherwise continue next loop, let planner re-plan
        
        # Ensure return value
        if not final_answer:
            if state.mem_work.get("machine_code"):
                code = state.mem_work["machine_code"]
                final_answer = f"鉁?Processing complete! \n\n```json\n{str(code)[:2000]}\n```"
            elif state.current_draft:
                final_answer = f"Reached max loop count. Current draft: \n\n{state.current_draft[:2000]}"
            else:
                final_answer = "Task incomplete. Reached max loop count. Please try simplifying your request."


        # Update chat history
        state.chat_history.append(HumanMessage(content=user_query))
        state.chat_history.append(AIMessage(content=final_answer))


        # Async storage
        self._mem_long_persist_async(
            state.session_id,
            [{"role": "assistant", "content": final_answer}]
        )


        # Save log
        self._save_session_log(state)


        # Save result
        if state.mem_work.get("machine_code"):
            self._save_result(state, "machine_code", state.mem_work["machine_code"])
        if state.current_draft:
            self._save_result(state, "draft", state.current_draft)


        return final_answer
    
    def _build_recovery_response(self, state: SessionState, reason: str) -> str:
        """Build recovery response"""
        if reason == "oscillation_detected":
            # Return the best current result
            if state.mem_work.get("machine_code"):
                return "鈿狅笍 Code validation failed multiple times. Below is the latest version (may require manual inspection):\n\n" + \
                    str(state.mem_work["machine_code"])[:2000]
            elif state.current_draft:
                return "鈿狅笍 Failed to complete automated code generation. Below is the validated experiment plan:\n\n" + \
                    state.current_draft[:2000]
            else:
                return "鈿狅笍 Encountered difficulties with the task. Please try simplifying the request or describing it step-by-step."

        return "Task terminated abnormally."
    
    # ==================== Interaction Entry ====================
    
    def run_session(self):
        """Main interaction loop"""
        state = self._create_session()

        print("\n" + "="*60)
        print("馃敩 ProAgent - Adaptive Biological Experiment Planning System")
        print("="*60)
        print("Enter your request, or type 'quit' to exit")
        print("Tip: You can enter 'show_state' to view the current state\n")

        while True:
            try:
                user_query = input("馃懁 You: ").strip()

                if not user_query:
                    continue

                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("馃憢 Goodbye!")
                    break

                if user_query.lower() == 'show_state':
                    self._print_state(state)
                    continue

                if user_query.lower() == 'clear':
                    state = self._create_session()
                    print("馃攧 Session has been reset")
                    continue

                response = self.process_query(user_query, state)
                print(f"\n馃 ProAgent: {response}\n")

            except KeyboardInterrupt:
                print("\n馃憢 Goodbye!")
                break
            except Exception as e:
                print(f"鉂?Error occurred: {e}")
                import traceback
                traceback.print_exc()
    
    def _save_session_log(self, state: SessionState):
        """Save session log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{state.session_id}_{timestamp}.json")

        log_data = {
            "session_id": state.session_id,
            "timestamp": timestamp,
            "experiment_context": state.global_exp_context,
            "mem_work": {k: str(v)[:500] for k, v in state.mem_work.items()},
            "mem_episodic": [
                {"tool": r.tool, "status": r.status, "summary": r.summary}
                for r in state.mem_episodic
            ],
            "chat_history": [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in state.chat_history
            ]
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"馃摑 Log saved to: {log_file}")
    
    def _save_result(self, state: SessionState, result_type: str, content: Any):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"{state.session_id}_{result_type}_{timestamp}.json")

        with open(result_file, "w", encoding="utf-8") as f:
            if isinstance(content, dict):
                json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(content))

        print(f"馃捑 Result saved: {result_file}")


    def _print_state(self, state: SessionState):
        """Print current state (for debugging)"""
        print("\n" + "="*40)
        print("馃搳 Current Session State")
        print("="*40)
        print(f"Session ID: {state.session_id}")
        print(f"Experiment Background: {state.global_exp_context[:100]}...")
        print(f"Has Draft: {'Yes' if state.current_draft else 'No'}")
        print(f"M_episodic Steps: {len(state.mem_episodic)}")
        print(f"M_work Keys: {list(state.mem_work.keys())}")
        print("="*40 + "\n")


# ========================================

if __name__ == "__main__":
    agent = ProAgent()
    agent.run_session()


