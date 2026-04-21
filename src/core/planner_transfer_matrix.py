"""State-augmented neuro-symbolic planner.

This module implements a deterministic control layer over probabilistic LLM
reasoning. The planner follows the paper-level FSM abstraction:
FSM = <S, Sigma, A, Delta, pi_theta>, where:
- S: cognitive states (e.g., DESIGN/VERIFY/RECTIFY-oriented runtime states)
- Sigma: boolean context signals extracted from memory and execution traces
  (including sigma_sci from K_s and sigma_phys from K_p)
- A: tool-action plans
- Delta: deterministic transition matrix over Sigma
- pi_theta: neural planner that proposes concrete tool sequences under state
  constraints.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.core.llms import fast_llm
from src.prompts.planner import build_planner_prompt


class AgentState(Enum):
    """Agent states aligned with ACL_26 FSM naming."""
    INIT = "INIT"
    CHATTING = "CHATTING"
    CLARIFY_INTENT = "CLARIFY_INTENT"
    DESIGN_DRAFT = "DESIGN_DRAFT"
    VERIFY_DRAFT = "VERIFY_DRAFT"
    RECTIFY_DRAFT = "RECTIFY_DRAFT"
    DESIGN_CODE = "DESIGN_CODE"
    RECTIFY_CODE = "RECTIFY_CODE"
    SUCCESS = "SUCCESS"


class CognitivePhase(Enum):
    """Paper-level macro cognitive phases for DVR-aligned reporting."""
    DESIGN = "DESIGN"
    VERIFY = "VERIFY"
    RECTIFY = "RECTIFY"
    OTHER = "OTHER"


@dataclass
class ExecutionRecord:
    """Execution Record"""
    tool: str
    status: str
    summary: str
    output: Any = None


@dataclass
class PlannerContext:
    """Planner Context"""
    session_id: str
    user_input: str
    experiment_context: str = ""
    current_state: AgentState = AgentState.INIT
    mem_episodic: List[ExecutionRecord] = field(default_factory=list)
    mem_work: Dict[str, Any] = field(default_factory=dict)
    mem_long: str = ""


@dataclass
class ContextSignals:
    """
    Context Signal Snapshot
    Converts complex raw data into clean boolean signals for state decision-making
    """

    # Signals in the ACL matrix
    sigma_ambiguous: bool = False
    sigma_know: bool = False
    sigma_draft: bool = False
    sigma_sci_pos: bool = False
    sigma_sci_neg: bool = False
    sigma_code: bool = False
    sigma_phys_pos: bool = False
    sigma_phys_neg: bool = False

    # User Intent Signals
    user_confirmed_yes: bool = False
    user_confirmed_no: bool = False
    user_skip_reflection: bool = False     # User requests to skip reflection and generate directly


class AdaptivePlanner:
    """Adaptive Planner with deterministic FSM control over neural planning."""
    KS_KEY = "ks_verification"
    KP_KEY = "kp_verification"
    
    def __init__(self, tools_description: str, llm=fast_llm):
        self.tools_description = tools_description
        self.llm = llm
        self.no_rag = False
        self.require_confirmation_before_code = False
        self.force_skip_reflection = False 
        self.force_skip_validator = False
        self.force_skip_clarification = False
    
    def set_llm(self, llm):
        """Set the LLM"""
        self.llm = llm
    
    def classify_intent(self, user_input: str, chat_history: str = "") -> Tuple[str, str]:
        """Intent Recognition & Chit-Chat Response"""
        prompt = f"""You are a ProAgent, a bio-experiment planning assistant.

## Task
# Check the user's intent and respond accordingly.

## Type of intent
- chat: casual conversation, greeting, ask who are you, any chats without experiments
- task: any request related to experimental design, planning, or execution

## Output format（Please strictly follow）
if it is chat, directly output a friendly reply, and can appropriately guide to the experimental topic.
if it is task, only output: [TASK]

## Chat history
{chat_history if chat_history else "none"}

## User input
{user_input}

## Your response:"""

        response = self.llm.invoke(prompt)
        content = response.content.strip()
        
        if content == "[TASK]" or content.startswith("[TASK]"):
            return "task", ""
        else:
            return "chat", content
            
    def _extract_signals(self, ctx: PlannerContext) -> ContextSignals:
        """
        Signal Extraction Layer
        Responsible for parsing clean boolean signals from context data
        """
        data = ctx.mem_work
        
        raw_context = ctx.experiment_context
        is_valid_context = (
            raw_context 
            and len(raw_context) > 5 
            and "Unknown" not in raw_context 
            and "Not yet confirmed" not in raw_context
        )
        
        history = ctx.mem_episodic
        user_input_lower = ctx.user_input.lower()
        s = ContextSignals()

        # 1. Basic Resource Checks
        has_exp_info = bool(data.get("exp_info") or is_valid_context)
        s.sigma_ambiguous = not has_exp_info
        s.sigma_know = bool(data.get("knowledge"))
        s.sigma_draft = bool(data.get("draft") or data.get("current_draft"))
        s.sigma_code = bool(data.get("machine_code"))

        # 2. Layer-1 Neuro-Scientific Verification (K_s) signals
        
        # A. Prioritize checking the most recent history records
        for record in reversed(history):
            if record.tool == "reflect_on_protocol" and record.status == "success":
                output = record.output
                if isinstance(output, str):
                    if output.strip().lower() == "done":
                        s.sigma_sci_pos = True
                    else:
                        s.sigma_sci_neg = True
                elif isinstance(output, dict):
                    status = output.get("status", "").lower()
                    if status == "pass" or status == "done":
                        s.sigma_sci_pos = True
                    else:
                        s.sigma_sci_neg = True
                break
        
        # B. Supplemental check from M_work[K_s]
        if not s.sigma_sci_pos and not s.sigma_sci_neg:
            validation_data = data.get(self.KS_KEY)
            if isinstance(validation_data, dict):
                status = validation_data.get("status", "").lower()
                if status == "pass" or status == "done" or status == "success":
                    s.sigma_sci_pos = True
                else:
                    s.sigma_sci_neg = True
                
        # 3. Layer-2 Symbolic-Physical Verification (K_p) signals
        
        # A. Check history records
        for record in reversed(history):
            if record.tool == "validate_machine_code" and record.status == "success":
                output = record.output
                if isinstance(output, dict):
                    is_valid = output.get("is_valid")
                    if is_valid is True:
                        s.sigma_phys_pos = True
                    elif is_valid is False:
                        s.sigma_phys_neg = True
                    elif output.get("status") == "success":
                        s.sigma_phys_pos = True
                elif isinstance(output, tuple) or isinstance(output, list):
                     if output[0] is True:
                         s.sigma_phys_pos = True
                     else:
                         s.sigma_phys_neg = True
                break
        
        # B. Check data snapshot
        code_val = data.get(self.KP_KEY)
        if isinstance(code_val, dict):
            if code_val.get("status") == "fail" or code_val.get("is_valid") is False:
                s.sigma_phys_neg = True
            elif code_val.get("status") == "success" or code_val.get("is_valid") is True:
                s.sigma_phys_pos = True

        # 4. User Intent Signals
        for record in reversed(history):
            if record.tool == "ask_user_confirmation":
                output = record.output
                if isinstance(output, dict):
                    confirmed = output.get("confirmed", False)
                    s.user_confirmed_yes = confirmed is True
                    s.user_confirmed_no = confirmed is False
                elif isinstance(output, str):
                    s.user_confirmed_yes = "yes" in output.lower() 
                    s.user_confirmed_no = not s.user_confirmed_yes
                break
        
        # 5. Skip Reflection Signal
        skip_keywords = ["generate code", "skip reflection", "DESIGN_CODE"]
        if any(k in user_input_lower for k in skip_keywords):
            s.user_skip_reflection = True
        
        if self.force_skip_reflection:
            if s.sigma_draft:
                s.sigma_sci_pos = True
                s.sigma_sci_neg = False
            if s.sigma_code:
                s.sigma_phys_pos = True
                s.sigma_phys_neg = False
        
        if self.force_skip_validator:
            if s.sigma_code:
                s.sigma_phys_pos = True
                s.sigma_phys_neg = False
            
        if self.force_skip_clarification:
            s.sigma_ambiguous = False
        
        return s

    def _infer_state(self, ctx: PlannerContext) -> AgentState:
        """
        Deterministic state inference (Delta) based on extracted signals (Sigma).
        """
        if isinstance(ctx.current_state, AgentState):
            prev_state = ctx.current_state
        elif isinstance(ctx.current_state, str):
            prev_state = AgentState.__members__.get(ctx.current_state, AgentState.INIT)
        else:
            prev_state = AgentState.INIT

        # 1. Get clean signals
        signals = self._extract_signals(ctx)

        # Priority matrix (ACL_26 naming, evaluated top-down).
        decision_matrix: List[Tuple[Callable[[ContextSignals], bool], AgentState]] = [
            # Priority 1: σ_phys- ∧ σ_code -> RECTIFY_CODE
            (lambda s: s.sigma_phys_neg and s.sigma_code, AgentState.RECTIFY_CODE),
            # Priority 2: σ_sci- ∧ σ_draft -> RECTIFY_DRAFT
            (lambda s: s.sigma_sci_neg and s.sigma_draft, AgentState.RECTIFY_DRAFT),
            # Priority 3: ¬σ_draft ∧ σ_know -> DESIGN_DRAFT
            (lambda s: (not s.sigma_draft) and s.sigma_know, AgentState.DESIGN_DRAFT),
            # Optional confirmation gate for Subset A style draft-only cutoff.
            (
                lambda s: self.require_confirmation_before_code and s.user_confirmed_no and s.sigma_draft and s.sigma_sci_pos,
                AgentState.SUCCESS,
            ),
            # Priority 4 (gated mode): verified draft waits for explicit user confirmation.
            (
                lambda s: (
                    self.require_confirmation_before_code
                    and s.sigma_draft
                    and s.sigma_sci_pos
                    and (not s.user_confirmed_yes)
                    and (not s.user_skip_reflection)
                ),
                AgentState.VERIFY_DRAFT,
            ),
            # Priority 4: σ_draft ∧ σ_sci+ -> DESIGN_CODE
            (
                lambda s: (
                    s.sigma_draft
                    and s.sigma_sci_pos
                    and (
                        (not self.require_confirmation_before_code)
                        or s.user_confirmed_yes
                        or s.user_skip_reflection
                    )
                ),
                AgentState.DESIGN_CODE,
            ),
            # Priority 5: σ_draft ∧ (σ_sci = 0) -> VERIFY_DRAFT
            (lambda s: s.sigma_draft and (not s.sigma_sci_pos) and (not s.sigma_sci_neg), AgentState.VERIFY_DRAFT),
            # Priority 6: σ_code ∧ σ_phys+ -> SUCCESS
            (lambda s: s.sigma_code and s.sigma_phys_pos, AgentState.SUCCESS),
            # Priority 7: σ_ambiguous -> CLARIFY_INTENT
            (lambda s: s.sigma_ambiguous, AgentState.CLARIFY_INTENT),
        ]

        # 3. Iterate through the matrix to execute decision
        for condition, state in decision_matrix:
            if condition(signals):
                return state
                
        # 4. Default fallback (Priority 8): self-loop at current state s_t
        return prev_state

    def _to_cognitive_phase(self, state: AgentState) -> CognitivePhase:
        """
        Map runtime states to paper-level DVR cognitive phases.
        This is a semantic layer for alignment/reporting and does not alter logic.
        """
        mapping = {
            AgentState.DESIGN_DRAFT: CognitivePhase.DESIGN,
            AgentState.DESIGN_CODE: CognitivePhase.DESIGN,
            AgentState.VERIFY_DRAFT: CognitivePhase.VERIFY,
            AgentState.RECTIFY_DRAFT: CognitivePhase.RECTIFY,
            AgentState.RECTIFY_CODE: CognitivePhase.RECTIFY,
        }
        return mapping.get(state, CognitivePhase.OTHER)
    
    def _get_state_guidance(self, state: AgentState) -> str:
        """Get planning guidance corresponding to the state"""
        plan_start = "Plan: generate_scientific_draft (direct generation)" if self.no_rag else "Plan: retrieve_knowledge → generate_scientific_draft"
        
        guidance = {
            AgentState.CHATTING: "Casual chat mode.",
            AgentState.INIT: "Plan: clarify_experiment_scope or retrieve_knowledge.",
            AgentState.CLARIFY_INTENT: "Plan: clarify_experiment_scope",
            AgentState.DESIGN_DRAFT: plan_start,
            AgentState.VERIFY_DRAFT: (
                "Plan: reflect_on_protocol; if pass, ask_user_confirmation"
                if self.require_confirmation_before_code
                else "Plan: reflect_on_protocol"
            ),
            AgentState.RECTIFY_DRAFT: "Plan: modify_protocol -> reflect_on_protocol",
            AgentState.DESIGN_CODE: "Plan: align_draft_to_automation -> generate_machine_code -> validate_machine_code",
            AgentState.RECTIFY_CODE: "Plan: fix_machine_code -> validate_machine_code",
            AgentState.SUCCESS: "Completed",
        }
        return guidance.get(state, "Please plan the next step based on the current state.")
    
    def _get_default_plan(self, state: AgentState, ctx: PlannerContext) -> List[Dict]:
        """Get the default plan based on the current state"""
        user_input = ctx.user_input
        
        if self.no_rag:
            plan_start = [
                {"tool_name": "generate_scientific_draft", "args": {"query": user_input, "exp_info": "$exp_info", "knowledge": "Internal Knowledge (RAG Disabled)"}},
                {"tool_name": "reflect_on_protocol", "args": {"protocol_text": "$step_0", "query": user_input}}
            ]
        else:
            plan_start = [
                {"tool_name": "retrieve_knowledge", "args": {"query": user_input, "keywords": ""}},
                {"tool_name": "generate_scientific_draft", "args": {"query": user_input, "exp_info": "$exp_info", "knowledge": "$step_0"}},
                {"tool_name": "reflect_on_protocol", "args": {"protocol_text": "$step_1", "query": user_input}}
            ]
        
        default_plans = {
            AgentState.INIT: [
                {"tool_name": "clarify_experiment_scope", "args": {"query": user_input}}
            ],
            AgentState.CLARIFY_INTENT: [
                {"tool_name": "clarify_experiment_scope", "args": {"query": user_input}}
            ],
            AgentState.DESIGN_DRAFT: plan_start,
            AgentState.VERIFY_DRAFT: [
                {"tool_name": "reflect_on_protocol", "args": {"protocol_text": "$step_0", "query": user_input}}
            ],
            AgentState.RECTIFY_DRAFT: [
                {"tool_name": "modify_protocol", "args": {"protocol": "$draft", "request": f"${self.KS_KEY}"}},
                {"tool_name": "reflect_on_protocol", "args": {"protocol_text": "$step_0", "query": user_input}}
            ],
            AgentState.DESIGN_CODE: [
                {"tool_name": "align_draft_to_automation", "args": {"draft": "$draft", "exp_info": "$exp_info"}},
                {"tool_name": "generate_machine_code", "args": {"aligned_protocol": "$step_0"}},
                {"tool_name": "validate_machine_code", "args": {"exp_flow_json": "$step_1"}},
                {"tool_name": "add_memory", "args": {"content": "Machine code Generated!", "role": "assistant"}}
            ],
            AgentState.RECTIFY_CODE: [
                {"tool_name": "fix_machine_code", "args": {"machine_code": "$machine_code", "errors": f"${self.KP_KEY}"}},
                {"tool_name": "validate_machine_code", "args": {"exp_flow_json": "$step_0"}}
            ],
            AgentState.SUCCESS: [],
        }

        if state == AgentState.VERIFY_DRAFT:
            validation = ctx.mem_work.get(self.KS_KEY)
            if (
                self.require_confirmation_before_code
                and isinstance(validation, dict)
                and str(validation.get("status", "")).lower() in {"pass", "done", "success"}
            ):
                return [{"tool_name": "ask_user_confirmation", "args": {"question": "Scientific draft passed review. Proceed to DESIGN_CODE?"}}]
            return [{"tool_name": "reflect_on_protocol", "args": {"protocol_text": "$draft", "query": user_input}}]

        return default_plans.get(state, [{"tool_name": "clarify_experiment_scope", "args": {"query": user_input}}])
    
    def generate_plan(self, ctx: PlannerContext) -> Tuple[List[Dict], AgentState, int]:
        """Generate an execution plan"""
        # 1. Infer current state
        current_state = self._infer_state(ctx)
        cognitive_phase = self._to_cognitive_phase(current_state)
        guidance = self._get_state_guidance(current_state)
        print(f"📊 [State] Current state: {current_state.value}")
        print(f"🧠 [Cognitive Phase] {cognitive_phase.value}")
        print(f"📌 [Guidance] {guidance}")
        
        # 2. If completed
        if current_state == AgentState.SUCCESS:
            return [], current_state, 0

        if current_state == AgentState.CHATTING:
            return self._get_default_plan(current_state, ctx), current_state, 0
        
        # 3. Attempt to call LLM to generate plan
        try:
            prompt = build_planner_prompt(
                user_input=ctx.user_input,
                experiment_context=ctx.experiment_context,
                mem_episodic=[
                    {"tool": r.tool, "status": r.status, "summary": r.summary}
                    for r in ctx.mem_episodic[-10:]
                ],
                mem_work=ctx.mem_work,
                mem_long=ctx.mem_long,
                tools_description=self.tools_description,
                session_id=ctx.session_id,
                current_state=current_state.value,
                state_guidance=guidance
            )
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            token_usage = 0
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                token_usage = usage.get('total_tokens', 0)
            
            print(f"🤖 [LLM Response] Tokens: {token_usage}")
            print(f"🤖 [LLM Response] {content[:500]}...")
            
            plan = self._parse_plan(content)
            
            if plan:
                plan = self._post_process_plan(plan, ctx)
                return plan, current_state, token_usage
            else:
                print(f"⚠️ [Fallback] use default plan")
                return self._get_default_plan(current_state, ctx), current_state, 0
                
        except Exception as e:
            print(f"❌ [Planner] LLM call failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_plan(current_state, ctx), current_state, 0
    
    def _parse_plan(self, content: str) -> List[Dict]:
        """
        Parse LLM output
        """

        def robust_loads(json_str):
            try:
                # 1. Try standard parsing
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 2. Fix logic: LLMs often use raw newlines in JSON string values (invalid syntax)
                try:
                    fixed_str = json_str.replace('\\n', '___ESCAPED_NEWLINE___')
                    fixed_str = fixed_str.replace('\n', '\\n')
                    fixed_str = fixed_str.replace('___ESCAPED_NEWLINE___', '\\n')
                    
                    return json.loads(fixed_str)
                except Exception:
                    raise 

        # 1. Try direct parsing
        try:
            result = robust_loads(content)
            if isinstance(result, list):
                print(f"✓ [Parse] Direct parsing succeeded")
                return result
        except Exception as e:
            print(f"⚠️ [Parse] Direct parsing failed: {e}")
            pass
        
        # 2. Try extracting Markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)```',
            r'```\s*([\s\S]*?)```',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    result = robust_loads(match.group(1).strip())
                    if isinstance(result, list):
                        print(f"✓ [Parse] Parsed successfully from code block")
                        return result
                except Exception as e:
                    print(f"⚠️ [Parse] Code block parsing failed: {e}")
        
        # 3. Try extracting the outermost list []
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                json_candidate = content[start : end+1]
                result = robust_loads(json_candidate)
                if isinstance(result, list):
                    print(f"✓ [Parse] Extracted [] successfully")
                    return result
        except Exception as e:
            print(f"⚠️ [Parse] Extracting [] failed: {e}")
        
        # 4. 兜底处理
        if "[]" in content or content.strip() == "[]":
            print(f"✓ [Parse] Empty array")
            return []
        
        print(f"❌ [Parse] All methods failed. First 100 chars of raw content: {content[:100]}...")
        return []
    
    def _post_process_plan(self, plan: List[Dict], ctx: PlannerContext) -> List[Dict]:
        """Post-process plan - Compatible with multiple LLM output formats"""
        processed = []
    
        for step in plan:
            if not isinstance(step, dict):
                continue
            
            tool_name = step.get("tool_name") or step.get("tool") or step.get("name")
            if not tool_name:
                continue
            
            args = step.get("args") or step.get("input") or step.get("parameters") or {}
            if not isinstance(args, dict):
                args = {}
            
            args["session_id"] = ctx.session_id
            
            # =========================================================
            # Rule 1: Reflection requires reference to Draft or Revision
            # =========================================================
            if tool_name == "reflect_on_protocol":
                for prev_idx in range(len(processed) - 1, -1, -1):
                    if processed[prev_idx]["tool_name"] in ["modify_protocol", "generate_scientific_draft"]:
                        args["protocol_text"] = f"$step_{prev_idx}"
                        break
                else:
                    if "draft" in ctx.mem_work or "current_draft" in ctx.mem_work:
                        args["protocol_text"] = "$draft"
                        
            # =========================================================
            # Rule 2: Modification requires reference to Reflection feedback
            # =========================================================
            if tool_name == "modify_protocol":
                protocol_arg = args.get("protocol", "")
                if isinstance(protocol_arg, str) and len(protocol_arg) > 50:
                    args["protocol"] = "$draft"
                
                if not args.get("request"):
                     args["request"] = "$last_tool_output(reflect_on_protocol)"

            # =========================================================
            # Rule 3: Validation requires reference to Code Gen or Fix results
            # =========================================================
            if tool_name == "validate_machine_code":
                for prev_idx in range(len(processed) - 1, -1, -1):
                    if processed[prev_idx]["tool_name"] in ["generate_machine_code", "fix_machine_code"]:
                        args["exp_flow_json"] = f"$step_{prev_idx}"
                        break
                else:
                    if "machine_code" in ctx.mem_work:
                        args["exp_flow_json"] = "$machine_code"

            # =========================================================
            # Rule 4: Code Gen requires reference to Alignment results
            # =========================================================
            if tool_name == "generate_machine_code":
                for prev_idx in range(len(processed) - 1, -1, -1):
                    if processed[prev_idx]["tool_name"] == "align_draft_to_automation":
                        args["aligned_protocol"] = f"$step_{prev_idx}"
                        break
                else:
                    if "aligned_protocol" in ctx.mem_work:
                        args["aligned_protocol"] = "$aligned_protocol"
                
                if self.KP_KEY in ctx.mem_work:
                    validation = ctx.mem_work[self.KP_KEY]
                    if isinstance(validation, dict) and (validation.get("status") == "fail" or validation.get("is_valid") is False):
                        args["suggestion"] = f"${self.KP_KEY}"

            # =========================================================
            # Rule 6: Fix requires reference to existing Code and Errors
            # =========================================================
            if tool_name == "fix_machine_code":
                if "machine_code" in ctx.mem_work:
                    args["machine_code"] = "$machine_code"
                if self.KP_KEY in ctx.mem_work:
                    args["errors"] = f"${self.KP_KEY}"
                    
            processed.append({
                "tool_name": tool_name,
                "args": args
            })
            
        return processed
