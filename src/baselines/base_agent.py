# src/baselines/base_agent.py
"""
Base class for all baseline agents
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
import re


@dataclass
class AgentResult:
    """Standardized Agent Output Result"""
    success: bool
    final_output: Any
    trajectory: List[Dict[str, Any]]
    total_trajectory: List[Dict[str, Any]]
    total_steps: int
    total_tokens: int
    total_time: float
    error_message: Optional[str] = None
    loop_detected: bool = False
    constraint_violations: List[str] = field(default_factory=list)
    
    
    # Additional metrics
    draft_generated: bool = False
    code_generated: bool = False
    validation_passed: bool = False


@dataclass 
class StepRecord:
    """Single-Step Execution Record"""
    step_id: int
    action_type: str  # "think", "act", "observe", "reflect"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_output: Optional[Any] = None
    tokens_used: int = 0
    time_elapsed: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all Baseline Agents
    Adapts to ProAgent's tool definitions and LLM interfaces
    """
    
    def __init__(
        self,
        llm,
        tools: Dict[str, Any],
        max_steps: int = 15,
        verbose: bool = True
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Runtime statistics
        self.total_tokens = 0
        self.trajectory: List[StepRecord] = []
        
        # Tool output key mapping (consistent with ProAgent)
        self.tool_output_keys = {
            "clarify_experiment_scope": "exp_info",
            "retrieve_knowledge": "knowledge",
            "generate_scientific_draft": "draft",
            "reflect_on_protocol": "ks_verification",
            "modify_protocol": "draft",
            "align_draft_to_automation": "aligned_protocol",
            "generate_machine_code": "machine_code",
            "validate_machine_code": "kp_verification",
            "fix_machine_code": "machine_code",
        }
        
    @abstractmethod
    def run(self, query: str, **kwargs) -> AgentResult:
        """run Agent"""
        pass
    
    def _call_llm(self, prompt: str) -> Tuple[str, int]:
        """Call LLM and record tokens"""
        start_time = time.time()
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            content = f"Error: {e}"
        
        elapsed = time.time() - start_time
        
        # Estimate token count
        tokens = len(prompt.split()) + len(content.split())
        self.total_tokens += tokens
        
        return content, tokens
    
    def _call_tool(self, tool_name: str, args: Dict, session_id: str = "baseline") -> Any:
        """Call tool - Adapt to ProAgent tool signatures"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found", "status": "fail"}
        
        try:
            if "session_id" not in args:
                args["session_id"] = session_id
            
            result = self.tools[tool_name].invoke(args)
            return result
        except Exception as e:
            return {"error": str(e), "status": "fail"}
    
    def _log(self, message: str):
        """Log output"""
        if self.verbose:
            print(message)
    
    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse action from LLM output"""
        
        # Format 1: Action: tool_name[{"arg": "value"}]
        match = re.search(r'Action:\s*(\w+)\s*\[(.*?)\]', text, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            try:
                if args_str.startswith('{'):
                    args = json.loads(args_str)
                else:
                    args = {"input": args_str}
            except:
                args = {"input": args_str}
            return tool_name, args
        
        # Format 2: ```json {"tool_name": "...", "args": {...}} ```
        json_match = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                return data.get("tool_name") or data.get("tool"), data.get("args", {})
            except:
                pass
        
        # Format 3: Tool: ..., Args: ...
        tool_match = re.search(r'Tool:\s*(\w+)', text, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1)
            args_match = re.search(r'Args?:\s*(\{.*?\})', text, re.DOTALL)
            try:
                args = json.loads(args_match.group(1)) if args_match else {}
            except:
                args = {}
            return tool_name, args
        
        return None, None
    
    def _is_terminal(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if it's a terminal action"""
        terminal_patterns = [
            r'Final Answer:\s*(.*)',
            #r'FINISH\[(.*?)\]',
            r'Task Complete[d]?:\s*(.*)',
        ]
        
        for pattern in terminal_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return True, match.group(1).strip()
        
        return False, None
    
    def _detect_loop(self, window: int = 4) -> bool:
        """Detect if the Agent is stuck in a loop"""
        if len(self.trajectory) < window:
            return False
        
        recent_actions = [
            (r.tool_name, str(r.tool_args)[:30]) 
            for r in self.trajectory[-window:]
            if r.tool_name
        ]
        
        if len(recent_actions) >= 2:
            unique = set(recent_actions)
            if len(unique) <= 2:
                return True
        
        return False
    
    def _check_validation_status(self, output: Any) -> Tuple[bool, str]:
        """Check the output status of the validation tool"""
        if isinstance(output, dict):
            status = output.get("status", "")
            if status == "pass" or status == "success":
                return True, "passed"
            elif status == "fail":
                errors = output.get("errors", [])
                if errors:
                    msg = errors[0].get("message", "") if isinstance(errors[0], dict) else str(errors[0])
                    return False, msg
                return False, "validation failed"
        return True, "unknown"
    
    def _build_tools_description(self) -> str:
        """build tools description"""
        desc_lines = []
        for name, tool in self.tools.items():
            doc = tool.description if hasattr(tool, 'description') else str(tool)
            desc_lines.append(f"- {name}: {doc[:100]}...")
        return "\n".join(desc_lines)
