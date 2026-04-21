# src/baselines/react_agent.py
"""
ReAct: Synergizing Reasoning and Acting in Language Models
"""

from typing import Dict, Any, List, Optional
import time
import json

from .base_agent import BaseAgent, AgentResult, StepRecord


class ReActAgent(BaseAgent):
    """
    ReAct Agent
    """
    
    def __init__(
        self,
        llm,
        tools: Dict[str, Any],
        max_steps: int = 15,
        verbose: bool = True
    ):
        super().__init__(llm, tools, max_steps, verbose)
        self.agent_name = "ReAct"
    
    def _build_prompt(
        self, 
        query: str, 
        history: List[str],
        mem_work: Dict[str, str]
    ) -> str:
        """build ReAct prompt"""
        
        history_text = "\n".join(history[-10:]) if history else "None"
        data_text = "\n".join([f"${k}: available" for k in mem_work.keys()]) if mem_work else "None"
        tools_desc = self._build_tools_description()

        prompt = f"""{query}
## Available Tools
{tools_desc}

## Available Data (use $key to reference)
{data_text}

## Previous Steps
{history_text}

## ReAct Instructions
Think step by step. At each step:
1. Thought: Analyze the current situation and what needs to be done.
2. Action: Choose a tool and provide arguments.

Format:
Thought: [reasoning, considering past lessons]
Action: tool_name[{{"arg1": "value1", "arg2": "value2"}}]

If complete:
Final Answer: [response]

## Your Response:
"""
        return prompt
    
    def run(
        self, 
        query: str,
        initial_data: Dict = None,
        session_id: str = "react_default",
        **kwargs
    ) -> AgentResult:
        """Run ReAct"""
        
        start_time = time.time()
        self.trajectory = []
        self.total_tokens = 0
        history = []
        mem_work = initial_data or {}
        
        self._log(f"\n{'='*50}")
        self._log(f"🔄 ReAct Agent Starting")
        self._log(f"{'='*50}")
        
        final_output = None
        success = False
        error_message = None
        loop_detected = False
        constraint_violations = []
        
        # Flags
        draft_generated = False
        code_generated = False
        validation_passed = False
        
        if "$draft" in mem_work.keys():
             history.append(f"System: Draft loaded into memory.")
        
        for step_i in range(self.max_steps):
            self._log(f"\n--- Step {step_i + 1} ---")
            
            # 1. build prompt
            prompt = self._build_prompt(query, history, mem_work)
            
            # 2. call LLM
            response, tokens = self._call_llm(prompt)
            self._log(f"LLM: {response[:300]}...")
            
            # 3. check for completion
            is_terminal, answer = self._is_terminal(response)
            if is_terminal:
                self._log(f"✅ Task completed")
                final_output = answer
                success = True
                break
            
            # 4. parse action
            tool_name, tool_args = self._parse_action(response)
            
            if not tool_name:
                history.append(f"Invalid action format. Please use: Action: tool_name[{{\"arg\": \"value\"}}]")
                self.trajectory.append(StepRecord(
                    step_id=step_i,
                    action_type="invalid",
                    content=response,
                    tokens_used=tokens
                ))
                continue
            
            self._log(f"Action: {tool_name}")
            
            # 5. resolve references
            if tool_args:
                for key, value in list(tool_args.items()):
                    if isinstance(value, str) and value.startswith("$"):
                        ref_key = value[1:]
                        if ref_key in mem_work:
                            tool_args[key] = mem_work[ref_key]
            
            # 6. execute tools
            tool_output = self._call_tool(tool_name, tool_args or {}, session_id)
            
            # 7. update data
            output_key = self.tool_output_keys.get(tool_name, tool_name)
            limit = 15000 if output_key in ["machine_code", "draft", "generate_machine_code"] else 2000 
            if isinstance(tool_output, str):
                mem_work[output_key] = tool_output[:limit]
            elif isinstance(tool_output, dict):
                mem_work[output_key] = json.dumps(tool_output, ensure_ascii=False)[:limit]
            
            # 8. update flags
            if tool_name == "generate_scientific_draft":
                draft_generated = True
            elif tool_name == "generate_machine_code":
                code_generated = True
            elif tool_name in ["validate_machine_code", "reflect_on_protocol"]:
                passed, msg = self._check_validation_status(tool_output)
                if passed:
                    validation_passed = True
                else:
                    constraint_violations.append(f"{tool_name}: {msg}")
            
            # 9. observation
            if isinstance(tool_output, dict):
                obs_str = json.dumps(tool_output, ensure_ascii=False)[:500]
            else:
                obs_str = str(tool_output)[:500]
            
            history.append(f"Action: {tool_name}\nObservation: {obs_str}")
            
            self.trajectory.append(StepRecord(
                step_id=step_i,
                action_type="act",
                content=response,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_output=tool_output,
                tokens_used=tokens
            ))
            
            # 10. detect loop
            if self._detect_loop():
                self._log("⚠️ Loop detected!")
                loop_detected = True
                error_message = "Agent stuck in loop"
                break
        
        # build final output
        if not success and not final_output:
            if "machine_code" in mem_work and "error" not in str(mem_work["machine_code"]):
                final_output = mem_work["machine_code"]
                success = code_generated and validation_passed
                error_message = "Max steps reached but find answer"
            elif "draft" in mem_work and "error" not in str(mem_work["draft"]):
                final_output = mem_work["draft"]
                success = draft_generated
                error_message = "Max steps reached but find answer"
            else:
                prompt = f"""
        ## User Query
        {query}

        ## Available Data
        {mem_work}

        ## Current Attempt History
        {history}

        ## Instructions
        Please use the available data and history to fulfill the user's query. 
        Only output the content that meets the user's query, and do not include your reasoning or other explanations.

        ## Your Response:
            """
                error_message = "Final Attempt"
                response, tokens = self._call_llm(prompt)
                final_output = response
                success = True

        
        if not final_output:
            error_message = error_message or "Max steps reached"
        
        total_time = time.time() - start_time
        
        return AgentResult(
            success=success,
            final_output=final_output,
            trajectory=[
                {"step": r.step_id, "type": r.action_type, "tool": r.tool_name}
                for r in self.trajectory
            ],
            total_steps=len(self.trajectory),
            total_tokens=self.total_tokens,
            total_time=total_time,
            error_message=error_message,
            loop_detected=loop_detected,
            constraint_violations=constraint_violations,
            draft_generated=draft_generated,
            code_generated=code_generated,
            validation_passed=validation_passed,
            total_trajectory=self.trajectory
        )