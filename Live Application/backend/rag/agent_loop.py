"""
The Gemini tool-calling loop shared by both agents:
    rag/AgenticRAG.py      employee leave policy + live Wikipedia
    rag/BankAgenticRAG.py  bank documents, internal tools only (guard-railed)

    plan -> call tools -> read results -> call more tools ... -> final answer
The model decides which tools to call; the loop runs them and feeds the results back, for at most
AGENT_MAX_STEPS rounds. Returns the final answer text, or None when the step limit was reached.
"""

from google.genai import types

from core.config import AGENT_MAX_STEPS
from core.llm import llm


def call_tool(tools, name, args):
    if name not in tools:
        return {"error": f"Unknown tool '{name}'. Use one of: {', '.join(tools)}"}
    try:
        return tools[name](**args)
    except Exception as e:               # report tool errors back so the agent can recover
        return {"error": f"{type(e).__name__}: {e}"}


def run_tool_loop(run, system_prompt, tools, question, max_steps=AGENT_MAX_STEPS, force_first_tool=True):
    """tools = {name: python function with type hints + docstring (Gemini reads them)}.
    force_first_tool: the first round MUST call a tool, so the agent cannot answer from memory."""
    def config(mode):
        return types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=0, tools=list(tools.values()),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode)))
    history = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    for round_no in range(max_steps):
        run.llm_calls += 1
        mode = "ANY" if force_first_tool and round_no == 0 else "AUTO"
        response = llm.generate(history, config(mode), model=run.model)
        history.append(response.candidates[0].content)
        if not response.function_calls:          # no more tools wanted -> final answer
            return (response.text or "").strip() or None
        parts = [types.Part.from_function_response(
                     name=fc.name, response=call_tool(tools, fc.name, dict(fc.args or {})))
                 for fc in response.function_calls]
        history.append(types.Content(role="user", parts=parts))
    return None
