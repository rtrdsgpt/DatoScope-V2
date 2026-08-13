"""
Autonomous agent that runs the extract -> transform -> validate -> load ->
EDA -> model -> compare pipeline end to end from a natural-language goal
(todo.md section 7), producing a written report.

A standard tool-calling loop over agent/tools.py's functions (the same
tools the MCP server exposes) — the LLM decides which DatoScope operations
to call and in what order; this module just executes whatever it picks and
feeds the result back until it produces a final answer instead of another
tool call.
"""

from __future__ import annotations

import json

from agent.groq_client import get_client
from agent.schema_gen import build_tool_schemas
from agent.tools import TOOLS

MODEL = "llama-3.3-70b-versatile"

_TOOLS_BY_NAME = {f.__name__: f for f in TOOLS}
_SCHEMAS = build_tool_schemas(TOOLS)
_TASK_BY_TRAIN_TOOL = {"train_regression": "regression", "train_classification": "classification", "train_clustering": "clustering"}

_SYSTEM_PROMPT = """You are DatoScope's autonomous data-science agent. Given a natural-language \
goal, use the available tools to run whatever subset of the extract -> clean -> EDA -> train -> \
compare pipeline is needed to satisfy it, then produce a final written report (no more tool \
calls) summarizing what you did, the winning model and its metrics, and why it won.

Critical rules — breaking these produces a report nobody can trust:
- NEVER invent a run_id, dataset name, column name, or any other identifier. Every argument you
  pass must either be a literal value from the user's goal, or come verbatim from a previous tool
  result in this conversation (e.g. the exact run_id string a prior tool call returned).
- Most tools' run_id parameter is OPTIONAL and defaults to "the latest run" — if you don't have a
  specific run_id from a previous tool result, OMIT the parameter entirely. Do not invent a
  placeholder string like "latest" or "cleaning_run" — that is not a valid run_id and will fail.
- If a tool call returns a dict containing an "error" key, that step FAILED. Do not proceed to the
  next pipeline stage using its output, and do not report success for that step. Either fix the
  actual problem (e.g. call the tool again with corrected, non-invented arguments) or stop and
  report the failure honestly in your final report.
- Your final report's numbers (metrics, row counts, model names) must come only from tool results
  actually returned in this conversation — never state a specific number you did not receive from
  a tool. If every attempt at a step failed, say so explicitly instead of writing a success
  narrative around fabricated figures.

Other guidelines:
- A dataset must be extracted (generate_dataset or kaggle_dataset) before it can be cleaned, and \
cleaned (clean_dataset) before EDA or training can read it from the warehouse.
- Pick a task (regression/classification/clustering) matching the goal and the dataset's target \
column; for classification/regression, choose reasonable numeric feature columns and the target \
column based on eda_summary's output or the columns the extract tool returned.
- Call compare_models with the exact dict a train_* tool returned (do not edit it) once you have \
results for the task at hand, passing dataset_name so the winner gets registered in MLflow.
- Keep tool arguments minimal and sensible; don't re-run a step you already have results for.
- When you have enough information, respond with a final plain-text report and no tool calls."""


def _had_errors(tool_calls_log: list[dict]) -> bool:
    return any(isinstance(tc["result"], dict) and "error" in tc["result"] for tc in tool_calls_log)


def run(goal: str, dataset_name: str | None = None, max_turns: int = 15) -> dict:
    client = get_client()
    user_content = goal if not dataset_name else f"{goal}\n\n(Use the existing dataset '{dataset_name}' rather than generating a new one, if it fits the goal.)"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    tool_calls_log = []
    # Groq/llama-3.3-70b reliably *reads* a large tool result but sometimes
    # fails to reproduce it verbatim as a *call* argument a couple of steps
    # later (observed: passing the literal string "classification_results"
    # instead of the actual dict for compare_models' `results` param). Rather
    # than trust the model to always echo JSON correctly, the orchestration
    # layer caches each train_* call's real output and substitutes it
    # whenever the model's own `results` argument isn't a valid dict.
    last_train_results: dict[str, dict] = {}

    for turn in range(max_turns):
        response = client.create_chat_completion(
            model=MODEL,
            messages=messages,
            tools=_SCHEMAS,
            tool_choice="auto",
            max_completion_tokens=2048,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # Deterministic integrity check, not just trusting the model's own
            # narrative: a report is only as good as the tool trace behind it,
            # and an LLM asked to summarize a failed run will sometimes write a
            # confident success story around fabricated numbers instead of
            # reporting the failure — this makes that discrepancy visible to
            # the caller rather than silently trusting free-form prose.
            return {
                "report": message.content,
                "tool_calls": tool_calls_log,
                "turns": turn + 1,
                "status": "completed",
                "had_tool_errors": _had_errors(tool_calls_log),
            }

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"Could not parse arguments: {exc}"}
                args = {}
            else:
                if name == "compare_models" and not isinstance(args.get("results"), dict) and args.get("task") in last_train_results:
                    args["results"] = last_train_results[args["task"]]

                fn = _TOOLS_BY_NAME.get(name)
                if fn is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = {"error": str(exc)}

                if name in _TASK_BY_TRAIN_TOOL and isinstance(result, dict) and "error" not in result:
                    last_train_results[_TASK_BY_TRAIN_TOOL[name]] = result

            tool_calls_log.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, default=str)})

    return {
        "report": "Reached the turn limit before producing a final report.",
        "tool_calls": tool_calls_log,
        "turns": max_turns,
        "status": "max_turns_reached",
        "had_tool_errors": _had_errors(tool_calls_log),
    }
