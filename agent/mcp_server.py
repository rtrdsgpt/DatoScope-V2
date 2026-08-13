#!/usr/bin/env python3
"""
MCP server exposing DatoScope's pipeline tools (agent/tools.py) plus the
grounded co-pilot and the autonomous pipeline agent, so Claude Desktop (or
any other MCP client) can drive DatoScope's analysis tools directly —
todo.md section 7. Requires the DatoScope API running (API_BASE_URL env
var, defaults to http://localhost:8000 — see utils/api_client.py).

Claude Desktop config (claude_desktop_config.json):
{
  "mcpServers": {
    "datoscope": {
      "command": "/path/to/DatoScope V2/.venv/bin/python",
      "args": ["/path/to/DatoScope V2/agent/mcp_server.py"]
    }
  }
}

Run directly: python agent/mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path when run directly

from mcp.server.fastmcp import FastMCP

from agent import tools as _tools
from agent.copilot import explain_eda as _explain_eda
from agent.copilot import recommend_preprocessing as _recommend_preprocessing
from agent.pipeline_agent import run as _run_agent

mcp = FastMCP("datoscope")

for _fn in _tools.TOOLS:
    mcp.tool()(_fn)


@mcp.tool()
def explain_eda_finding(dataset_name: str, question: str, run_id: str | None = None) -> dict:
    """Explain an EDA finding for a cleaned dataset already in the warehouse (e.g. "why is x1
    skewed?"), grounded in sklearn/scipy documentation with verified inline citations."""
    from api.routers.copilot import _build_findings

    findings = _build_findings(dataset_name, run_id)
    return _explain_eda(question, findings).model_dump()


@mcp.tool()
def recommend_preprocessing_steps(dataset_name: str, run_id: str | None = None) -> dict:
    """Recommend missing-value/outlier/scaling preprocessing steps for a raw dataset, grounded in
    sklearn/scipy documentation with verified inline citations."""
    from api.routers.copilot import _build_findings

    findings = _build_findings(dataset_name, run_id)
    return _recommend_preprocessing(findings).model_dump()


@mcp.tool()
def run_pipeline_agent(goal: str, dataset_name: str | None = None, max_turns: int = 15) -> dict:
    """Run the full extract -> clean -> EDA -> train -> compare pipeline autonomously from a
    natural-language goal (e.g. "find the best classifier for this churn dataset") and produce a
    written report. Use the single-purpose tools instead if you want to drive each step yourself."""
    return _run_agent(goal, dataset_name=dataset_name, max_turns=max_turns)


if __name__ == "__main__":
    mcp.run()
