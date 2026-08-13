"""
LLM co-pilot — explains EDA findings and recommends preprocessing steps,
grounded in the RAG corpus (agent/rag.py) over sklearn/scipy docstrings.

Citations are inline [S1]-style markers the model is required to use, then
deterministically verified: each cited quote must actually substring-match
its source chunk's text. This is a real hallucination guard, not just a
prompt instruction the model may or may not follow — same pattern as the
sibling Financial-Anomaly-Detection-Using-RAG project's grounded_explainer.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from agent.groq_client import get_client
from agent.rag import RetrievedChunk, get_retriever

MODEL = "llama-3.3-70b-versatile"


class Citation(BaseModel):
    marker: str = Field(description="Inline marker used in the answer text, e.g. 'S1'")
    source_text: str = Field(description="The exact quoted substring from the cited source chunk")


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    unverified_citation_markers: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="source identifiers of every chunk retrieved, cited or not")

    @property
    def is_fully_grounded(self) -> bool:
        return len(self.unverified_citation_markers) == 0


def _normalize_ws(text: str) -> str:
    """Docstrings wrap mid-sentence with embedded newlines; a model quoting
    one naturally (correctly) flattens it to a single line. Collapsing
    whitespace before comparing avoids flagging accurate quotes as
    unverified just because the source happened to wrap differently."""
    return re.sub(r"\s+", " ", text).strip()


def _verify(answer: GroundedAnswer, source_chunks: list[RetrievedChunk]) -> GroundedAnswer:
    source_texts = [_normalize_ws(c.text) for c in source_chunks]
    unverified = [
        c.marker
        for c in answer.citations
        if not c.source_text.strip() or not any(_normalize_ws(c.source_text) in s for s in source_texts)
    ]
    answer.unverified_citation_markers = unverified
    return answer


def _source_block(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[S{i + 1}] ({c.source} — {c.topic})\n{c.text}" for i, c in enumerate(chunks))


def _parse(raw: str) -> tuple[str, list[Citation]]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        data = json.loads(match.group(0)) if match else {"answer": raw or "", "citations": []}
    return data.get("answer", ""), [Citation(**c) for c in data.get("citations", [])]


def _ask(prompt: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
    client = get_client()
    response = client.create_chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=1024,
        response_format={"type": "json_object"},
    )
    answer_text, citations = _parse(response.choices[0].message.content)
    answer = GroundedAnswer(answer=answer_text, citations=citations, sources=[c.source for c in chunks])
    return _verify(answer, chunks)


def explain_eda(question: str, findings: dict, top_k: int = 5) -> GroundedAnswer:
    """Explain an EDA finding (e.g. "why is x1 skewed?") grounded in the
    retrieved statistics documentation. `findings` is whatever summary stats
    the caller already has (e.g. from GET /eda/{name}/summary) — passed
    through as context, not re-fetched here."""
    chunks = get_retriever().retrieve(question, top_k=top_k)
    prompt = f"""You are a data-analysis co-pilot explaining exploratory data analysis findings to a
user. Answer the question below using ONLY the numbered sources provided plus the findings data.
Every factual claim about what a statistic/method means must cite a source marker like [S1], [S2]
inline in the answer text. Do not cite a source number that isn't listed below. If the sources
don't cover something, say so rather than making a claim you can't cite.

Sources:
{_source_block(chunks)}

EDA findings (JSON):
{json.dumps(findings, indent=2, default=str)}

Question: {question}

Respond with a single JSON object matching this schema exactly:
{{
  "answer": "<answer text with inline [S1]-style citation markers>",
  "citations": [
    {{"marker": "S1", "source_text": "<exact quoted substring from source S1>"}}
  ]
}}
Only include a citation entry for a marker if you actually used it in the answer text.
The "source_text" for each citation must be an exact, verbatim substring of that source's text above."""
    return _ask(prompt, chunks)


def recommend_preprocessing(eda_summary: dict, top_k: int = 6) -> GroundedAnswer:
    """Recommend cleaning/preprocessing steps grounded in the retrieved
    documentation, given an EDA summary (missingness, skew, outlier stats —
    whatever shape the caller has, e.g. GET /eda/{name}/summary +
    /missing + /boxplot combined)."""
    query = "missing value imputation, outlier removal, and feature scaling method selection"
    chunks = get_retriever().retrieve(query, top_k=top_k)
    prompt = f"""You are a data-analysis co-pilot recommending preprocessing steps for a dataset.
Using ONLY the numbered sources below plus the EDA summary data, recommend a missing-value
strategy, an outlier-handling method, and a scaler, with brief reasoning. Every factual claim
about what a method does must cite a source marker like [S1], [S2] inline. Do not cite a source
number that isn't listed below.

Sources:
{_source_block(chunks)}

EDA summary (JSON):
{json.dumps(eda_summary, indent=2, default=str)}

Respond with a single JSON object matching this schema exactly:
{{
  "answer": "<recommendation text with inline [S1]-style citation markers>",
  "citations": [
    {{"marker": "S1", "source_text": "<exact quoted substring from source S1>"}}
  ]
}}
Only include a citation entry for a marker if you actually used it in the answer text.
The "source_text" for each citation must be an exact, verbatim substring of that source's text above."""
    return _ask(prompt, chunks)
