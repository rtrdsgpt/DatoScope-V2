"""
Generates OpenAI/Groq-format tool-calling JSON schemas from a plain Python
function's signature + docstring, so agent/tools.py's schemas can't drift
out of sync with the actual function signatures (the usual failure mode of
hand-maintaining both).
"""

from __future__ import annotations

import inspect
import types
import typing


def _json_type(annotation) -> dict:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(non_none[0]) if len(non_none) == 1 else {"type": "string"}
    if origin in (list, typing.List):
        (item_type,) = typing.get_args(annotation) or (str,)
        return {"type": "array", "items": _json_type(item_type)}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }.get(annotation, {"type": "string"})


def _is_optional(annotation) -> bool:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return type(None) in typing.get_args(annotation)
    return False


def to_tool_schema(func) -> dict:
    sig = inspect.signature(func)
    # `typing.get_type_hints` (not `param.annotation` directly) because
    # every module here uses `from __future__ import annotations`, which
    # stringifies annotations at def time ("list[str]" as text) — only
    # get_type_hints resolves them back into real type objects `_json_type`
    # can introspect; `inspect.signature` alone would silently type
    # everything as "string" (bools and lists included).
    hints = typing.get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    description = " ".join(doc.split("\n\n")[0].split())

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        properties[name] = _json_type(annotation)
        has_default = param.default is not inspect.Parameter.empty
        if not has_default and not _is_optional(annotation):
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def build_tool_schemas(functions: list) -> list[dict]:
    return [to_tool_schema(f) for f in functions]
