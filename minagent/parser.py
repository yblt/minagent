# -*- coding: utf-8 -*-
"""LLM 输出解析：提取思考过程 / 工具调用 / 最终答案。

协议（写入 system prompt）：模型每轮只输出一个 JSON 对象——
  {"thought": "...", "action": {"tool": "名字", "args": {...}}}   # 要调工具
  {"thought": "...", "answer": "..."}                             # 直接回答
解析容错：允许 ```json 代码围栏与前后杂文本；解析失败返回 error，由主循环回喂纠错提示。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Parsed:
    thought: str = ""
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    error: Optional[str] = None  # 非 None 表示本轮输出不可用


def _extract_json_block(text: str) -> Optional[str]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # 出现了 '{' 但没有配平的 '}'：交给 json.loads 报错而不是当纯文本
    return text[start:]


def parse_llm_output(raw: str) -> Parsed:
    if not raw or not raw.strip():
        return Parsed(error="empty LLM output")
    block = _extract_json_block(raw)
    if block is None:  # 没有 JSON：视为模型直接给的自然语言答案（宽松兜底）
        return Parsed(answer=raw.strip())
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as e:
        return Parsed(error=f"invalid JSON from model: {e}")
    if not isinstance(obj, dict):
        return Parsed(error="model output is not a JSON object")

    thought = str(obj.get("thought", ""))
    if "answer" in obj and obj["answer"] is not None:
        return Parsed(thought=thought, answer=str(obj["answer"]))
    action = obj.get("action")
    if isinstance(action, dict):
        tool, args = action.get("tool"), action.get("args", {})
        if not isinstance(tool, str) or not tool:
            return Parsed(thought=thought, error="action.tool missing or not a string")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return Parsed(thought=thought, error="action.args must be an object")
        return Parsed(thought=thought, tool=tool, args=args)
    return Parsed(
        thought=thought,
        error='output must contain either "answer" or "action"',
    )
