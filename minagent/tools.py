# -*- coding: utf-8 -*-
"""工具注册与实现。

每个工具 = 名称 + 描述 + 参数 JSON Schema + 执行函数。
LLM 只看得到 schema（注入 system prompt），执行时由注册表统一校验参数并分发。
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolError(Exception):
    """工具可预期错误：错误信息会作为 observation 回喂给 LLM 重试/绕行。"""


# 允许出现在 session.state 里的工具私有状态，见 todo 工具
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema (object)
    func: Callable[..., str] = field(repr=False, default=lambda **_: "")

    def schema_line(self) -> str:
        return json.dumps(
            {"name": self.name, "description": self.description, "parameters": self.parameters},
            ensure_ascii=False,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, parameters: Dict[str, Any]):
        """装饰器风格注册。"""

        def deco(fn: Callable[..., str]) -> Callable[..., str]:
            self.register(Tool(name, description, parameters, fn))
            return fn

        return deco

    def names(self) -> List[str]:
        return list(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def prompt_schemas(self) -> str:
        return "\n".join(t.schema_line() for t in self._tools.values())

    def call(self, name: str, args: Dict[str, Any], session=None) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool '{name}', available: {self.names()}")
        if not isinstance(args, dict):
            raise ToolError(f"args of '{name}' must be an object, got {type(args).__name__}")
        # 必填参数校验（schema.required）
        for key in tool.parameters.get("required", []):
            if key not in args:
                raise ToolError(f"missing required arg '{key}' for tool '{name}'")
        try:
            return tool.func(session=session, **args)
        except ToolError:
            raise
        except TypeError as e:  # 参数名不对等
            raise ToolError(f"bad args for '{name}': {e}")
        except Exception as e:  # 兜底，绝不让工具异常炸掉主循环
            raise ToolError(f"tool '{name}' failed: {e.__class__.__name__}: {e}")


# --------------------------------------------------------------------------
# 具体工具实现
# --------------------------------------------------------------------------

_CALC_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)


def _safe_eval(node: ast.AST) -> float:
    """AST 白名单求值：只允许数字、四则运算、括号、一元正负。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _CALC_BINOPS):
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ToolError("division by zero")
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 64:
                raise ToolError("exponent too large")
            return left**right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    raise ToolError(f"unsupported expression element: {node.__class__.__name__}")


_CORPUS = {
    "kv cache": "KV Cache 是大模型推理中缓存注意力层 Key/Value 张量的技术，可避免重复计算，显存占用随序列长度线性增长。",
    "react": "ReAct 范式让模型交替产生推理轨迹(Reasoning)与行动(Acting)，行动结果再作为观察进入下一轮推理。",
    "mcp": "MCP(Model Context Protocol) 是智能体与外部工具/数据源之间的标准化调用协议，定义了 tools/resources/prompts 三类能力。",
    "限流": "滑动窗口限流记录每个请求的时间戳，判断任意连续窗口内请求数是否超限，比固定窗口更精确但内存开销更大。",
}


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        "calculator",
        "计算一个算术表达式，支持 + - * / % ** 与括号",
        {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "如 '(3+4)*2'"}},
            "required": ["expression"],
        },
    )
    def calculator(expression: str, session=None) -> str:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError:
            raise ToolError(f"invalid expression: {expression!r}")
        return str(_safe_eval(tree))

    @reg.tool(
        "search",
        "检索内部知识库（mock），返回最相关的若干条资料",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词"}},
            "required": ["query"],
        },
    )
    def search(query: str, session=None) -> str:
        hits = [v for k, v in _CORPUS.items() if k in query.lower() or query.lower() in k]
        if not hits:  # 退化为子串匹配
            hits = [v for k, v in _CORPUS.items() if any(w in k for w in query.split())]
        return json.dumps(hits[:3] or ["no result"], ensure_ascii=False)

    @reg.tool(
        "weather",
        "查询城市天气（mock 数据）",
        {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
            "required": ["city"],
        },
    )
    def weather(city: str, session=None) -> str:
        data = {"北京": "晴 26℃ 东南风2级", "上海": "多云 24℃", "广州": "雷阵雨 29℃"}
        if city not in data:
            raise ToolError(f"no weather data for city '{city}'")
        return f"{city}: {data[city]}"

    @reg.tool(
        "todo",
        "管理当前会话的待办事项。action=add 新增(content)；list 查看全部；done 完成(index,从1开始)",
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "done"]},
                "content": {"type": "string", "description": "action=add 时的待办内容"},
                "index": {"type": "integer", "description": "action=done 时的序号，从1开始"},
            },
            "required": ["action"],
        },
    )
    def todo(action: str, content: Optional[str] = None, index: Optional[int] = None, session=None) -> str:
        if session is None:
            raise ToolError("todo requires a session context")
        items: List[Dict[str, Any]] = session.state.setdefault("todos", [])
        if action == "add":
            if not content:
                raise ToolError("action=add requires 'content'")
            items.append({"content": content, "done": False})
            return f"added, total {len(items)} todos"
        if action == "list":
            if not items:
                return "todos: (empty)"
            return "\n".join(f"{i+1}. [{'x' if t['done'] else ' '}] {t['content']}" for i, t in enumerate(items))
        if action == "done":
            if not isinstance(index, int) or not 1 <= index <= len(items):
                raise ToolError(f"invalid index {index}, valid range 1..{len(items)}")
            items[index - 1]["done"] = True
            return f"marked done: {items[index-1]['content']}"
        raise ToolError(f"unknown action '{action}'")

    return reg
