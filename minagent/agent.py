# -*- coding: utf-8 -*-
"""Agent 核心循环（Runtime 全部手写，不依赖任何 agent 框架）。

Loop：
  接收用户输入 → 组 context → LLM → 解析
    ├─ answer：结束，返回给用户
    ├─ action：执行工具 → 结果作为 observation 回喂 → 继续 loop
    ├─ 解析失败：回喂纠错提示 → 继续 loop（计入轮次）
    └─ LLM/工具异常：LLM 重试一次；工具异常转 observation
  超过 max_iterations 强制收尾。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import ContextManager
from .llm import LLMError
from .parser import parse_llm_output
from .session import Session, SessionStore
from .tools import ToolError, ToolRegistry, build_default_registry
from .trace import Tracer

SYSTEM_PROMPT_TMPL = """你是一个会自主决定使用工具的 Agent。只输出一个 JSON 对象，两种格式二选一：
{{"thought": "你的推理", "action": {{"tool": "工具名", "args": {{...}}}}}}  # 需要工具时
{{"thought": "你的推理", "answer": "给用户的最终回答"}}                  # 能回答时

可用工具（name/description/参数 Schema）：
{tools}

规则：
1. 一次只调用一个工具；工具结果会以 role=tool 的消息返回给你，你再决定继续调用还是给出答案。
2. 工具返回 error 时，根据错误信息修正参数重试，或改用其他方式回答。
3. 回答用户追问前，先查看历史中已有的工具结果，能复用就不要重复调用。
4. 当前会话待办快照：{todos}
"""


@dataclass
class AgentReply:
    text: str
    steps: int
    thought: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)
    finished: bool = True  # False = 触顶 max_iterations 被强制收尾


class Agent:
    def __init__(
        self,
        llm,
        registry: Optional[ToolRegistry] = None,
        store: Optional[SessionStore] = None,
        ctx: Optional[ContextManager] = None,
        max_iterations: int = 8,
        llm_retries: int = 1,
        trace_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry or build_default_registry()
        self.store = store or SessionStore()
        self.ctx = ctx or ContextManager()
        self.max_iterations = max_iterations
        self.llm_retries = llm_retries
        self.trace_dir = Path(trace_dir) if trace_dir else None

    # -- helpers ---------------------------------------------------------
    def _system_prompt(self, session: Session) -> str:
        todos = session.state.get("todos", [])
        snap = json_dumps_compact(todos) if todos else "（空）"
        return SYSTEM_PROMPT_TMPL.format(tools=self.registry.prompt_schemas(), todos=snap)

    def _call_llm(self, messages, tracer: Tracer) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.llm_retries + 1):
            try:
                tracer.log("llm_request", attempt=attempt, n_messages=len(messages))
                return self.llm.chat(messages)
            except LLMError as e:
                last_err = e
                tracer.log("llm_error", attempt=attempt, error=str(e))
        raise last_err  # type: ignore[misc]

    # -- main loop -------------------------------------------------------
    def chat(self, user_id: str, window_id: str, user_input: str) -> AgentReply:
        session = self.store.get_or_create(user_id, window_id)
        tracer = Tracer(session.session_id, self.trace_dir / f"{session.session_id.replace(':', '_')}.jsonl" if self.trace_dir else None)
        session.append("user", user_input)
        steps = 0
        try:
            for steps in range(1, self.max_iterations + 1):
                messages = self.ctx.build(self._system_prompt(session), session.messages)
                try:
                    raw = self._call_llm(messages, tracer)
                except LLMError as e:
                    text = f"抱歉，模型服务暂时不可用：{e}"
                    tracer.log("final", text=text, reason="llm_unavailable")
                    return AgentReply(text=text, steps=steps, finished=False)

                parsed = parse_llm_output(raw)
                tracer.log("llm_raw", step=steps, raw=raw[:2000])

                if parsed.error:  # 输出不合法 → 回喂纠错，继续 loop
                    session.append("assistant", raw)
                    session.append("user", f"[系统提示] 你的输出不符合协议（{parsed.error}），请只输出一个合法 JSON 对象。")
                    tracer.log("parse_error", step=steps, error=parsed.error)
                    continue

                if parsed.answer is not None:
                    session.append("assistant", raw)
                    tracer.log("final", step=steps, answer=parsed.answer)
                    return AgentReply(text=parsed.answer, steps=steps, thought=parsed.thought, trace=tracer.events)

                # 工具调用
                session.append("assistant", raw)  # 保留 thought+action，供追问回溯
                try:
                    observation = self.registry.call(parsed.tool, parsed.args, session=session)  # type: ignore[arg-type]
                    role_content = observation
                    tracer.log("tool_call", step=steps, tool=parsed.tool, args=parsed.args, ok=True)
                except ToolError as e:
                    role_content = f"error: {e}"
                    tracer.log("tool_call", step=steps, tool=parsed.tool, args=parsed.args, ok=False, error=str(e))
                session.append("tool", self.ctx.trim_tool_result(role_content))
        finally:
            self.store.save(session)
        text = "已达到最大循环轮次，仍未得到最终答案。请把问题拆小一点再试。"
        tracer.log("final", text=text, reason="max_iterations")
        return AgentReply(text=text, steps=self.max_iterations, trace=tracer.events, finished=False)


def json_dumps_compact(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
