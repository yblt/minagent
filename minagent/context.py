# -*- coding: utf-8 -*-
"""Context 管理：决定每轮发给 LLM 的消息集合，控制长度与成本。

策略（与 README 中 memory 说明一致）：
1. system prompt（工具协议 + 会话内待办摘要）永远保留在首位。
2. 历史按消息数控制：超过 max_messages 时，保留最近 keep_recent 条，
   更早的消息压缩成一条 "[历史摘要]" system 消息——基础压缩，不调用 LLM。
3. 工具结果超长单独截断（结果里最有信息量的是头部）。
4. 保留什么：用户输入、工具调用（含参数）、工具结果、最终答案都入历史；
   thought 也保留但会被摘要压缩——它是"追问为什么这么做"的依据。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

Message = Dict[str, str]

SUMMARY_PREFIX = "[历史摘要]"


def default_summarizer(messages: List[Message]) -> str:
    """朴素摘要：每条消息压成 role:前80字，一行一条。"""
    lines = []
    for m in messages:
        content = m["content"].replace("\n", " ")
        if len(content) > 80:
            content = content[:80] + "…"
        lines.append(f"{m['role']}: {content}")
    return "\n".join(lines)


class ContextManager:
    def __init__(
        self,
        max_messages: int = 24,
        keep_recent: int = 10,
        tool_result_limit: int = 500,
        summarizer: Optional[Callable[[List[Message]], str]] = None,
    ) -> None:
        assert keep_recent >= 2, "keep_recent 至少要保留一问一答"
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.tool_result_limit = tool_result_limit
        self.summarizer = summarizer or default_summarizer

    def trim_tool_result(self, content: str) -> str:
        if len(content) > self.tool_result_limit:
            return content[: self.tool_result_limit] + "…(truncated)"
        return content

    def build(self, system_prompt: str, history: List[Message]) -> List[Message]:
        """history 不含 system；返回发送给 LLM 的完整 messages。"""
        msgs = history
        if len(msgs) > self.max_messages:
            split = len(msgs) - self.keep_recent
            # 切点回退到最近的用户消息边界：一个 turn = user + assistant/tool 若干，
            # 从中间切会把 action 和它的工具结果拆进摘要两侧，追问时对不上号
            while split > 0 and msgs[split]["role"] != "user":
                split -= 1
            if split > 0:
                old, recent = msgs[:split], msgs[split:]
                summary = self.summarizer(old)
                msgs = [{"role": "system", "content": f"{SUMMARY_PREFIX}\n{summary}"}] + recent
        return [{"role": "system", "content": system_prompt}] + msgs

    def needs_compression(self, history: List[Message]) -> bool:
        return len(history) > self.max_messages
