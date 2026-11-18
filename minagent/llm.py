# -*- coding: utf-8 -*-
"""LLM 客户端：可换后端。

- MockLLM：脚本化回复，测试与离线演示用。
- OpenAICompatClient：任意 OpenAI 兼容 API（DeepSeek/通义/OpenAI），仅用标准库 urllib，
  通过环境变量配置：MINAGENT_BASE_URL / MINAGENT_API_KEY / MINAGENT_MODEL。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional


class LLMError(Exception):
    """LLM 调用层错误（网络/鉴权/限流），Agent 主循环捕获后重试或优雅退出。"""


Message = Dict[str, str]


def to_wire_messages(messages: List[Message]) -> List[Message]:
    """出站映射：内部 role=tool 的观察转为 role=user。

    OpenAI 兼容协议要求 role=tool 必须带 tool_call_id 且前面有 assistant.tool_calls；
    本项目走 JSON-in-text 自定义协议，没有该结构，原样发送会被真实 API 拒绝（400）。
    存储层保留 role=tool（trace 与追问回溯用），只在发送边界转换。
    """
    wire: List[Message] = []
    for m in messages:
        if m["role"] == "tool":
            wire.append({"role": "user", "content": f"[工具结果] {m['content']}"})
        else:
            wire.append(dict(m))
    return wire


class MockLLM:
    """按脚本顺序吐出回复；脚本耗尽后返回兜底 answer。记录每次收到的 messages 供断言。"""

    def __init__(self, script: Optional[List[str]] = None) -> None:
        self.script = list(script or [])
        self.calls: List[List[Message]] = []

    def chat(self, messages: List[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if self.script:
            return self.script.pop(0)
        return json.dumps({"thought": "fallback", "answer": "（mock 脚本耗尽的兜底回复）"}, ensure_ascii=False)


class OpenAICompatClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MINAGENT_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("MINAGENT_API_KEY", "")
        self.model = model or os.environ.get("MINAGENT_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        if not self.api_key:
            raise LLMError("缺少 API Key：设置环境变量 MINAGENT_API_KEY，或改用 MockLLM")

    def chat(self, messages: List[Message]) -> str:
        body = json.dumps(
            {"model": self.model, "messages": to_wire_messages(messages), "temperature": 0}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"LLM network error: {e}") from e
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {str(payload)[:200]}") from e
