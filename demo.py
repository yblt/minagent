# -*- coding: utf-8 -*-
"""演示脚本。

离线（默认，MockLLM 脚本化，复现考题场景）：
    python demo.py
真实 LLM（任一 OpenAI 兼容 API）：
    set MINAGENT_API_KEY=sk-xxx
    set MINAGENT_BASE_URL=https://api.deepseek.com/v1
    set MINAGENT_MODEL=deepseek-chat
    python demo.py --real

场景（题目原设定）：用户 A 开窗口1 查天气记待办；开窗口2 写周报记待办；
两个 session 互不影响，且各自可续聊追问。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minagent import Agent, MemoryStore, MockLLM, OpenAICompatClient, SessionStore
from minagent.tools import build_default_registry


def j(thought: str, tool: str, **args) -> str:
    return json.dumps({"thought": thought, "action": {"tool": tool, "args": args}}, ensure_ascii=False)


def ja(thought: str, answer: str) -> str:
    return json.dumps({"thought": thought, "answer": answer}, ensure_ascii=False)


def run_mock() -> None:
    trace_dir = Path(".minagent_traces")
    store = SessionStore(persist_dir=Path(".minagent_sessions"))
    memory = MemoryStore(persist_path=Path(".minagent_sessions") / "memory.json")

    # 窗口1：查天气 → 记待办 → 回答；再追问（复用历史里的工具结果，不再调工具）
    llm1 = MockLLM(
        [
            j("先查北京天气", "weather", city="北京"),
            j("天气拿到了，记待办", "todo", action="add", content="给妈妈打电话"),
            ja("都有结果了", "北京晴 26℃；待办已记录：给妈妈打电话"),
            ja("历史里有天气和待办，直接回答", "北京 26℃ 晴；你当前有 1 条待办：给妈妈打电话"),
            ja("已记住", "好的，已记住你的常驻城市是北京"),
        ]
    )
    agent1 = Agent(llm1, store=store, memory=memory, trace_dir=trace_dir)
    print("== 窗口1 第一轮 ==")
    r = agent1.chat("userA", "win1", "查一下北京天气，然后帮我记个待办：给妈妈打电话")
    print(f"[{r.steps} steps] {r.text}")
    print("== 窗口1 追问（应复用历史，不再调工具）==")
    r = agent1.chat("userA", "win1", "刚才天气多少度来着？我待办里有几条？")
    print(f"[{r.steps} steps] {r.text}")

    # 窗口2：同一用户另一个窗口，独立 session
    llm2 = MockLLM(
        [
            j("先记周报待办", "todo", action="add", content="提交周报"),
            ja("给出周报要点并确认待办", "周报要点：1) MCP 链路调通 2) DDL 自动化上线；待办已记录"),
            ja("长期记忆里有常驻城市", "你常驻北京——这条来自跨窗口的长期记忆，不在本窗口历史里"),
        ]
    )
    agent2 = Agent(llm2, store=store, memory=memory, trace_dir=trace_dir)
    print("== 窗口2（与窗口1 完全隔离）==")
    r = agent2.chat("userA", "win2", "帮我列一下周报要点，再记待办：提交周报")
    print(f"[{r.steps} steps] {r.text}")

    print("== 长期记忆：窗口1 写入，窗口2 召回 ==")
    r = agent1.chat("userA", "win1", "记住我的常驻城市是北京")
    print(f"窗口1> {r.text}")
    r = agent2.chat("userA", "win2", "我常驻城市哪来着？")
    print(f"窗口2> {r.text}")
    print(f"窗口2 收到的 system 含长期记忆段: {'用户长期记忆' in llm2.calls[-1][0]['content']}")

    w1 = store.get("userA::win1")
    w2 = store.get("userA::win2")
    print("== 隔离性验证 ==")
    print(f"窗口1 待办: {json.dumps(w1.state.get('todos'), ensure_ascii=False)}")
    print(f"窗口2 待办: {json.dumps(w2.state.get('todos'), ensure_ascii=False)}")
    print(f"trace 文件: {(trace_dir / 'userA__win1.jsonl').exists() and sorted(p.name for p in trace_dir.glob('*.jsonl'))}")


def run_real() -> None:
    store = SessionStore(persist_dir=Path(".minagent_sessions"))
    memory = MemoryStore(persist_path=Path(".minagent_sessions") / "memory.json")
    agent = Agent(OpenAICompatClient(), store=store, memory=memory, trace_dir=Path(".minagent_traces"))
    print("真实 LLM 交互模式，输入 exit 退出。")
    while True:
        text = input("你> ").strip()
        if text in ("exit", "quit"):
            break
        r = agent.chat("cli-user", "main", text)
        print(f"Agent[{r.steps} steps]> {r.text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="使用真实 LLM API（需环境变量，见文件头注释）")
    args = ap.parse_args()
    run_real() if args.real else run_mock()
