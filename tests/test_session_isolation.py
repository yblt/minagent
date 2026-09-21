# -*- coding: utf-8 -*-
"""session 隔离测试：复现考题场景（用户A 两个窗口互不影响、可分别续聊）。"""
import json

from minagent import Agent, MockLLM, SessionStore


def act(tool, **args):
    return json.dumps({"thought": "t", "action": {"tool": tool, "args": args}}, ensure_ascii=False)


def ans(text):
    return json.dumps({"thought": "t", "answer": text}, ensure_ascii=False)


def test_two_windows_isolated():
    store = SessionStore()
    a1 = Agent(MockLLM([act("todo", action="add", content="窗口1待办"), ans("窗口1已记录")]), store=store)
    a2 = Agent(MockLLM([act("todo", action="add", content="窗口2待办"), ans("窗口2已记录")]), store=store)

    r1 = a1.chat("userA", "win1", "记待办：窗口1待办")
    r2 = a2.chat("userA", "win2", "记待办：窗口2待办")
    assert r1.text and r2.text

    w1 = store.get("userA::win1")
    w2 = store.get("userA::win2")
    assert [t["content"] for t in w1.state["todos"]] == ["窗口1待办"]
    assert [t["content"] for t in w2.state["todos"]] == ["窗口2待办"]
    # 历史也完全独立
    assert all("窗口2" not in m["content"] for m in w1.messages)


def test_resume_window_later():
    store = SessionStore()
    agent = Agent(MockLLM([ans("你之前说：项目A延期")]), store=store)
    agent.chat("u", "w", "我上周说项目A延期了，先这样")
    # 隔了很久回来接着聊：历史仍在同一 session
    agent.llm = MockLLM([ans("是的，你说项目A延期")])
    r = agent.chat("u", "w", "上次我说的是什么来着")
    assert "延期" in r.text
    hist = [m["content"] for m in agent.llm.calls[0] if m["role"] == "user"]
    assert any("项目A延期" in h for h in hist)


def test_persistence_roundtrip(tmp_path):
    store = SessionStore(persist_dir=tmp_path)
    agent = Agent(MockLLM([ans("ok")]), store=store)
    agent.chat("u", "w", "hello")
    # 重新加载 = 模拟进程重启
    store2 = SessionStore(persist_dir=tmp_path)
    s = store2.get("u::w")
    assert s is not None and s.messages[0]["content"] == "hello"


def test_system_prompt_shows_current_todos():
    store = SessionStore()
    agent = Agent(MockLLM([act("todo", action="add", content="买咖啡"), ans("好")]), store=store)
    agent.chat("u", "w", "记待办买咖啡")
    agent.llm = MockLLM([ans("你已经有待办了")])
    agent.chat("u", "w", "继续")
    sysmsg = agent.llm.calls[0][0]["content"]
    assert "买咖啡" in sysmsg  # 状态快照注入 system，而不是塞满对话历史
