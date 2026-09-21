# -*- coding: utf-8 -*-
"""核心循环测试：工具链路、异常处理、轮次上限、trace、追问复用。"""
import json

from minagent import Agent, MockLLM, SessionStore
from minagent.parser import parse_llm_output


def act(tool, **args):
    return json.dumps({"thought": f"call {tool}", "action": {"tool": tool, "args": args}}, ensure_ascii=False)


def ans(text):
    return json.dumps({"thought": "done", "answer": text}, ensure_ascii=False)


def test_tool_then_answer():
    llm = MockLLM([act("weather", city="北京"), ans("北京晴 26℃")])
    agent = Agent(llm)
    r = agent.chat("u", "w", "北京天气怎么样")
    assert r.text == "北京晴 26℃" and r.steps == 2 and r.finished
    # 历史中：user, assistant(action), tool(结果), assistant(answer)
    roles = [m["role"] for m in agent.store.get("u::w").messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = agent.store.get("u::w").messages[2]
    assert "26" in tool_msg["content"]


def test_multi_tool_chain():
    llm = MockLLM(
        [
            act("calculator", expression="(3+4)*2"),
            act("todo", action="add", content="记住结果14"),
            ans("算完是 14，已记待办"),
        ]
    )
    agent = Agent(llm)
    r = agent.chat("u", "w", "算 (3+4)*2 并记个待办")
    assert r.steps == 3 and r.text == "算完是 14，已记待办"


def test_tool_error_becomes_observation_and_loop_continues():
    llm = MockLLM(
        [
            act("weather", city="火星"),          # 工具报错
            ans("火星没有天气数据，无法查询"),       # 修正后回答
        ]
    )
    agent = Agent(llm)
    r = agent.chat("u", "w", "火星天气")
    assert r.finished
    tool_msg = agent.store.get("u::w").messages[2]
    assert tool_msg["content"].startswith("error:")
    # 第二次 LLM 调用能看到错误 observation
    assert any(m["role"] == "tool" and "error" in m["content"] for m in llm.calls[1])


def test_bad_llm_output_gets_corrected():
    # 输出了 '{' 但不是合法 JSON → 协议纠错回喂；完全不含 JSON 的纯文本走宽松 answer 兜底
    llm = MockLLM(['{"thought": 坏了, 没有引号', ans("我重新回答了")])
    agent = Agent(llm)
    r = agent.chat("u", "w", "你好")
    assert r.text == "我重新回答了"
    # 纠错提示以 user 消息回喂
    assert any("不符合协议" in m["content"] for m in llm.calls[1] if m["role"] == "user")


def test_max_iterations_guard():
    # 永远只调工具不出答案
    llm = MockLLM(script=[act("todo", action="list")] * 10)
    agent = Agent(llm, max_iterations=3)
    r = agent.chat("u", "w", "无限循环吧")
    assert not r.finished and "最大循环轮次" in r.text
    assert len(llm.calls) == 3


def test_trace_events_recorded():
    llm = MockLLM([act("weather", city="上海"), ans("上海多云")])
    agent = Agent(llm)
    r = agent.chat("u", "w", "上海天气")
    kinds = [e["kind"] for e in r.trace]
    assert "llm_request" in kinds and "tool_call" in kinds and "final" in kinds
    tc = next(e for e in r.trace if e["kind"] == "tool_call")
    assert tc["tool"] == "weather" and tc["ok"] is True


def test_followup_reuses_history_tool_result():
    agent_llm1 = MockLLM([act("weather", city="北京"), ans("北京 26℃")])
    store = SessionStore()
    agent = Agent(agent_llm1, store=store)
    agent.chat("u", "w", "北京天气")
    # 追问：mock 直接依据历史回答；断言第二次会话请求里带着工具结果
    agent.llm = MockLLM([ans("刚才查过：26℃")])
    r = agent.chat("u", "w", "多少度来着")
    assert "26" in r.text
    last_msgs = agent.llm.calls[0]
    assert any(m["role"] == "tool" and "26" in m["content"] for m in last_msgs)
    # system prompt 中包含待办快照占位
    assert "可用工具" in last_msgs[0]["content"]


def test_llm_unavailable_degrades_gracefully():
    class BrokenLLM:
        def __init__(self):
            self.n = 0
        def chat(self, messages):
            from minagent.llm import LLMError
            self.n += 1
            raise LLMError("connection refused")

    agent = Agent(BrokenLLM(), llm_retries=1)
    r = agent.chat("u", "w", "hi")
    assert not r.finished and "不可用" in r.text
    assert agent.llm.n == 2  # 首次 + 重试1次
