# -*- coding: utf-8 -*-
"""context 管理测试：轮次上限、压缩、工具结果截断。"""
from minagent import ContextManager
from minagent.context import SUMMARY_PREFIX


def msgs(n):
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"m{i}"})
    return out


def test_no_compression_under_limit():
    ctx = ContextManager(max_messages=10, keep_recent=4)
    built = ctx.build("SYS", msgs(6))
    assert built[0]["content"] == "SYS"
    assert not any(SUMMARY_PREFIX in m["content"] for m in built)
    assert len(built) == 7


def test_compression_keeps_system_and_recent():
    ctx = ContextManager(max_messages=8, keep_recent=4)
    built = ctx.build("SYS", msgs(20))
    assert built[0]["content"] == "SYS"
    summary = built[1]
    assert summary["role"] == "system" and summary["content"].startswith(SUMMARY_PREFIX)
    assert [m["content"] for m in built[-4:]] == [f"m{i}" for i in range(16, 20)]
    # 被压缩的旧消息进入摘要行
    assert "m0" in summary["content"] and "m7" not in [m["content"] for m in built if not m["content"].startswith(SUMMARY_PREFIX)]


def test_custom_summarizer():
    ctx = ContextManager(max_messages=4, keep_recent=2, summarizer=lambda old: f"{len(old)}条被压缩")
    built = ctx.build("S", msgs(10))
    assert "8条被压缩" in built[1]["content"]  # 10 - keep_recent(2) = 8 条进摘要


def test_tool_result_truncation():
    ctx = ContextManager(tool_result_limit=10)
    t = ctx.trim_tool_result("x" * 100)
    assert len(t) < 30 and t.endswith("…(truncated)")


def test_needs_compression_flag():
    ctx = ContextManager(max_messages=5)
    assert ctx.needs_compression(msgs(6))
    assert not ctx.needs_compression(msgs(5))
