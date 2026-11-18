# -*- coding: utf-8 -*-
"""工具注册表与各工具单测。"""
import pytest

from minagent.session import Session
from minagent.tools import ToolError, ToolRegistry, build_default_registry, Tool


@pytest.fixture()
def reg():
    return build_default_registry()


def test_registry_has_all_tools(reg):
    assert set(reg.names()) == {"calculator", "search", "weather", "todo", "read_docs"}


def test_calculator_basic(reg):
    assert reg.call("calculator", {"expression": "(3+4)*2"}) == "14"
    assert reg.call("calculator", {"expression": "10/4"}) == "2.5"


def test_calculator_rejects_code_injection(reg):
    with pytest.raises(ToolError):
        reg.call("calculator", {"expression": "__import__('os').system('dir')"})
    with pytest.raises(ToolError):
        reg.call("calculator", {"expression": "open('x')"})


def test_calculator_div_zero(reg):
    with pytest.raises(ToolError):
        reg.call("calculator", {"expression": "1/0"})


def test_search_mock(reg):
    out = reg.call("search", {"query": "KV cache 是什么"})
    assert "KV Cache" in out


def test_weather_unknown_city(reg):
    with pytest.raises(ToolError):
        reg.call("weather", {"city": "火星"})


def test_todo_session_scoped(reg):
    s1 = Session("u::w1", "u", "w1")
    s2 = Session("u::w2", "u", "w2")
    reg.call("todo", {"action": "add", "content": "A任务"}, session=s1)
    reg.call("todo", {"action": "add", "content": "B任务"}, session=s2)
    assert "A任务" in reg.call("todo", {"action": "list"}, session=s1)
    assert "B任务" not in reg.call("todo", {"action": "list"}, session=s1)
    reg.call("todo", {"action": "done", "index": 1}, session=s1)
    assert "[x] A任务" in reg.call("todo", {"action": "list"}, session=s1)


def test_todo_bad_index(reg):
    s = Session("u::w", "u", "w")
    with pytest.raises(ToolError):
        reg.call("todo", {"action": "done", "index": 5}, session=s)


def test_unknown_tool_and_missing_args(reg):
    with pytest.raises(ToolError):
        reg.call("nope", {})
    with pytest.raises(ToolError):
        reg.call("calculator", {})  # 缺 required expression


def test_custom_registration():
    reg = ToolRegistry()

    @reg.tool("echo", "回显", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]})
    def echo(x, session=None):
        return x

    assert reg.call("echo", {"x": "hi"}) == "hi"
