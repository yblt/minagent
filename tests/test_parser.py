# -*- coding: utf-8 -*-
"""LLM 输出解析单测：思考/工具调用/答案/异常格式。"""
from minagent.parser import parse_llm_output


def test_parse_action():
    p = parse_llm_output('{"thought": "先查天气", "action": {"tool": "weather", "args": {"city": "北京"}}}')
    assert p.tool == "weather" and p.args == {"city": "北京"} and p.answer is None and p.error is None
    assert p.thought == "先查天气"


def test_parse_answer():
    p = parse_llm_output('{"thought": "有结果了", "answer": "北京 26℃"}')
    assert p.answer == "北京 26℃" and p.tool is None


def test_parse_fenced_json():
    p = parse_llm_output('好的，我来查。\n```json\n{"thought":"t","answer":"ok"}\n```')
    assert p.answer == "ok"


def test_parse_json_with_noise():
    p = parse_llm_output('前置废话 {"thought":"t","action":{"tool":"todo","args":{"action":"list"}}} 后缀废话')
    assert p.tool == "todo" and p.args == {"action": "list"}


def test_parse_invalid_json_returns_error():
    p = parse_llm_output('{"thought": 坏了, 没有引号')
    assert p.error and "JSON" in p.error


def test_parse_missing_fields():
    p = parse_llm_output('{"thought": "只有思考"}')
    assert p.error and "answer" in p.error
    p = parse_llm_output('{"thought":"t","action":{"args":{}}}')
    assert p.error


def test_parse_non_dict_args():
    p = parse_llm_output('{"action": {"tool": "weather", "args": "北京"}}')
    assert p.error


def test_plain_text_treated_as_answer():
    p = parse_llm_output("北京今天晴，26 摄氏度。")
    assert p.answer == "北京今天晴，26 摄氏度。"


def test_empty_output():
    assert parse_llm_output("").error
