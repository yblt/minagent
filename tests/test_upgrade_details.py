# -*- coding: utf-8 -*-
"""本轮升级的细节测试：出站消息映射 / 压缩边界对齐 / read_docs。"""
import pytest

from minagent import ContextManager, ToolError, build_default_registry, to_wire_messages


# -- 出站映射（真实 OpenAI 兼容 API 不接受无 tool_call_id 的 role=tool） ------

def test_wire_messages_maps_tool_role():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": '{"action":…}'},
        {"role": "tool", "content": "北京: 晴 26℃"},
    ]
    wire = to_wire_messages(msgs)
    assert {m["role"] for m in wire} <= {"system", "user", "assistant"}
    assert wire[3] == {"role": "user", "content": "[工具结果] 北京: 晴 26℃"}
    assert msgs[3]["role"] == "tool"  # 存储层不被改动，trace/追问仍可用


# -- 压缩边界：assistant(action) 与 tool 结果不能被切进摘要两侧 ---------------

def test_compression_boundary_keeps_tool_pair_together():
    ctx = ContextManager(max_messages=6, keep_recent=3)
    history = []
    for i in range(2):
        history += [
            {"role": "user", "content": f"问题{i}"},
            {"role": "assistant", "content": f"action{i}"},
            {"role": "tool", "content": f"结果{i}"},
            {"role": "assistant", "content": f"答案{i}"},
        ]
    # 共 8 条 > 6；naive 切点 len-3=5 落在 tool 上，应回退到 4（user 边界）
    out = ctx.build("sys", history)
    assert out[0]["content"] == "sys"
    assert out[1]["content"].startswith("[历史摘要]")
    assert out[2]["role"] == "user" and out[2]["content"] == "问题1"
    # 完整 turn（user/assistant/tool/assistant）都在 recent 侧
    assert [m["role"] for m in out[2:]] == ["user", "assistant", "tool", "assistant"]


def test_compression_skipped_when_no_user_boundary():
    ctx = ContextManager(max_messages=2, keep_recent=2)
    history = [
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "b"},
    ]
    out = ctx.build("sys", history)
    assert len(out) == 4  # 找不到用户边界（split 回退到 0）→ 不压缩，全保留


# -- read_docs ---------------------------------------------------------------

@pytest.fixture()
def docs_reg(tmp_path):
    (tmp_path / "guide.md").write_text("hello docs", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    return build_default_registry(docs_dir=tmp_path)


def test_read_docs_list_and_read(docs_reg):
    assert "guide.md" in docs_reg.call("read_docs", {})
    assert docs_reg.call("read_docs", {"name": "guide.md"}) == "hello docs"


def test_read_docs_rejects_traversal(docs_reg):
    with pytest.raises(ToolError, match="outside"):
        docs_reg.call("read_docs", {"name": "../secrets.txt"})


def test_read_docs_missing_file(docs_reg):
    with pytest.raises(ToolError, match="no such doc"):
        docs_reg.call("read_docs", {"name": "nope.md"})