# -*- coding: utf-8 -*-
"""memory 子系统测试：抽取 / 去重与淘汰 / 召回打分 / 跨窗口集成 / 持久化 / 放置方式。"""
import json

from minagent import Agent, MemoryStore, MockLLM, SessionStore, extract_facts
from minagent.memory import tokenize


def ja(answer: str) -> str:
    return json.dumps({"thought": "t", "answer": answer}, ensure_ascii=False)


# -- 抽取 ------------------------------------------------------------------

def test_extract_facts_patterns():
    facts = extract_facts("你好，记住我的工位在3楼。我叫杨彬林，我喜欢喝美式，以后都用中文回复我")
    assert "记住我的工位在3楼" in facts
    assert "我叫杨彬林" in facts
    assert "我喜欢喝美式" in facts
    assert "以后都用中文回复我" in facts


def test_extract_facts_ignores_plain_chat():
    assert extract_facts("帮我查一下北京天气，然后记个待办") == []
    assert extract_facts("刚才天气多少度来着？") == []


def test_extract_facts_allergy():
    assert extract_facts("我对花生过敏，别推荐含花生的") == ["我对花生过敏"]


# -- 存储 ------------------------------------------------------------------

def test_remember_dedup_and_eviction():
    m = MemoryStore(max_per_user=3)
    assert m.remember("u", "事实一", "u::w1")
    assert not m.remember("u", "事实一", "u::w2")  # 同文去重
    for t in ("事实二", "事实三", "事实四"):
        m.remember("u", t, "u::w1")
    assert m.count("u") == 3  # 最旧的"事实一"被淘汰


def test_recall_scores_and_isolation():
    m = MemoryStore()
    m.remember("u", "我的常驻城市是北京", "u::w1")
    m.remember("u", "我喜欢喝美式", "u::w1")
    hits = m.recall("u", "北京天气怎么样")
    assert [h.text for h in hits] == ["我的常驻城市是北京"]
    assert m.recall("u", "推荐点喝的") == []
    assert m.recall("other", "北京天气") == []  # 用户隔离


def test_recall_top_k():
    m = MemoryStore(top_k=1)
    m.remember("u", "我的常驻城市是北京", "s")
    m.remember("u", "我的公司在北京海淀", "s")
    assert len(m.recall("u", "北京")) == 1


def test_tokenize_cjk_bigrams_and_ascii():
    toks = tokenize("查一下KV cache")
    assert "kv" in toks and "cache" in toks
    assert "一下" not in toks  # 停用二元组
    assert "北京" in tokenize("北京天气")


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "mem.json"
    m = MemoryStore(persist_path=p)
    m.remember("u", "我叫杨彬林", "u::w1")
    m2 = MemoryStore(persist_path=p)
    assert [i.text for i in m2.recall("u", "我叫什么")] == ["我叫杨彬林"]


# -- 与 Agent 集成 ----------------------------------------------------------

def test_memory_cross_window_recall_and_placement():
    store = SessionStore()
    mem = MemoryStore()
    llm1 = MockLLM([ja("好的，已记住")])
    a1 = Agent(llm1, store=store, memory=mem)
    a1.chat("userA", "win1", "记住我的常驻城市是北京")
    assert mem.count("userA") == 1

    # 窗口2：全新 session，历史为空，但 system prompt 应带上召回的长期记忆
    llm2 = MockLLM([ja("你常驻北京")])
    a2 = Agent(llm2, store=store, memory=mem)
    a2.chat("userA", "win2", "我常驻城市哪来着？")
    sent = llm2.calls[0]
    assert sent[0]["role"] == "system"
    assert "用户长期记忆" in sent[0]["content"]
    assert "我的常驻城市是北京" in sent[0]["content"]
    # 放置方式：只在 system，不混入对话历史
    assert all("常驻城市是北京" not in m["content"] for m in sent[1:] if m["role"] != "system")
    # session 历史仍隔离：窗口2 看不到窗口1 的消息
    assert len(store.get("userA::win2").messages) == 2  # user + assistant


def test_memory_not_recalled_when_irrelevant():
    mem = MemoryStore()
    mem.remember("u", "我喜欢喝美式", "u::w1")
    llm = MockLLM([ja("晴")])
    Agent(llm, memory=mem).chat("u", "w2", "查一下北京天气")
    assert "用户长期记忆" not in llm.calls[0][0]["content"]


def test_memory_events_in_trace():
    mem = MemoryStore()
    llm1 = MockLLM([ja("好")])
    r1 = Agent(llm1, memory=mem).chat("u", "w1", "我叫杨彬林")
    assert "memory_write" in [e["kind"] for e in r1.trace]
    llm2 = MockLLM([ja("你叫杨彬林")])
    r2 = Agent(llm2, memory=mem).chat("u", "w2", "我叫什么名字？")
    assert "memory_recall" in [e["kind"] for e in r2.trace]


def test_memory_write_even_when_llm_down():
    class BrokenLLM:
        def chat(self, messages):
            from minagent.llm import LLMError
            raise LLMError("down")

    mem = MemoryStore()
    Agent(BrokenLLM(), memory=mem, retry_backoff=0).chat("u", "w", "我叫杨彬林")
    assert mem.count("u") == 1  # 用户说出的事实不依赖回答成功