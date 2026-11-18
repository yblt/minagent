# -*- coding: utf-8 -*-
"""用户级长期记忆：跨 session 共享，与 session 内的历史/待办快照分层。

三个决策点（README 有对应说明）：
- 写入时机：每轮对话收尾时从用户输入里规则抽取显式事实（"记住/我叫/我的X是Y/
  我喜欢/以后都"），不依赖 LLM，可离线测试；extractor 可注入换成 LLM 抽取。
- 召回时机：组 context 前，用当前用户输入与记忆条目做关键词重叠打分，取 top-k。
  中文按字二元组、英文数字按词切分，无向量依赖。
- 放置方式：召回结果注入 system prompt 的独立段（"用户长期记忆"），
  不进对话历史——与待办快照同理：它是"当前事实"不是"发生过的事"，
  放历史会被压缩丢掉，放 system 永远可见且成本恒定。
隔离边界：user 级共享（窗口1 记的事实窗口2 能召回），不同 user 完全隔离。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# 显式事实句式：命中即整句入库（保留主语，避免"叫杨彬林"这种丢主语碎片）
_FACT_PATTERNS = [
    r"记住[：:，,]?\s*[^，。！？!?\n]{2,80}",
    r"我叫[^，。！？!?,.\s\n]{1,20}",
    r"我的[^，。！？!?,.\s\n]{1,20}是[^，。！？!?,.\s\n]{1,40}",
    r"我(?:喜欢|讨厌|不吃)[^，。！？!?,.\s\n]{1,40}",
    r"我对[^，。！？!?,.\s\n]{1,20}过敏",
    r"以后(?:都|请|一律|默认)[^，。！？!?\n]{2,60}",
]

# 疑问/指令高频二元组：参与打分会让"帮我查一下"召回一切，剔除
_STOP_BIGRAMS = {"帮我", "一下", "然后", "什么", "怎么", "可以", "请问", "查一", "一个"}


def tokenize(text: str) -> List[str]:
    """英文数字按词、中文按字二元组。'北京天气' → ['北京','京天','天气']。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[一-鿿]+", text):
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in tokens if t not in _STOP_BIGRAMS]


def extract_facts(user_input: str) -> List[str]:
    facts: List[str] = []
    for pat in _FACT_PATTERNS:
        for m in re.finditer(pat, user_input):
            fact = m.group(0).strip()
            if fact not in facts:
                facts.append(fact)
    # 嵌套去重："记住我的X是Y"同时命中两条模式时，只留最长的那句
    facts.sort(key=len, reverse=True)
    kept: List[str] = []
    for f in facts:
        if not any(f in longer for longer in kept):
            kept.append(f)
    return kept


@dataclass
class MemoryItem:
    text: str
    source: str  # 来源 session_id，供追溯
    created_at: float = field(default_factory=time.time)
    tokens: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(self.text)


class MemoryStore:
    def __init__(
        self,
        persist_path: Optional[Path] = None,
        top_k: int = 3,
        max_per_user: int = 200,
        extractor: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.top_k = top_k
        self.max_per_user = max_per_user
        self.extractor = extractor or extract_facts
        self._items: Dict[str, List[MemoryItem]] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            for user, rows in raw.items():
                self._items[user] = [
                    MemoryItem(text=r["text"], source=r["source"], created_at=r["created_at"])
                    for r in rows
                ]

    def remember(self, user_id: str, text: str, source: str) -> bool:
        """同文去重；超过 max_per_user 淘汰最旧。返回是否真的写入。"""
        items = self._items.setdefault(user_id, [])
        if any(it.text == text for it in items):
            return False
        items.append(MemoryItem(text=text, source=source))
        if len(items) > self.max_per_user:
            del items[: len(items) - self.max_per_user]
        self._save()
        return True

    def recall(self, user_id: str, query: str, top_k: Optional[int] = None) -> List[MemoryItem]:
        """关键词重叠打分：overlap / sqrt(记忆长度)，长记忆不因词多而占便宜。"""
        q = set(tokenize(query))
        if not q:
            return []
        scored = []
        for it in self._items.get(user_id, []):
            overlap = len(q.intersection(it.tokens))
            if overlap > 0:
                scored.append((overlap / (len(it.tokens) ** 0.5 or 1), it))
        scored.sort(key=lambda p: (-p[0], -p[1].created_at))
        return [it for _, it in scored[: (top_k if top_k is not None else self.top_k)]]

    def count(self, user_id: str) -> int:
        return len(self._items.get(user_id, []))

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            user: [
                {"text": it.text, "source": it.source, "created_at": it.created_at}
                for it in items
            ]
            for user, items in self._items.items()
        }
        self.persist_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")