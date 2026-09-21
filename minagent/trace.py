# -*- coding: utf-8 -*-
"""执行 trace：每个事件一行 JSONL，落盘 + 内存双份，便于复盘与测试断言。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Tracer:
    def __init__(self, session_id: str, path: Optional[Path] = None) -> None:
        self.session_id = session_id
        self.events: List[Dict[str, Any]] = []
        self.path = Path(path) if path else None

    def log(self, kind: str, **fields: Any) -> None:
        event = {"ts": round(time.time(), 3), "session": self.session_id, "kind": kind, **fields}
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def kinds(self) -> List[str]:
        return [e["kind"] for e in self.events]
