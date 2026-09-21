# -*- coding: utf-8 -*-
"""Session 管理：同一用户的不同窗口彼此隔离，可随时续聊。

session_id = "<user_id>::<window_id>"，每个 session 独立保存：
- messages：对话历史（含工具调用与结果，供追问回溯）
- state：工具侧状态（如 todo 列表），不混入 context，按需由工具读写
可选落盘为 JSON 文件（persist_dir），重启后可续聊。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

Message = Dict[str, str]


@dataclass
class Session:
    session_id: str
    user_id: str
    window_id: str
    messages: List[Message] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "window_id": self.window_id,
            "messages": self.messages,
            "state": self.state,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        return cls(
            session_id=d["session_id"],
            user_id=d["user_id"],
            window_id=d["window_id"],
            messages=d.get("messages", []),
            state=d.get("state", {}),
            updated_at=d.get("updated_at", time.time()),
        )


class SessionStore:
    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        self._sessions: Dict[str, Session] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            for f in self.persist_dir.glob("*.json"):
                s = Session.from_dict(json.loads(f.read_text(encoding="utf-8")))
                self._sessions[s.session_id] = s

    def get_or_create(self, user_id: str, window_id: str) -> Session:
        sid = f"{user_id}::{window_id}"
        if sid not in self._sessions:
            self._sessions[sid] = Session(sid, user_id, window_id)
        return self._sessions[sid]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        if self.persist_dir:
            safe = session.session_id.replace(":", "_").replace("\\", "_")
            (self.persist_dir / f"{safe}.json").write_text(
                json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def count(self) -> int:
        return len(self._sessions)
