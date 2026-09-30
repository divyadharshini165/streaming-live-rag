"""Ephemeral, session-scoped memory. Nothing is persisted across sessions."""
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class AnswerVersion:
    version: int
    answer: str
    citations: list
    uncertainty: str | None
    sub_queries: list
    evidence_ids: list
    turn_id: int
    turn_type: str


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    versions: list = field(default_factory=list)       # AnswerVersion lineage
    evidence: dict = field(default_factory=dict)       # chunk_id -> Chunk seen in this session
    topic_queries: list = field(default_factory=list)  # sub-queries of the active topic
    turn_count: int = 0

    @property
    def current(self) -> AnswerVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def has_answer(self) -> bool:
        return bool(self.versions)


class SessionStore:
    def __init__(self, ttl_s=3600):
        self._s: dict[str, Session] = {}
        self.ttl_s = ttl_s

    def new(self) -> Session:
        self._gc()
        s = Session()
        self._s[s.session_id] = s
        return s

    def get(self, sid) -> Session:
        if sid not in self._s:
            raise KeyError(sid)
        return self._s[sid]

    def drop(self, sid):
        self._s.pop(sid, None)

    def _gc(self):
        now = time.time()
        for k in [k for k, s in self._s.items() if now - s.created > self.ttl_s]:
            del self._s[k]
