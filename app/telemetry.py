"""Structured observability events, one JSON object per line in logs/<session_id>.jsonl.

Every event carries: ts (unix), session_id, turn_id, event, stream_t_s (position in the
simulated/real audio stream) and wall_ms (processing time since the turn started).
See README "Telemetry schema" for the event list.
"""
import json
import os
import time


class Telemetry:
    def __init__(self, log_dir: str, session_id: str):
        self.session_id = session_id
        self.events = []
        self.turn_id = 0
        self._turn_t0 = time.perf_counter()
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{session_id}.jsonl")

    def start_turn(self, turn_id: int):
        self.turn_id = turn_id
        self._turn_t0 = time.perf_counter()

    def wall_ms(self) -> float:
        return round((time.perf_counter() - self._turn_t0) * 1000, 2)

    def emit(self, event: str, stream_t_s: float | None = None, **payload):
        rec = {"ts": round(time.time(), 3), "session_id": self.session_id, "turn_id": self.turn_id,
               "event": event, "stream_t_s": stream_t_s, "wall_ms": self.wall_ms(), **payload}
        self.events.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def turn_events(self, turn_id: int):
        return [e for e in self.events if e["turn_id"] == turn_id]
