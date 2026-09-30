"""HTTP / WebSocket API.

REST
  POST   /sessions                              -> {"session_id"}
  POST   /sessions/{sid}/chunks  {text, t}      -> controller decision for one transcript chunk
  POST   /sessions/{sid}/end     {t}            -> final output record for the utterance
  POST   /sessions/{sid}/utterance {text}       -> replay a full utterance as timed chunks
  GET    /sessions/{sid}/telemetry              -> all telemetry events of the session
  DELETE /sessions/{sid}                        -> drop session memory
  GET    /health
WebSocket
  /ws/{sid}  send {"type":"chunk","text":..,"t":..} or {"type":"end","t":..};
             receives {"type":"decision",...}, {"type":"token",...}, {"type":"final",...}
"""
import asyncio
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .engine import StreamingRAGEngine

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Streaming Live RAG", version="1.0")
engine: StreamingRAGEngine | None = None


def get_engine() -> StreamingRAGEngine:
    global engine
    if engine is None:
        engine = StreamingRAGEngine()
    return engine


@app.on_event("startup")
def _startup():
    get_engine()


class ChunkIn(BaseModel):
    text: str
    t: float


class EndIn(BaseModel):
    t: float


class UtteranceIn(BaseModel):
    text: str
    realtime: bool = False


def _session(sid):
    e = get_engine()
    try:
        e.store.get(sid)
    except KeyError:
        raise HTTPException(404, f"unknown session {sid}")
    return e


@app.get("/health")
def health():
    e = get_engine()
    return {"status": "ok", "chunks": len(e.retriever.chunks), "retrieval_mode": e.retriever.mode,
            "llm": e.cfg.llm_model if e.llm.available else None, "index_ms": e.index_ms}


@app.post("/sessions")
def new_session():
    return {"session_id": get_engine().new_session()}


@app.post("/sessions/{sid}/chunks")
def post_chunk(sid: str, body: ChunkIn):
    return _session(sid).on_chunk(sid, body.text, body.t)


@app.post("/sessions/{sid}/end")
def post_end(sid: str, body: EndIn):
    return _session(sid).end_utterance(sid, body.t)


@app.post("/sessions/{sid}/utterance")
def post_utterance(sid: str, body: UtteranceIn):
    return _session(sid).run_utterance(sid, body.text, realtime=body.realtime)


@app.get("/sessions/{sid}/telemetry")
def telemetry(sid: str):
    return _session(sid).tel[sid].events


@app.delete("/sessions/{sid}")
def delete_session(sid: str):
    _session(sid).end_session(sid)
    return {"deleted": sid}


@app.websocket("/ws/{sid}")
async def ws(sid: str, sock: WebSocket):
    await sock.accept()
    e = get_engine()
    try:
        e.store.get(sid)
    except KeyError:
        await sock.send_json({"type": "error", "detail": "unknown session"})
        await sock.close()
        return
    loop = asyncio.get_running_loop()
    try:
        while True:
            msg = await sock.receive_json()
            if msg.get("type") == "chunk":
                out = await loop.run_in_executor(None, e.on_chunk, sid, msg["text"], float(msg["t"]))
                await sock.send_json({"type": "decision", **out})
            elif msg.get("type") == "end":
                q: asyncio.Queue = asyncio.Queue()

                def on_token(tok):
                    loop.call_soon_threadsafe(q.put_nowait, tok)

                fut = loop.run_in_executor(None, lambda: e.end_utterance(sid, float(msg["t"]), on_token))
                while not fut.done() or not q.empty():
                    try:
                        tok = await asyncio.wait_for(q.get(), timeout=0.05)
                        await sock.send_json({"type": "token", "text": tok})
                    except asyncio.TimeoutError:
                        pass
                rec = await fut
                rec.pop("grounding_details", None)
                await sock.send_json({"type": "final", **rec})
    except WebSocketDisconnect:
        pass
