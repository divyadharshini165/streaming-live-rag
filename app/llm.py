"""Thin client for any OpenAI-compatible chat endpoint (Groq, Gemini, OpenRouter, Ollama).

If LLM_API_KEY is empty the system runs fully offline: rule-based decomposition and
extractive (sentence-level) grounded synthesis.
"""
import logging
import time

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        if cfg.llm_api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=30)

    @property
    def available(self) -> bool:
        return self.client is not None

    def cost(self, tin: int, tout: int) -> float:
        return (tin * self.cfg.price_in_per_m + tout * self.cfg.price_out_per_m) / 1_000_000

    def complete(self, messages, max_tokens=400, on_token=None, json_mode=False):
        """Returns dict(text, tokens_in, tokens_out, ttft_ms, latency_ms). Streams when possible."""
        t0 = time.perf_counter()
        kwargs = dict(model=self.cfg.llm_model, messages=messages, temperature=0, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            ms = (time.perf_counter() - t0) * 1000
            return dict(text=text, tokens_in=getattr(u, "prompt_tokens", 0) or _est(messages),
                        tokens_out=getattr(u, "completion_tokens", 0) or len(text) // 4,
                        ttft_ms=ms, latency_ms=ms)
        try:
            stream = self.client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kwargs)
        except Exception:
            stream = self.client.chat.completions.create(stream=True, **kwargs)
        parts, ttft, usage = [], None, None
        for ev in stream:
            if getattr(ev, "usage", None):
                usage = ev.usage
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                tok = ev.choices[0].delta.content
                parts.append(tok)
                if on_token:
                    on_token(tok)
        text = "".join(parts)
        return dict(text=text,
                    tokens_in=getattr(usage, "prompt_tokens", None) or _est(messages),
                    tokens_out=getattr(usage, "completion_tokens", None) or max(1, len(text) // 4),
                    ttft_ms=ttft or (time.perf_counter() - t0) * 1000,
                    latency_ms=(time.perf_counter() - t0) * 1000)


def _est(messages) -> int:
    return sum(len(m["content"]) for m in messages) // 4
