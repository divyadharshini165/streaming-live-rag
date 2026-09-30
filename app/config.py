"""All runtime settings come from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name, default):
    return os.getenv(name, default)


@dataclass
class Config:
    corpus_dir: str = field(default_factory=lambda: _env("CORPUS_DIR", "data/corpus"))
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", "logs"))

    # Retrieval
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))  # hybrid | sparse | dense
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    evidence_cap: int = field(default_factory=lambda: int(_env("EVIDENCE_CAP", "8")))

    # Pipeline switches (used for baseline and ablations)
    streaming: bool = True          # False = wait for utterance end (baseline)
    decomposition: bool = True      # False = one query per utterance
    refinement: bool = True         # False = every turn restarts from scratch
    suppression: bool = True        # False = always retrieve
    decomposer_mode: str = field(default_factory=lambda: _env("DECOMPOSER_MODE", "rule"))  # rule | llm

    # LLM (any OpenAI-compatible endpoint: Groq, Gemini, OpenRouter, Ollama ...)
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama-3.1-8b-instant"))
    price_in_per_m: float = field(default_factory=lambda: float(_env("PRICE_IN_PER_M_USD", "0.05")))
    price_out_per_m: float = field(default_factory=lambda: float(_env("PRICE_OUT_PER_M_USD", "0.08")))

    # Simulated speech
    words_per_second: float = field(default_factory=lambda: float(_env("WORDS_PER_SECOND", "2.8")))
    words_per_chunk: int = field(default_factory=lambda: int(_env("WORDS_PER_CHUNK", "4")))
    endpoint_silence_s: float = 0.3
