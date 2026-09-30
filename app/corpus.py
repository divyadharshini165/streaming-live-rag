"""Loads any folder of .md / .txt / .pdf files and splits it into citable chunks.

Chunk IDs look like "Doc_07 §4". If a file name starts with Doc_<n> that ID is kept,
otherwise IDs are assigned in sorted file order. Markdown '##' headings become sections;
files without headings are split into ~180-word windows.
"""
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}
WINDOW_WORDS = 180


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.doc_title}. {self.section_title}. {self.text}"


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _windows(text: str):
    words = text.split()
    for i in range(0, len(words), WINDOW_WORDS):
        yield " ".join(words[i:i + WINDOW_WORDS])


def _parse(doc_id: str, fallback_title: str, raw: str) -> list[Chunk]:
    title_m = re.search(r"^#\s+(.+)$", raw, re.M)
    title = title_m.group(1).strip() if title_m else fallback_title
    parts = re.split(r"^#{2,3}\s+(.+)$", raw, flags=re.M)
    chunks = []
    if len(parts) > 1:
        # parts = [preamble, head1, body1, head2, body2, ...]
        for n, i in enumerate(range(1, len(parts), 2), start=1):
            head, body = parts[i].strip(), " ".join(parts[i + 1].split())
            if not body:
                continue
            m = re.search(r"§\s*([\w.]+)", head)
            sec = m.group(1) if m else str(n)
            sec_title = re.sub(r"§\s*[\w.]+\s*", "", head).strip() or head
            wins = list(_windows(body))
            for w_i, w in enumerate(wins):
                sid = sec if len(wins) == 1 else f"{sec}-{w_i + 1}"
                chunks.append(Chunk(f"{doc_id} §{sid}", doc_id, title, sec_title, w))
    else:
        body = re.sub(r"^#\s+.+$", "", raw, flags=re.M)
        for n, w in enumerate(_windows(" ".join(body.split())), start=1):
            chunks.append(Chunk(f"{doc_id} §{n}", doc_id, title, f"Part {n}", w))
    return chunks


def load_corpus(corpus_dir: str) -> list[Chunk]:
    files = sorted(p for p in Path(corpus_dir).rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No .md/.txt/.pdf files found in {corpus_dir}")
    used, next_id, chunks = set(), 1, []
    named = {}
    for p in files:
        m = re.match(r"(doc_\d+)", p.stem, re.I)
        if m:
            named[p] = "Doc_" + m.group(1).split("_")[1]
            used.add(named[p])
    for p in files:
        doc_id = named.get(p)
        if not doc_id:
            while f"Doc_{next_id:02d}" in used:
                next_id += 1
            doc_id = f"Doc_{next_id:02d}"
            used.add(doc_id)
        fallback = p.stem.replace("_", " ")
        chunks.extend(_parse(doc_id, fallback, _read(p)))
    return chunks
