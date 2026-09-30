"""Multi-Intent Decomposer.

Rule mode (default, zero cost): split the utterance at clause boundaries, carry shared
context (named entities + the main topic word) into short clauses, merge fragments, and drop
near-duplicate sub-queries (guards against over-fragmentation).
LLM mode: one JSON call; falls back to rule mode on any error.
"""
import json
import re

from .text import NUMBER_WORDS, STOPWORDS, content_terms, content_words, jaccard, stem

SPLIT_RE = re.compile(r"[,;?!]+|\.(?!\d)|\b(?:and also|and then|as well as|along with|and|also|plus)\b", re.I)
GROUP_SIZE = re.compile(r"\b(\d+|ten|twenty|thirty|forty|fifty|sixty|hundred)\s+(people|person|persons|participants|attendees|pax|guests|members|employees)\b", re.I)
PRONOUN = re.compile(r"\b(it|its|them|their|they|there|that|those|these|the (hall|venue|room|place|trip|event))\b", re.I)
FILLER = re.compile(r"\b(actually|instead|sorry|oh|what if|correction|i meant)\b", re.I)
MAX_SUBQ = 4


class Decomposer:
    def __init__(self, retriever, llm=None, mode="rule"):
        self.r = retriever
        self.llm = llm
        self.mode = mode
        self.last_usage = None

    # ---------- helpers ----------
    def _info(self, text):
        return [w for w in content_words(text) if self.r.informative(stem(w))]

    def _topic(self, text, n=1):
        """Topic words = query words that also appear in the titles of the best-matching documents."""
        ws = [w for w in dict.fromkeys(self._info(text)) if not w.isdigit()]
        if not ws:
            return []
        titles = set()
        for h in self.r.search(" ".join(ws), 3):
            titles |= set(content_terms(f"{h.chunk.doc_title} {h.chunk.section_title}"))
        on_topic = [w for w in ws if stem(w) in titles] or ws
        return sorted(on_topic, key=lambda w: -self.r.idf.get(stem(w), 0))[:n]

    @staticmethod
    def _entities(text):
        words, out = text.split(), []
        for i, w in enumerate(words):
            c = re.sub(r"[^\w]", "", w)
            if not c:
                continue
            if c.isdigit():
                nxt = re.sub(r"[^\w]", "", words[i + 1]).lower() if i + 1 < len(words) else ""
                out.append(f"{c} {nxt}" if nxt and nxt not in STOPWORDS else c)
            elif c[0].isupper() and i > 0 and c.lower() not in STOPWORDS and not words[i - 1].endswith((".", "?", "!")):
                out.append(c)
        return out

    @staticmethod
    def split_clauses(text, complete_only=False):
        pieces = [p.strip() for p in SPLIT_RE.split(text) if p and p.strip()]
        if complete_only and pieces:
            last = list(SPLIT_RE.finditer(text))
            if not last or last[-1].end() < len(text.rstrip()):
                pieces = pieces[:-1]  # trailing clause still being spoken
        return pieces

    def _dedupe(self, queries):
        out = []
        for q in queries:
            t = content_terms(q)
            if len(t) < 2:
                continue
            if any(jaccard(t, content_terms(o)) >= 0.6 for o in out):
                continue
            out.append(q)
        return out[:MAX_SUBQ]

    # ---------- public ----------
    def rule_decompose(self, text, complete_only=False):
        clauses = self.split_clauses(text, complete_only)
        if not clauses:
            return []
        ents = self._entities(text)
        cap_ents = [e for e in ents if not e[0].isdigit()]
        topic = self._topic(clauses[0])
        queries = []
        for i, cl in enumerate(clauses):
            words = content_words(cl)
            info = self._info(cl)
            if not info:
                continue
            if i == 0:
                q = words
            else:
                refers = bool(PRONOUN.search(cl))
                own_ent = any(e in cl for e in cap_ents)
                extra = []
                if not own_ent and (refers or len(info) <= 2):
                    extra += cap_ents          # carry place / names
                if refers or len(info) <= 1:
                    extra += topic             # carry the main topic word
                have = {x.lower() for x in words}
                extra = [w for w in extra if w.lower() not in have]
                q = words + extra
                if len(info) == 1 and not extra and queries:
                    queries[-1] = queries[-1] + " " + " ".join(words)   # merge fragment
                    continue
            g = GROUP_SIZE.search(cl)
            if g:  # a group size implies a room-capacity need: query place + size + capacity
                size = f"{NUMBER_WORDS.get(g.group(1).lower(), g.group(1))} {g.group(2)}"
                q = [e for e in cap_ents if e.lower() in cl.lower()] + [size, "venue", "capacity", "seats"]
            queries.append(_join(q))
        return self._dedupe(queries)

    def llm_decompose(self, text):
        prompt = [
            {"role": "system", "content":
                "You split one spoken request into the minimal set of distinct, self-contained search queries "
                "(1 to 4). Resolve pronouns and carry shared context (place, group size, topic) into every query. "
                "Never split one question into near-duplicates. If there is only one need, return one query. "
                'Respond only with JSON: {"sub_queries": ["..."]}'},
            {"role": "user", "content": text}]
        res = self.llm.complete(prompt, max_tokens=200, json_mode=True)
        self.last_usage = res
        qs = json.loads(res["text"]).get("sub_queries", [])
        return self._dedupe([q for q in qs if isinstance(q, str)]) or self.rule_decompose(text)

    def decompose(self, text):
        self.last_usage = None
        if self.mode == "llm" and self.llm and self.llm.available:
            try:
                return self.llm_decompose(text)
            except Exception:
                pass
        return self.rule_decompose(text) or [" ".join(content_words(text))]

    def delta_queries(self, detail_text, prior_sub_queries, complete_only=False):
        """Targeted queries for a late-arriving detail: the new constraint + the prior topic."""
        prior = " ".join(prior_sub_queries)
        topic = self._topic(prior, n=3)
        out = []
        for cl in self.split_clauses(FILLER.sub(" ", detail_text), complete_only):
            words = content_words(cl)
            info = [w for w in self._info(cl) if not w.isdigit()]
            if not words:
                continue
            # carry the prior topic only when the detail cannot stand on its own
            needs_topic = bool(PRONOUN.search(cl)) or len(info) <= 1 or not any(w[0].isupper() for w in cl.split()[1:])
            extra = [w for w in topic if w.lower() not in (x.lower() for x in words)] if needs_topic else []
            out.append(_join(words + extra[:1 if len(info) >= 2 else 3]))
        return self._dedupe(out) or ([" ".join(content_words(detail_text) + topic)] if not complete_only else [])


def _join(words):
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return " ".join(out)
