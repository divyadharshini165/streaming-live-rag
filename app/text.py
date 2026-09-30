"""Lightweight, language-level text helpers (no corpus- or test-specific rules)."""
import re

STOPWORDS = set("""
a an the and or but if then so of to in on at for from by with about into over under
is are was were be been being am do does did done have has had having
i me my mine we us our you your he him his she her it its they them their
this that these those there here what which who whom whose when where why how
can could should would will shall may might must please tell know want need like
just also some any all as not no yes than too very much many get got give let lets
make made one im i'm it's that's what's there's ok okay hey hi hello thanks thank
well um uh so actually really kind sort
plan planning organise organize organising organizing arrange arranging host hosting summarize summarise
explain check find understand looking trying wondering hoping mean guess think
fit suitable good best right possible option options detail details information info
rule rules rupee rupees rs bucks works
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.pop("zero"); NUMBER_WORDS.pop("one")  # "which one" is not a number
NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "hundred": "100",
                     "thousand": "000"})
CITE_RE = re.compile(r"Doc_\d+\s*§\s*[\w.\-]+")


def tokenize(text: str) -> list[str]:
    return [NUMBER_WORDS.get(t, t) for t in TOKEN_RE.findall(text.lower().replace("₹", " "))]


def stem(w: str) -> str:
    w = _strip_suffix(w)
    return w[:-1] if len(w) > 4 and w.endswith("e") else w  # reimburse / reimbursed -> reimburs


def _strip_suffix(w: str) -> str:
    if w.isdigit() or len(w) <= 3:
        return w
    for suf in ("ies", "ed", "s"):  # no "-ing": "parking" must not match "Tech Park"
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)] + ("y" if suf == "ies" else "")
            if suf == "ed" and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]  # cancelled -> cancel
            if suf == "s" and base.endswith("s"):
                return w  # "class", "process"
            return base
    return w


def content_words(text: str) -> list[str]:
    """Original-form content words, in order."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def content_terms(text: str) -> list[str]:
    """Stemmed content terms used for matching."""
    return [stem(t) for t in content_words(text)]


TAG_AFTER_PERIOD = re.compile(r"([.!?])\s*(\[[^\]]*Doc_[^\]]*\])")


def attach_tags(text: str) -> str:
    """'Fact. [Doc_1 §2] Next' -> 'Fact [Doc_1 §2]. Next' so each tag stays with its sentence."""
    return TAG_AFTER_PERIOD.sub(r" \2\1", text)


def split_sentences(text: str) -> list[str]:
    text = attach_tags(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z₹0-9\[•])|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize_cite(c: str) -> str:
    m = re.match(r"(Doc_\d+)\s*§\s*([\w.\-]+)", c)
    return f"{m.group(1)} §{m.group(2)}" if m else c.strip()


def extract_citations(text: str) -> list[str]:
    seen, out = set(), []
    for c in CITE_RE.findall(text):
        c = normalize_cite(c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
