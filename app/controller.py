"""Retrieval Controller.

Decides, for every incoming transcript chunk, whether to WAIT, RETRIEVE (provisionally,
before the utterance ends) or SUPPRESS retrieval, and classifies the finished turn as
new_query / late_detail / presentation / conversational.

Rules are generic language cues plus corpus statistics (vocabulary built from the indexed
corpus), so nothing is tied to particular test utterances.
"""
import re

from .text import content_words, stem, tokenize

WAIT, RETRIEVE, SUPPRESS, NO_RETRIEVAL = "wait", "retrieve", "suppress", "no_retrieval"

PRESENTATION = re.compile(
    r"\b(repeat|rephrase|reword|restate|reformat|shorten|shorter|simplify|simpler|translate|"
    r"summari[sz]e (that|this|it|your|the (last|previous|above))|"
    r"(in|as) (\d+|one|two|three|four|five) (bullet|point|line|sentence)s?|bullet points?|"
    r"as a (table|list)|in simple (words|terms)|say (that|it) again|tl;?dr)\b", re.I)
REFERS_BACK = re.compile(r"\b(that|this|it|your (last|previous)|the (last|previous|above) answer|again)\b", re.I)
CONVERSATIONAL = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok(ay)?|cool|great|nice|got it|perfect|bye|good (morning|evening)|"
    r"can you hear me|are you there)\b", re.I)
LATE_DETAIL = re.compile(
    r"\b(actually|instead|what if|it'?s for|it is for|it was|correction|sorry|oh and|one more thing|"
    r"i meant|make it|change (it|that)|the \w+ (was|is|will be|were))\b", re.I)
QUESTION = re.compile(
    r"^\s*(what|how|which|where|when|who|why|can|could|is|are|do|does|should|tell|explain|summari[sz]e|"
    r"list|give|show|i need|i want|we need|we want|find|compare)\b", re.I)
DANGLING = {"in", "at", "for", "to", "of", "and", "or", "the", "a", "an", "with", "on", "from", "by",
            "about", "is", "are", "my", "our", "i", "we", "need", "want", "which", "what", "their", "its"}


class RetrievalController:
    def __init__(self, retriever, min_terms: int = 3):
        self.r = retriever
        self.min_terms = min_terms

    def informative_words(self, text: str) -> list[str]:
        return [w for w in content_words(text) if self.r.informative(stem(w))]

    def classify(self, text: str, has_prior: bool) -> str:
        """Turn type from text seen so far."""
        info = self.informative_words(text)
        if has_prior and PRESENTATION.search(text) and (REFERS_BACK.search(text) or len(info) <= 2):
            return "presentation"
        if CONVERSATIONAL.search(text) and len(info) <= 1:
            return "conversational"
        if not info:
            return "conversational"
        if has_prior and (LATE_DETAIL.search(text) or not QUESTION.search(text)):
            return "late_detail"
        return "new_query"

    def on_chunk(self, text_so_far: str, has_prior: bool, already_retrieved: bool):
        """Returns (decision, reason, provisional_query | None)."""
        kind = self.classify(text_so_far, has_prior)
        if kind == "presentation":
            return SUPPRESS, "presentation_restructure", None
        if kind == "conversational":
            return WAIT, "no_retrievable_intent_yet", None
        info = self.informative_words(text_so_far)
        toks = tokenize(text_so_far)
        dangling = bool(toks) and toks[-1] in DANGLING
        if already_retrieved:
            return WAIT, "accumulating", None
        if kind == "late_detail":
            if len(info) >= 1 and not dangling:
                return RETRIEVE, "late_detail_stable", None
            return WAIT, "late_detail_unstable", None
        if dangling and len(info) < 5:
            return WAIT, "intent_unstable", None
        rare = [w for w in info if w.isdigit() or self.r.idf.get(stem(w), 0) >= 1.5]
        if len(info) >= self.min_terms or (len(info) >= 2 and rare):
            return RETRIEVE, "intent_stable", " ".join(dict.fromkeys(info))
        return WAIT, "insufficient_intent", None
