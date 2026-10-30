"""Feature detectors that read a spaCy dependency parse.

Each detector yields the tokens to highlight for one occurrence. Definitions
follow Clark, Martin and Mahowald (2026), "Why is AI-generated language so
hard to understand?", with lexicon heuristics in place of their classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b

from spacy.tokens import Span, Token

from .resources import Resources
from .text import CODE

DEP_BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 30))
ANIMATE_WORDS = {
    "others", "someone", "everyone", "anyone", "somebody", "everybody", "anybody", "folks",
    "individual", "group", "culture", "board", "builder", "community", "audience", "society",
}
INANIMATE_PRONOUNS = {"it", "this", "that", "these", "those", "which", "something", "nothing", "everything", "anything"}
ANIMATE_ENTS = {"PERSON", "NORP", "ORG", "GPE"}
RELATIVIZERS = {"WDT", "WP", "WP$", "WRB"}
# Heads of relative clauses that are adverbial ("the way light interacts"), not object gaps.
ADVERBIAL_HEADS = {"way", "reason", "time", "place", "day", "year", "moment", "extent", "point", "case", "week", "night"}
# Modifier or argument positions where a -ing/-ed "verb" is really a noun or adjective.
NOMINAL_DEPS = {"amod", "compound", "nsubj", "nsubjpass", "dobj", "pobj", "nmod", "acomp", "attr", "appos"}

CLAUSAL_DEPS = {"nsubj", "nsubjpass", "csubj", "dobj", "aux", "auxpass", "prt"}

COUNT_FEATURES = ("inanimate_subject", "stacked_nouns", "reduced_orc", "overt_orc", "denominal_verb")


def pair_hash(a: str, b: str) -> int:
    return int.from_bytes(blake2b(f"{a} {b}".encode(), digest_size=4).digest(), "little")


def is_word(tok: Token) -> bool:
    return not (tok.is_punct or tok.is_space or tok.pos_ in ("PUNCT", "SYM", "SPACE", "X"))


def _is_code(tok: Token) -> bool:
    return tok.text == CODE


def inanimate_subjects(sent: Span, res: Resources):
    for tok in sent:
        if tok.dep_ != "nsubj" or tok.head.pos_ != "VERB" or tok.head.lemma_.lower() not in res.animate_verbs:
            continue
        verb = tok.head
        children = list(verb.children)
        if any(c.dep_ in ("auxpass", "nsubjpass") for c in children):
            continue
        # A bare base-form verb with a noun subject is almost always a mis-parsed imperative.
        if verb.tag_ == "VB" and not any(c.dep_ == "aux" for c in children):
            continue
        subj = tok
        if tok.tag_ == "WDT" and verb.dep_ == "relcl":
            subj = verb.head
        if _is_inanimate(subj, res):
            yield [tok, verb]


def _is_inanimate(tok: Token, res: Resources) -> bool:
    if _is_code(tok):
        return True
    lower, lemma = tok.lower_, tok.lemma_.lower()
    if tok.pos_ == "PRON":
        return lower in INANIMATE_PRONOUNS
    if tok.ent_type_ in ANIMATE_ENTS or lower in ANIMATE_WORDS or lemma in ANIMATE_WORDS:
        return False
    if lemma in res.animate_nouns or lower in res.animate_nouns:
        return False
    if tok.pos_ == "NOUN":
        return tok.tag_ != "VBG" and not lower.endswith("ing")
    # Sentence-initial capitals make common nouns look like names ("Chaos expects").
    return tok.pos_ == "PROPN" and not tok.ent_type_ and tok.i == tok.sent.start and lower.isalpha()


def stacked_nouns(sent: Span, res: Resources):
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)
    for tok in sent:
        if tok.pos_ != "NOUN" or tok.dep_ == "compound" or not _stack_noun(tok):
            continue
        chain = sorted(_compound_nouns(tok), key=lambda t: t.i)
        if len(chain) < 3:
            continue
        # Verbless list items ("Build symbol table") tag the leading verb as a noun.
        if chain[0].i == sent.start and not has_verb:
            continue
        yield chain


def _stack_noun(tok: Token) -> bool:
    return tok.is_alpha and len(tok) > 1 and not _is_code(tok)


def _compound_nouns(tok: Token) -> list[Token]:
    out = [tok]
    for child in tok.children:
        if child.dep_ == "compound" and child.pos_ == "NOUN" and _stack_noun(child):
            out.extend(_compound_nouns(child))
    return out


def object_relatives(sent: Span, res: Resources):
    """Yield (kind, tokens) for object relative clauses, kind 'reduced' or 'overt'."""
    for verb in sent:
        head = verb.head
        if verb.dep_ != "relcl" or head.pos_ not in ("NOUN", "PROPN", "PRON"):
            continue
        if head.lemma_.lower() in ADVERBIAL_HEADS or head.lower_ == "what":
            continue
        children = list(verb.children)
        if any(c.dep_ in ("auxpass", "nsubjpass") for c in children):
            continue
        if any(c.dep_ == "aux" and c.lower_ == "to" for c in children):
            continue
        left = [t for t in verb.subtree if t.i < verb.i]
        rel = [t for t in left if t.tag_ in RELATIVIZERS or (t.lower_ == "that" and t.head == verb)]
        subj = [c for c in children if c.dep_ == "nsubj" and c not in rel]
        if not subj:
            continue
        objects = [c for c in children if c.dep_ in ("dobj", "ccomp") and c not in rel]
        if objects:
            continue
        if rel:
            if any(r.dep_ == "dobj" and r.head == verb and r.lower_ != "what" for r in rel):
                yield "overt", [head, *rel, verb]
        else:
            yield "reduced", [head, *subj, verb]


def denominal_verbs(sent: Span, res: Resources):
    for tok in sent:
        if tok.pos_ != "VERB" or tok.dep_ in NOMINAL_DEPS or tok.lemma_.lower() not in res.denominal:
            continue
        # Nouns mis-tagged as verbs rarely get their own subject, object or auxiliary.
        if any(c.dep_ in CLAUSAL_DEPS for c in tok.children):
            yield [tok]


@dataclass
class Counts:
    words: int = 0
    sentences: int = 0
    pairs: int = 0
    unattested: int = 0
    dep_sum: list[float] = field(default_factory=lambda: [0.0] * len(DEP_BINS))
    dep_arcs: list[int] = field(default_factory=lambda: [0] * len(DEP_BINS))
    hits: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNT_FEATURES, 0))

    def add(self, other: Counts) -> None:
        self.words += other.words
        self.sentences += other.sentences
        self.pairs += other.pairs
        self.unattested += other.unattested
        for i in range(len(DEP_BINS)):
            self.dep_sum[i] += other.dep_sum[i]
            self.dep_arcs[i] += other.dep_arcs[i]
        for k, v in other.hits.items():
            self.hits[k] += v

    def to_dict(self) -> dict:
        return {
            "words": self.words, "sentences": self.sentences, "pairs": self.pairs,
            "unattested": self.unattested, "dep_sum": self.dep_sum, "dep_arcs": self.dep_arcs,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Counts:
        c = cls()
        for k, v in d.items():
            setattr(c, k, v)
        return c


@dataclass
class Flag:
    feature: str
    sentence: str
    spans: list[tuple[int, int]]
    source: str = ""


def _dependency_arcs(words: list[Token]) -> tuple[int, int, int] | None:
    n = len(words)
    for b, (lo, hi) in enumerate(DEP_BINS):
        if lo <= n <= hi:
            break
    else:
        return None
    index = {t.i: k for k, t in enumerate(words)}
    total = arcs = 0
    for t in words:
        if t.dep_ == "ROOT" or t.head.i not in index:
            continue
        total += abs(index[t.i] - index[t.head.i])
        arcs += 1
    return b, total, arcs


def analyze_sentence(sent: Span, res: Resources, counts: Counts, flags: list[Flag] | None, source: str = "") -> None:
    words = [t for t in sent if is_word(t)]
    if not words:
        return
    counts.sentences += 1
    counts.words += sum(1 for t in words if not _is_code(t))

    prev = None
    unattested_toks = []
    for t in sent:
        cur = t.lower_ if t.is_alpha and not _is_code(t) else None
        if prev is not None and cur is not None:
            counts.pairs += 1
            if pair_hash(prev.lower_, cur) not in res.bigrams:
                counts.unattested += 1
                unattested_toks.append([prev, t])
        prev = t if cur is not None else None

    arcs = _dependency_arcs(words)
    if arcs:
        b, total, n = arcs
        counts.dep_sum[b] += total
        counts.dep_arcs[b] += n

    found: list[tuple[str, list[Token]]] = []
    found += [("inanimate_subject", toks) for toks in inanimate_subjects(sent, res)]
    found += [("stacked_nouns", toks) for toks in stacked_nouns(sent, res)]
    found += [(f"{kind}_orc", toks) for kind, toks in object_relatives(sent, res)]
    found += [("denominal_verb", toks) for toks in denominal_verbs(sent, res)]
    for feat, _ in found:
        counts.hits[feat] += 1

    if flags is not None:
        found += [("unattested_bigram", toks) for toks in unattested_toks]
        for feat, toks in found:
            spans = [(t.idx - sent.start_char, t.idx - sent.start_char + len(t.text)) for t in toks]
            flags.append(Flag(feat, sent.text, spans, source))
