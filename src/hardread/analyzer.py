from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .features import Counts, Flag, analyze_sentence
from .resources import Resources
from .text import clean


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    unit: str
    difficulty: bool = True


FEATURES = (
    Feature("inanimate_subject", "Inanimate subject + animate verb", "per 1k words"),
    Feature("unattested_bigram", "Unattested word pairs", "per 1k pairs"),
    Feature("dependency_length", "Dependency length", "words per arc"),
    Feature("stacked_nouns", "Stacked nouns (3+)", "per 1k words"),
    Feature("reduced_orc", "Object relatives, no 'that'", "per 1k words"),
    Feature("denominal_verb", "Denominal verbs", "per 1k words"),
    Feature("overt_orc", "Object relatives with 'that'", "per 1k words", difficulty=False),
)
FEATURE_BY_KEY = {f.key: f for f in FEATURES}

# Cap per-feature ratios in the index so one rare feature cannot dominate a short text.
INDEX_CAP = 5.0

TEXT_FIELDS = ("text", "response", "output", "completion", "answer", "content")


def compute_rates(c: Counts, dep_weights: list[float]) -> dict[str, float | None]:
    per_k = lambda n, d: 1000 * n / d if d else None
    rates = {k: per_k(v, c.words) for k, v in c.hits.items()}
    rates["unattested_bigram"] = per_k(c.unattested, c.pairs)
    used = [(w, s / a) for w, s, a in zip(dep_weights, c.dep_sum, c.dep_arcs) if a]
    total_w = sum(w for w, _ in used)
    rates["dependency_length"] = sum(w * m for w, m in used) / total_w if total_w else None
    return rates


@dataclass
class Report:
    name: str
    counts: Counts
    rates: dict[str, float | None]
    flags: dict[str, list[Flag]] = field(default_factory=dict)
    documents: int = 0

    def ratios(self, baseline: dict[str, float]) -> dict[str, float | None]:
        out = {}
        for k, v in self.rates.items():
            b = baseline.get(k)
            out[k] = v / b if v is not None and b else None
        return out

    def index(self, baseline: dict[str, float]) -> float | None:
        """Mean ratio to the human baseline over the difficulty features (1.0 = human-like)."""
        r = self.ratios(baseline)
        vals = [min(r[f.key], INDEX_CAP) for f in FEATURES if f.difficulty and r.get(f.key) is not None]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self, baseline: dict[str, float] | None = None, examples: int = 0) -> dict:
        d = {"name": self.name, "documents": self.documents, "words": self.counts.words,
             "sentences": self.counts.sentences, "rates": self.rates}
        if baseline:
            d["baseline"] = {k: baseline.get(k) for k in self.rates}
            d["ratios"] = self.ratios(baseline)
            d["index"] = self.index(baseline)
        if examples:
            d["examples"] = {
                k: [{"sentence": f.sentence, "spans": f.spans, "source": f.source} for f in v[:examples]]
                for k, v in self.flags.items()
            }
        return d


class Analyzer:
    def __init__(self, model: str = "en_core_web_sm", resources: Resources | None = None,
                 n_process: int = 1, batch_size: int = 64, max_flags: int = 200):
        self.model = model
        self.res = resources or Resources()
        self.n_process = n_process
        self.batch_size = batch_size
        self.max_flags = max_flags
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = load_model(self.model)
        return self._nlp

    @property
    def baselines(self) -> dict:
        return self.res.baselines

    def baseline(self, name: str) -> dict[str, float]:
        corpora = self.baselines["corpora"]
        if name not in corpora:
            raise KeyError(f"unknown baseline {name!r}; choose from {', '.join(sorted(corpora))}")
        return corpora[name]["rates"]

    def analyze(self, docs: str | Iterable[str] | Iterable[tuple[str, str]], name: str = "text",
                keep_flags: bool = True) -> Report:
        """Analyze one text, or an iterable of texts or (source, text) pairs, as one pooled sample."""
        if isinstance(docs, str):
            docs = [docs]
        counts = Counts()
        flags: dict[str, list[Flag]] = defaultdict(list)
        n_docs = 0

        def paragraphs() -> Iterator[tuple[str, str]]:
            nonlocal n_docs
            for i, d in enumerate(docs):
                source, text = d if isinstance(d, tuple) else (f"{name}#{i}", d)
                n_docs += 1
                for para in clean(text).split("\n\n"):
                    yield para, source

        for doc, source in self.nlp.pipe(paragraphs(), as_tuples=True, batch_size=self.batch_size,
                                         n_process=self.n_process):
            sent_flags: list[Flag] | None = [] if keep_flags else None
            for sent in doc.sents:
                analyze_sentence(sent, self.res, counts, sent_flags, source)
            for f in sent_flags or ():
                if len(flags[f.feature]) < self.max_flags:
                    flags[f.feature].append(f)
        rates = compute_rates(counts, self.baselines["dep_weights"])
        return Report(name, counts, rates, dict(flags), n_docs)


def load_model(model: str):
    import spacy

    try:
        return spacy.load(model)
    except OSError:
        print(f"hardread: downloading spaCy model {model} (one time)", file=sys.stderr)
        from spacy.cli import download

        download(model)
        return spacy.load(model)


def iter_texts(path: str | Path, field: str | None = None) -> Iterator[tuple[str, str]]:
    """Yield (source, text) from a file, a directory of files, or '-' for stdin.

    .txt and .md files are one document each. .jsonl and .json files hold one
    document per record: a string, an object with a text field, or a chat
    transcript under "messages" whose assistant turns are used.
    """
    if str(path) == "-":
        yield "stdin", sys.stdin.read()
        return
    p = Path(path)
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix.lower() in (".txt", ".md", ".jsonl", ".json"):
                yield from iter_texts(child, field)
        return
    if p.suffix.lower() == ".jsonl":
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield from _record_texts(json.loads(line), f"{p.name}:{i + 1}", field)
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        for i, rec in enumerate(data if isinstance(data, list) else [data]):
            yield from _record_texts(rec, f"{p.name}[{i}]", field)
    else:
        yield p.name, p.read_text(encoding="utf-8", errors="replace")


def _record_texts(rec, source: str, field: str | None) -> Iterator[tuple[str, str]]:
    if isinstance(rec, str):
        yield source, rec
        return
    if not isinstance(rec, dict):
        return
    if field:
        if isinstance(rec.get(field), str):
            yield source, rec[field]
        return
    for key in ("messages", "conversation", "conversations"):
        turns = rec.get(key)
        if isinstance(turns, list) and turns and isinstance(turns[0], dict):
            for t in turns:
                role = t.get("role") or t.get("from")
                if role in ("assistant", "gpt", "model") and isinstance(t.get("content", t.get("value")), str):
                    yield source, t.get("content", t.get("value"))
            return
    for key in TEXT_FIELDS:
        if isinstance(rec.get(key), str):
            yield source, rec[key]
            return
