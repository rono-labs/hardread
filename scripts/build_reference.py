"""Build hardread's bundled reference data from human-written corpora.

Step 1 parses the reference splits and writes:
  bigrams.u32.gz        hashes of word pairs seen at least --min-count times
  denominal_verbs.txt   verb lemmas used mostly as nouns in the reference
Step 2 parses the held-out baseline splits and writes baselines.json with the
dependency-length weights (from the weights corpus) and each corpus's rates.

Input files are JSONL with a "text" field. Example:
  python scripts/build_reference.py --ref corpora/*.ref.jsonl \
      --base chat=corpora/chat_human.base.jsonl wiki=corpora/wiki.base.jsonl \
      --weights corpora/wiki.base.jsonl --out src/hardread/data -j 16
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from hardread.analyzer import Analyzer, load_model
from hardread.features import DEP_BINS, is_word, pair_hash
from hardread.resources import BigramSet, Resources
from hardread.text import CODE, clean


def read_texts(path: str, limit_words: int | None = None):
    words = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"] if isinstance(rec, dict) else rec
            yield text
            words += len(text.split())
            if limit_words and words >= limit_words:
                return


def paragraphs(paths: list[str]):
    for p in paths:
        for text in read_texts(p):
            yield from (para for para in clean(text).split("\n\n") if para)


def scan_reference(paths: list[str], model: str, jobs: int):
    nlp = load_model(model)
    pairs: Counter[int] = Counter()
    pos: dict[str, Counter] = {"NOUN": Counter(), "VERB": Counter()}
    n_words = 0
    for doc in nlp.pipe(paragraphs(paths), batch_size=256, n_process=jobs, disable=["ner"]):
        for sent in doc.sents:
            prev = None
            for t in sent:
                cur = t.lower_ if t.is_alpha and t.text != CODE else None
                if prev is not None and cur is not None:
                    pairs[pair_hash(prev, cur)] += 1
                prev = cur
                if t.pos_ in pos:
                    pos[t.pos_][t.lemma_.lower()] += 1
                n_words += is_word(t)
    return pairs, pos, n_words


# Noun-dominant lemmas whose verb uses are mostly tagger errors on -ing nouns.
NOT_DENOMINAL = {"color", "colour", "signal", "film", "interest"}


def denominal_lemmas(pos: dict[str, Counter], n_words: int, min_total: int = 30, noun_share: float = 0.8,
                     min_verb: int = 3, max_verb_per_million: float = 5.0) -> list[str]:
    """Lemmas used mostly as nouns whose verb use is rare: ad hoc conversions, not "point" or "face"."""
    max_verb = max_verb_per_million * n_words / 1e6
    out = []
    for lemma, v in pos["VERB"].items():
        n = pos["NOUN"].get(lemma, 0)
        if (lemma.isalpha() and len(lemma) > 2 and lemma not in NOT_DENOMINAL and min_verb <= v <= max_verb
                and n + v >= min_total and n / (n + v) >= noun_share):
            out.append(lemma)
    return sorted(out)


def dep_weights(path: str, model: str, jobs: int) -> list[float]:
    nlp = load_model(model)
    arcs = [0] * len(DEP_BINS)
    for doc in nlp.pipe(paragraphs([path]), batch_size=256, n_process=jobs, disable=["ner"]):
        for sent in doc.sents:
            words = [t for t in sent if is_word(t)]
            for b, (lo, hi) in enumerate(DEP_BINS):
                if lo <= len(words) <= hi:
                    arcs[b] += len(words) - 1
    total = sum(arcs)
    return [a / total for a in arcs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", default=[], help="name=path[:description]")
    ap.add_argument("--weights", required=True, help="corpus whose sentence lengths set the dependency weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--skip-ref", action="store_true", help="reuse bigrams and lexicon already in --out")
    args = ap.parse_args()
    out = Path(args.out)

    meta_path = out / "reference.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if not args.skip_ref:
        pairs, pos, n_words = scan_reference(args.ref, args.model, args.jobs)
        kept = np.array([h for h, c in pairs.items() if c >= args.min_count], dtype=np.uint32)
        BigramSet(kept).save(out / "bigrams.u32.gz")
        den = denominal_lemmas(pos, n_words)
        (out / "denominal_verbs.txt").write_text(
            "# Lemmas tagged as nouns at least 80% of the time in the reference corpora, with\n"
            "# fewer than 5 verb uses per million words.\n" + "\n".join(den) + "\n")
        meta = {"reference_words": n_words, "unique_pairs": len(pairs), "kept_pairs": int(len(kept)),
                "min_count": args.min_count, "denominal_lemmas": len(den),
                "reference_files": [Path(p).name for p in args.ref], "model": args.model}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))

    baselines = {"dep_weights": dep_weights(args.weights, args.model, args.jobs), "corpora": {}}
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")
    analyzer = Analyzer(model=args.model, resources=Resources(out), n_process=args.jobs, batch_size=256)
    for spec in args.base:
        name, rest = spec.split("=", 1)
        path, _, desc = rest.partition(":")
        report = analyzer.analyze(read_texts(path), name=name, keep_flags=False)
        baselines["corpora"][name] = {
            "description": desc or name, "file": Path(path).name, "words": report.counts.words,
            "rates": report.rates, "counts": report.counts.to_dict(),
        }
        print(name, report.counts.words, {k: round(v, 3) for k, v in report.rates.items() if v is not None})
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")


if __name__ == "__main__":
    main()
