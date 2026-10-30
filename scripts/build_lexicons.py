"""Build the animate-noun lexicon shipped with hardread from WordNet 3.0.

A noun lemma counts as animate when at least half of its sense-frequency mass
(SemCor counts + 1 smoothing) falls in noun.person, noun.animal or noun.group (or the person/animal/organism
top synsets).

usage: NLTK_DATA=<dir with corpora/wordnet> python scripts/build_lexicons.py <out_file>
"""
import sys
from pathlib import Path

from nltk.corpus import wordnet as wn

ANIMATE = {"noun.person", "noun.animal", "noun.group"}
ANIMATE_TOPS = {"person.n.01", "animal.n.01", "organism.n.01"}
# Nouns WordNet files under person/animal that modern technical prose uses for things.
NOT_ANIMATE = {
    "bug", "browser", "server", "compiler", "interpreter", "parser", "scanner", "router",
    "controller", "handler", "listener", "runner", "loader", "computer", "printer", "monitor",
    "receiver", "transmitter", "terminal", "host", "mouse", "virus", "worm", "python", "cookie",
}


def main(out: str) -> None:
    animate = set()
    for lemma_name in wn.all_lemma_names("n"):
        if "_" in lemma_name or not lemma_name.isalpha():
            continue
        lemmas = [l for s in wn.synsets(lemma_name, "n") for l in s.lemmas() if l.name().lower() == lemma_name]
        if not lemmas:
            continue
        total = sum(l.count() + 1 for l in lemmas)
        hit = sum(l.count() + 1 for l in lemmas if l.synset().lexname() in ANIMATE or l.synset().name() in ANIMATE_TOPS)
        if hit / total >= 0.5 and lemma_name not in NOT_ANIMATE:
            animate.add(lemma_name.lower())
    Path(out).write_text("\n".join(sorted(animate)) + "\n")
    print(f"{len(animate)} animate nouns -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
