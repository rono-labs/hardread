import numpy as np
import pytest

from hardread.resources import BigramSet
from hardread.features import pair_hash


def found(analyzer, sentence, feature):
    report = analyzer.analyze(sentence)
    return [[f.sentence[s:e] for s, e in f.spans] for f in report.flags.get(feature, [])]


@pytest.mark.parametrize("sentence,words", [
    ("Sometimes chaos expects you to remember it independently.", ["chaos", "expects"]),
    ("A song doesn't ask you to turn melancholy into a calendar item.", ["song", "ask"]),
    ("`is_file()` swallows that and returns False.", ["Xcode", "swallows"]),
])
def test_inanimate_subject(analyzer, sentence, words):
    assert words in found(analyzer, sentence, "inanimate_subject")


def test_animate_subject_not_flagged(analyzer):
    assert found(analyzer, "My sister expects you to call her tonight.", "inanimate_subject") == []


def test_stacked_nouns(analyzer):
    hits = found(analyzer, "Summer park district classes are a whole thing in Chicago.", "stacked_nouns")
    assert hits == [["Summer", "park", "district", "classes"]]
    assert found(analyzer, "The coffee shop is open.", "stacked_nouns") == []


def test_reduced_object_relative(analyzer):
    hits = found(analyzer, "It is one of the best genres you could pick.", "reduced_orc")
    assert hits and hits[0][0] == "genres"


def test_overt_object_relative(analyzer):
    assert found(analyzer, "This is the house that the contractors built.", "overt_orc")
    assert not found(analyzer, "The man who saw the dog left.", "overt_orc")


def test_rates_are_per_thousand_words(analyzer):
    report = analyzer.analyze("Summer park district classes are fun.")
    assert report.counts.words == 6
    assert report.rates["stacked_nouns"] == pytest.approx(1000 / 6)


def test_bigramset_roundtrip(tmp_path):
    hashes = np.array([pair_hash("of", "the"), pair_hash("in", "a"), 7], dtype=np.uint32)
    BigramSet(hashes).save(tmp_path / "b.gz")
    loaded = BigramSet.load(tmp_path / "b.gz")
    assert len(loaded) == 3
    assert pair_hash("of", "the") in loaded
    assert pair_hash("purple", "theorem") not in loaded


def test_common_pairs_attested(analyzer):
    assert pair_hash("of", "the") in analyzer.res.bigrams
