import json

from hardread.analyzer import iter_texts
from hardread.cli import main

TEXT = ("The docstring promises they're skipped. Sometimes chaos expects you to remember it. "
        "Summer park district classes are a whole thing in Chicago. That's one of the best genres you could pick.")


def test_json_output(tmp_path, capsys):
    f = tmp_path / "reply.txt"
    f.write_text(TEXT)
    assert main([str(f), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["words"] > 20
    assert set(out["ratios"]) >= {"inanimate_subject", "stacked_nouns", "dependency_length"}
    assert out["index"] > 0


def test_fail_above(tmp_path, capsys):
    f = tmp_path / "reply.txt"
    f.write_text(TEXT)
    assert main([str(f), "--fail-above", "0.01", "--json"]) == 1
    assert main([str(f), "--fail-above", "1000", "--json"]) == 0


def test_compare_two_sources(tmp_path, capsys):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text(TEXT)
    b.write_text("I went to the store and bought some milk. Then I walked home with my dog.")
    assert main([str(a), str(b), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in out] == ["a.md", "b.md"]


def test_list_baselines(capsys):
    assert main(["--list-baselines"]) == 0
    assert "chat" in capsys.readouterr().out


def test_iter_texts_reads_chat_jsonl(tmp_path):
    f = tmp_path / "chats.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there."}]},
        {"text": "Plain record."},
        {"response": "Another one."},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert [t for _, t in iter_texts(f)] == ["Hello there.", "Plain record.", "Another one."]


def test_missing_path(capsys):
    assert main(["does-not-exist.txt"]) == 2
