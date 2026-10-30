from hardread.text import CODE, clean


def test_strips_code_blocks_and_markdown():
    md = "# Title\n\nSome **bold** text with a [link](http://x.y).\n\n```python\nx = 1\n```\n\n- item one\n- item two\n"
    out = clean(md)
    assert "Title" not in out
    assert "x = 1" not in out
    assert "Some bold text with a link." in out
    assert out.split("\n\n")[-2:] == ["item one", "item two"]


def test_inline_code_becomes_placeholder():
    assert clean("`is_file()` swallows that") == f"{CODE} swallows that"


def test_snake_case_survives():
    assert clean("call my_func_name now") == "call my_func_name now"


def test_think_blocks_removed():
    assert clean("<think>plan</think>Answer.") == "Answer."


def test_long_paragraphs_are_chunked():
    from hardread.analyzer import MAX_CHUNK, _chunks

    para = "This is a sentence. " * 20000
    chunks = list(_chunks(para))
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK for c in chunks)
    assert sum(len(c.split()) for c in chunks) == len(para.split())
