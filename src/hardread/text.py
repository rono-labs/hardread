import re

CODE = "Xcode"

_FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1[^\n]*$", re.S | re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)
_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
_QUOTE = re.compile(r"^\s*>\s?", re.M)
_EMPH = re.compile(r"(?<!\w)(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1(?!\w)")


def clean(text: str) -> str:
    """Strip markdown and code so the parser sees prose.

    Fenced blocks, headings, tables and URLs are dropped. Inline code becomes a
    single placeholder noun so sentences like "`is_file()` swallows that" keep
    their structure.
    """
    text = _THINK.sub(" ", text)
    text = _FENCE.sub("\n", text)
    text = _HEADING.sub("\n\n", text)
    text = _TABLE.sub("", text)
    text = _RULE.sub("", text)
    text = _INLINE_CODE.sub(CODE, text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _HTML.sub(" ", text)
    text = _BULLET.sub("\n\n", text)
    text = _QUOTE.sub("", text)
    text = _EMPH.sub(r"\2", text)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)
