from __future__ import annotations

import argparse
import json
import os
import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__
from .analyzer import FEATURES, Analyzer, Report, iter_texts
from .text import CODE

DEFAULT_BASELINE = "chat"


def _fmt(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _ratio_text(r: float | None) -> Text:
    if r is None:
        return Text("-", style="dim")
    style = "bold red" if r >= 1.5 else "yellow" if r >= 1.15 else "green" if r <= 0.87 else ""
    return Text(f"{r:.2f}x", style=style)


def _bar(r: float | None, width: int = 12) -> Text:
    if r is None:
        return Text("")
    n = max(0, min(width, round(r / 3 * width)))
    mid = round(width / 3)
    t = Text("█" * n + " " * (width - n), style="red" if r >= 1.15 else "green")
    if mid < width:
        t.stylize("on grey23", mid, mid + 1)
    return t


def highlight(sentence: str, spans: list[tuple[int, int]], mark: bool = False) -> Text:
    if mark:
        for s, e in sorted(set(spans), reverse=True):
            sentence = sentence[:s] + "[" + sentence[s:e] + "]" + sentence[e:]
        spans = []
    t = Text(sentence)
    for s, e in spans:
        t.stylize("bold reverse", s, e)
    plain = t.plain
    i = plain.find(CODE)
    while i >= 0:
        t = t[:i] + Text("`code`", style="cyan") + t[i + len(CODE):]
        plain = t.plain
        i = plain.find(CODE, i + 6)
    return t


def render_single(console: Console, report: Report, baseline: dict, base_name: str, examples: int) -> None:
    console.print(
        f"[bold]hardread[/] {report.name}  "
        f"[dim]{report.documents:,} doc{'s' if report.documents != 1 else ''}, {report.counts.sentences:,} sentences, {report.counts.words:,} words[/]"
    )
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("feature")
    table.add_column("unit", style="dim")
    table.add_column("this text", justify="right")
    table.add_column(f"human ({base_name})", justify="right", style="dim")
    table.add_column("ratio", justify="right")
    table.add_column("")
    ratios = report.ratios(baseline)
    for f in FEATURES:
        table.add_row(f.label, f.unit, _fmt(report.rates.get(f.key)), _fmt(baseline.get(f.key)),
                      _ratio_text(ratios.get(f.key)), _bar(ratios.get(f.key)) if f.difficulty else Text(""))
    console.print(table)
    idx = report.index(baseline)
    if idx is not None:
        console.print(Text.assemble("Difficulty index ", _ratio_text(idx),
                                    (f"  mean ratio to human {base_name} text over 6 features (each capped at 5x); 1.00x = human-like", "dim")))
    if report.counts.words < 2000:
        console.print("[dim]Fewer than 2,000 words: rates are noisy.[/]")
    if examples:
        _render_examples(console, report, examples)


def _render_examples(console: Console, report: Report, n: int) -> None:
    for f in FEATURES:
        if not f.difficulty or f.key == "unattested_bigram":
            continue
        flags = report.flags.get(f.key, [])
        if not flags:
            continue
        console.print(f"\n[bold]{f.label}[/] [dim]({report.counts.hits.get(f.key, 0)} found)[/]")
        merged: dict[tuple[str, str], list] = {}
        for flag in flags:
            merged.setdefault((flag.source, flag.sentence), []).extend(flag.spans)
        for (source, sentence), spans in list(merged.items())[:n]:
            line = Text("  ") + highlight(sentence, spans, mark=not console.is_terminal)
            if source:
                line.append(f"  {source}", style="dim")
            console.print(line)


def render_compare(console: Console, reports: list[Report], baseline: dict, base_name: str) -> None:
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("feature")
    for r in reports:
        table.add_column(r.name, justify="right")
    table.add_column(f"human ({base_name})", justify="right", style="dim")
    for f in FEATURES:
        cells = []
        for r in reports:
            ratio = r.ratios(baseline).get(f.key)
            cell = Text(_fmt(r.rates.get(f.key)))
            if ratio is not None:
                cell.append(" ")
                cell.append_text(_ratio_text(ratio))
            cells.append(cell)
        table.add_row(f.label, *cells, _fmt(baseline.get(f.key)))
    table.add_row(Text("Difficulty index", style="bold"),
                  *[_ratio_text(r.index(baseline)) for r in reports], "1.00x")
    table.add_row(Text("words", style="dim"), *[Text(f"{r.counts.words:,}", style="dim") for r in reports], "")
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hardread",
        description="Measure how hard text is to read, by the features that make AI prose hard to process, "
                    "against a human baseline.",
    )
    p.add_argument("paths", nargs="*", help="files or folders (.txt .md .jsonl .json), or - for stdin; "
                                            "several paths are compared side by side")
    p.add_argument("-b", "--baseline", default=DEFAULT_BASELINE,
                   help="human reference to compare with (default: chat; see --list-baselines)")
    p.add_argument("-e", "--examples", type=int, default=3, metavar="N",
                   help="flagged sentences to show per feature (default 3, 0 to hide)")
    p.add_argument("--field", help="JSON field holding the text (default: auto-detect)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.add_argument("--model", default="en_core_web_sm", help="spaCy model (default en_core_web_sm)")
    p.add_argument("-j", "--jobs", type=int, default=1, help="parser processes (default 1)")
    p.add_argument("--fail-above", type=float, metavar="X",
                   help="exit with status 1 if the difficulty index exceeds X (for CI and agents)")
    p.add_argument("--list-baselines", action="store_true", help="list bundled human baselines and exit")
    p.add_argument("-V", "--version", action="version", version=f"hardread {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(highlight=False, soft_wrap=False)
    analyzer = Analyzer(model=args.model, n_process=max(1, args.jobs))
    corpora = analyzer.baselines["corpora"]

    if args.list_baselines:
        for name, c in corpora.items():
            console.print(f"[bold]{name:<8}[/] {c['description']} [dim]({c['words']:,} words)[/]")
        return 0
    if not args.paths:
        if sys.stdin.isatty():
            build_parser().print_help()
            return 2
        args.paths = ["-"]
    if args.baseline not in corpora:
        console.print(f"[red]unknown baseline {args.baseline!r}[/]; choose from {', '.join(corpora)}")
        return 2
    baseline = corpora[args.baseline]["rates"]

    reports = []
    for path in args.paths:
        if path != "-" and not os.path.exists(path):
            console.print(f"[red]no such file or folder: {path}[/]")
            return 2
        name = "stdin" if path == "-" else os.path.basename(os.path.normpath(path))
        docs = list(iter_texts(path, args.field))
        if not docs:
            console.print(f"[red]no text found in {path}[/]")
            return 2
        reports.append(analyzer.analyze(docs, name=name, keep_flags=bool(args.examples)))

    if args.json:
        out = [r.to_dict(baseline, args.examples) for r in reports]
        print(json.dumps(out[0] if len(out) == 1 else out, indent=2))
    elif len(reports) == 1:
        render_single(console, reports[0], baseline, args.baseline, args.examples)
    else:
        render_compare(console, reports, baseline, args.baseline)

    if args.fail_above is not None:
        worst = max((r.index(baseline) or 0) for r in reports)
        if worst > args.fail_above:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
