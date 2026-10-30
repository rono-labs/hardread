"""Score a set of JSONL corpora and write a results table.

usage: python scripts/benchmark.py corpora/*.jsonl --baseline chat --out docs/benchmark -j 16
       python scripts/benchmark.py --from-json docs/benchmark.json --out docs/benchmark
Writes <out>.json (all rates, ratios and a few flagged sentences per feature) and
<out>.md (a markdown table sorted by difficulty index).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hardread.analyzer import FEATURES, INDEX_CAP, Analyzer, iter_texts


def write_markdown(rows: list[dict], out: str) -> None:
    for d in rows:
        vals = [min(d["ratios"][f.key], INDEX_CAP) for f in FEATURES if f.difficulty and d["ratios"][f.key] is not None]
        d["index"] = sum(vals) / len(vals)
    rows.sort(key=lambda d: d["index"])
    head = "| source | words | index | " + " | ".join(f.label for f in FEATURES) + " |"
    lines = [head, "|" + "---|" * (len(FEATURES) + 3)]
    for d in rows:
        cells = [f"{d['rates'][f.key]:.2f} ({d['ratios'][f.key]:.2f}x)" if d["ratios"][f.key] is not None
                 else "-" for f in FEATURES]
        lines.append(f"| {d['name']} | {d['words']:,} | {d['index']:.2f}x | " + " | ".join(cells) + " |")
    Path(out + ".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--baseline", default="chat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-json", help="re-render the table from an earlier run")
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text())
        write_markdown(data["results"], args.out)
        return

    analyzer = Analyzer(model=args.model, n_process=args.jobs, batch_size=128)
    base = analyzer.baseline(args.baseline)
    rows = []
    for f in args.files:
        name = Path(f).name.removesuffix(".jsonl").removesuffix(".base")
        report = analyzer.analyze(iter_texts(f), name=name)
        d = report.to_dict(base, args.examples)
        rows.append(d)
        print(f"{name:40s} words={d['words']:>8,} index={d['index']:.2f}", flush=True)

    write_markdown(rows, args.out)
    Path(args.out + ".json").write_text(json.dumps({"baseline": args.baseline, "results": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
