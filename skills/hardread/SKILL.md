---
name: hardread
description: Check prose you wrote (replies, docs, commit messages, PR descriptions) for constructions that make AI text hard to read, and rewrite the flagged sentences. Use before handing long prose to a human.
---

# hardread: check your prose before a human reads it

Run hardread on any prose you are about to hand to a person:

```bash
hardread draft.md                 # table + flagged sentences
hardread draft.md --json -e 20    # machine-readable, 20 examples per feature
hardread draft.md --fail-above 1.5  # exit 1 if the difficulty index is over 1.5x human
```

Pick the baseline that matches the genre: `-b chat` (default), `-b qa` for technical answers, `-b wiki` for documentation.

## How to fix what it flags

- **Inanimate subject + animate verb** ("the docstring promises", "this test bites"): name the real actor or use a plain verb. "The docstring says", "this test fails when".
- **Stacked nouns** ("dotfile backup path handling"): unpack with prepositions. "how backup paths for dotfiles are handled".
- **Object relative without "that"** ("the papercut people feel"): add "that", or restructure. "the papercut that people feel".
- **Long dependencies**: keep the subject next to its verb; move long modifiers after the verb or into a second sentence.
- **Denominal verbs** ("pooling judgments", "cushions spikes"): prefer the common verb if one exists ("combining", "softens").
- **Unattested word pairs**: rare collocations ("advice glosses"). Use the everyday phrasing.

Aim for a difficulty index near 1.0x. Do not chase zero: humans use all of these constructions. Rewrite only the sentences that read as compressed.
