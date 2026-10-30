# How hardread measures difficulty

hardread follows the feature set in Clark, Martin and Mahowald (2026), "Why is AI-generated language so hard to understand?" ([arXiv:2610.29162](https://arxiv.org/abs/2610.29162)). Every feature comes from a spaCy dependency parse (`en_core_web_sm` by default). The paper used the transformer model and a proprietary classifier for two features. hardread swaps in lexicons and parse rules so it runs on a laptop CPU. The numbers are close in spirit to the paper's, not identical.

## Preprocessing

Markdown is stripped before parsing. Fenced code blocks, tables, headings, URLs and `<think>` blocks are removed. Inline code becomes a placeholder noun, so a sentence like "`is_file()` swallows that" keeps its subject. Each paragraph or list item is parsed separately.

## Features

| feature | rule | unit |
|---|---|---|
| Inanimate subject + animate verb | An active verb from a list of 124 verbs that normally need a sentient subject (expect, promise, forget, swallow, bite, ...). Its subject is an inanimate noun, inline code, or it/this/that/which. Animacy comes from a WordNet-derived list of 16.7k person, animal and group nouns, plus spaCy PERSON/NORP/ORG/GPE entities. | per 1k words |
| Unattested word pairs | Adjacent lowercase word pairs in a sentence that never occur in a 10.6M-word human reference (Wikipedia, OpenAssistant human replies, StackExchange, Gutenberg fiction, human essays). The reference ships as 2M 32-bit hashes in 4 MB. | per 1k pairs |
| Dependency length | Mean distance in words between each word and its syntactic head. Punctuation and the root are excluded. The mean is standardized across five sentence-length bins (1-5, ..., 21-30 words; longer sentences are dropped) using Wikipedia's length distribution as weights, as in the paper's appendix. | words per arc |
| Stacked nouns | Three or more nouns chained by `compound` arcs ("dotfile backup path handling"). | per 1k words |
| Object relatives, no "that" | A relative clause that has its own subject, no object of its own, is not passive, and has no relativizer ("the job you hate"). Adverbial heads (way, reason, time, ...) are excluded. | per 1k words |
| Denominal verbs | A verb whose lemma is tagged as a noun at least 80% of the time in the reference and is used as a verb fewer than 5 times per million words (ad hoc conversions such as "cushions", "pooling"). | per 1k words |
| Object relatives with "that" | The same clause with an overt that/which/who. Shown for context; the paper finds models *under*use it. | per 1k words |

## Baselines

Each bundled baseline is the same measurement run on a held-out sample of human text that is not part of the bigram reference:

| name | source | words |
|---|---|---|
| `chat` (default) | Human-written assistant replies from OpenAssistant oasst2 | 289k |
| `qa` | StackExchange answers | 305k |
| `wiki` | English Wikipedia paragraphs | 305k |
| `essay` | Human essays from the Ghostbuster corpus | 98k |
| `fiction` | Project Gutenberg fiction passages | 205k |

## Difficulty index

The index is the mean of the six difficulty-feature ratios to the chosen baseline. Each ratio is capped at 5x so one rare feature can't dominate a short text. 1.00x means human-like. Overt object relatives are left out because using fewer of them is not harder to read.

## Accuracy and limits

- A hand-judged sample of 40 flags per feature on model and human chat text gave these precisions for the first rule set: inanimate subjects 50%, stacked nouns 65%, reduced object relatives 85%, overt object relatives 68%, denominal verbs 70%. The current rules fix the most common false-positive patterns found in that audit. Treat single flags as prompts to reread, not verdicts.
- Rates under about 2,000 words are noisy. The CLI warns when that happens.
- Comparisons across sources with different prompts mix up style with topic. For a clean comparison, run the same prompts through each model.
- English only.
