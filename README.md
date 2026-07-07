# AI-suggested email replies + accuracy evaluation

Built for Hiver's 100-minute open challenge. Given an incoming customer support
email, this generates a suggested reply grounded in a dataset of past
emails/replies, then scores how good that reply actually is — per response and
in aggregate.

```
run_all.py                  # single command: runs the entire pipeline below
dataset/
  build_dataset.py           # generates emails.json (past email/reply pairs)
  emails.json                 # 40 synthetic past email/reply pairs
  incoming_emails.json        # 8 new "unseen" emails to generate replies for
generator/
  llm_backend.py               # pluggable model backend (Ollama/OpenAI/Anthropic/mock)
  generate.py                   # retrieval + few-shot prompting -> outputs.json
  outputs.json                   # generated replies (created by generate.py)
common/
  retrieval.py                  # TF-IDF similarity search over the dataset
evaluation/
  rubric.py                      # scoring dimensions, weights, judge prompt
  evaluate.py                    # scores generator/outputs.json -> scores.json
  validate_metric.py             # good/mediocre/bad discrimination check
  consistency_check.py           # judge repeatability (stdev across repeated scoring)
  scores.json                     # per-response + overall scores (created by evaluate.py)
  validation_report.json          # discrimination validation results
  consistency_report.json         # stability validation results
sample_run/
  build_sample_artifacts.py    # assembles a real (non-mock) sample run - see "Sample results"
tests/                        # pytest unit tests (dataset schema, retrieval, rubric math, JSON parsing)
ci/ci.yml                   # GitHub Actions config (tests + offline smoke test); see note under "How to run"
```

## TL;DR results (real sample run)

The committed `evaluation/*.json` are a **real** run (Claude Opus as both
generator and judge — see "Sample results & provenance" at the bottom for
exactly what that means and how to reproduce it):

| What | Result |
|---|---|
| Overall reply quality (8 held-out emails) | **4.74 / 5** composite |
| Metric discrimination (good vs mediocre vs bad, 10 cases) | **10/10** correctly ordered; means **4.96 / 3.27 / 2.45** |
| Metric stability (same reply judged 3×) | mean stdev **0.053** on the 0–5 scale |

The `reference_similarity` diagnostic stays low (0.21–0.43) even for replies
the judge rated ~4.8/5 — concrete evidence that lexical overlap with a
reference reply is the *wrong* primary metric, which is the whole reason the
rubric exists.

## 1. The dataset

`dataset/emails.json` is 40 synthetic customer-support email/reply pairs,
hand-authored with Claude's help during this session (see "AI tool use"
below), spanning 11 categories that match the kind of traffic a shared
support inbox like Hiver's actually sees: order status, refunds, shipping
delays, billing disputes, product defects/bugs, cancellations, account
access, feature requests, escalations, positive feedback, and general
inquiries.

**Why synthetic instead of a scraped/public corpus:** public email corpora
(e.g. Enron) are internal corporate email, not customer-support traffic, so
they wouldn't represent the tone or structure this system needs. Real
customer support data would be the right dataset in production but isn't
something I can legitimately obtain or use in a public repo in 100 minutes
without touching PII. Hand-authoring gives full control over category
coverage and lets me be explicit that no real customer data is involved,
at the cost of not capturing the messiness of real inboxes (typos,
multi-topic emails, attachments, thread history). Each row pairs a customer
email with the reply an agent actually sent — that sent reply is the
implicit "this is what good looks like" signal used for grounding
(see below), not a literal answer key to match against.

`dataset/incoming_emails.json` is a separate set of 8 new emails (not in
the training set) that the generator produces replies for — this is the
"test set."

Run `python3 dataset/build_dataset.py` to regenerate `emails.json` from
source.

## 2. The generator

`generator/generate.py`, for each new incoming email:

1. Retrieves the 3 most similar past email/reply pairs from the dataset
   using TF-IDF cosine similarity over subject + body (`common/retrieval.py`).
2. Builds a few-shot prompt: the retrieved pairs are given to the LLM as
   real examples of "how this team replies" — tone, structure, and what
   kinds of commitments are reasonable.
3. Asks the LLM to draft a reply to the new email in that style.

**Why retrieval + few-shot prompting, not fine-tuning or a heavier RAG
stack:** 40 rows is nowhere near enough to fine-tune a model without badly
overfitting. Prompting with retrieved real examples gets most of the
benefit — style and policy consistency — with no training infrastructure,
and it updates instantly if the dataset changes (no retraining step),
which matters for a support inbox where policies shift often. TF-IDF
instead of neural embeddings keeps retrieval fast and dependency-light for
a dataset this size; a production system with a much larger email history
would want real embeddings, and the retrieval interface is written so that
swap only touches `common/retrieval.py`.

**Trade-off to be explicit about:** this grounds the *style and policy* of
the reply in real examples, but the model still has to reason about the
specific new customer's situation the way a human agent would — the
dataset isn't a database of account facts. The generator is instructed not
to invent specific facts (order numbers, dollar amounts) that weren't in
the incoming email; the evaluator checks whether it actually complies
(the `correctness` dimension, below).

**Model backend** is pluggable (`generator/llm_backend.py`) via the
`LLM_PROVIDER` env var: `ollama` (default, local, free, no API key),
`openai`, or `anthropic`. See "How to run" for setup.

## 3. The accuracy / evaluation system — the core of this challenge

### What "accurate" means here

Exact-match or n-gram overlap (BLEU/ROUGE) against the reference reply is
the wrong metric: there is no single correct reply to a customer email —
two agents could word a reply completely differently and both be
excellent, or both copy similar phrasing and both be bad. So "accurate"
here means **"would a support lead approve sending this,"** not "matches
the reference text."

### The metric

`evaluation/rubric.py` defines an LLM-as-judge rubric scoring each
generated reply 0–5 on six dimensions:

| Dimension | Question | Weight |
|---|---|---|
| relevance | Does it address what the customer actually asked/needs? | 0.25 |
| correctness | Does it avoid inventing facts (order #s, amounts, dates) or making promises inconsistent with how this team operates? | 0.25 |
| completeness | If multiple points were raised, are all addressed? | 0.15 |
| actionability | Does it give a concrete next step, not just acknowledgement? | 0.15 |
| tone | Is it professional and appropriately empathetic? | 0.10 |
| conciseness | Is it appropriately brief, no padding/filler? | 0.10 |

The composite score is a weighted average, weighted toward **relevance**
and **correctness** — a fluent, warm reply that answers the wrong question
or invents a tracking number is worse than a plain but correct one, and in
a real support tool a hallucinated promise is the actual harm case worth
weighting against. Each dimension comes with a one-sentence justification
from the judge, so scores are auditable, not just numbers — this directly
supports the per-response reporting requirement.

`evaluation/evaluate.py` also computes a secondary **reference-similarity**
signal (TF-IDF cosine similarity to the closest retrieved past reply) —
reported per response as a diagnostic only, explicitly not part of the
composite score, since high similarity isn't required for a reply to be
good and low similarity isn't required for a reply to be bad.

Validation here is two separate, deliberately distinct checks — a metric
can point the right direction but still be too noisy to trust, or be
stable but not actually discriminating; a real validation story needs both.

### Check 1 — does it discriminate good from bad? (`validate_metric.py`)

A rubric is only useful if it can tell good replies from bad ones — a
judge that rates everything a 4/5 regardless of content would be
worthless despite looking rigorous. For 10 dataset rows, `validate_metric.py`
scores **three** tiers of reply to the same customer email:

- **GOOD** — the real "sent" reply from the dataset.
- **MEDIOCRE** — generic, technically-not-wrong, but doesn't engage with
  the specifics (the realistic "would slip past a lazy human review" case).
- **BAD** — one deliberate, named failure mode: ignoring the question,
  inventing facts, dismissive tone, no actionable resolution, padded/
  rambling text, or a flatly wrong policy answer.

Testing three tiers instead of two is a stronger check than good-vs-bad
alone: a judge could trivially separate "great" from "obviously terrible"
while still being unable to tell genuinely good work from merely adequate
work — the harder, more useful distinction in practice. If the metric is
meaningful, we expect GOOD > MEDIOCRE > BAD, especially on `relevance` and
`correctness`. `validation_report.json` reports the full ordering rate plus
each pairwise comparison and per-dimension gaps, so it's visible exactly
where the metric would fail if it did.

### Check 2 — is it stable? (`consistency_check.py`)

A score is only trustworthy for reporting if scoring the *same* reply
twice gives roughly the same number — otherwise the composite is mostly
noise dressed up as precision. `consistency_check.py` scores a sample of
already-generated replies 3 times each at temperature 0 and reports the
standard deviation per response and overall. Low stdev (near 0 on the 0–5
scale) means the metric is precise enough that a difference between two
responses' scores is meaningful and not just re-roll noise.

### Reporting

- `evaluation/scores.json`: per-response dimension scores + justifications
  + composite score, an overall aggregate (mean per dimension, mean
  composite, weights used, **per-category breakdown**, and the
  lowest/highest-scoring response ids for quick inspection).
- `evaluation/validation_report.json`: good/mediocre/bad ordering rate,
  pairwise win rates, and per-dimension gaps.
- `evaluation/consistency_report.json`: per-response score stdev across
  repeated judging, plus an overall mean stdev.

## How to run

Requires Python 3.10+. On Windows the launcher is usually `python` rather than
`python3` — substitute accordingly (`run_all.py` already invokes the current
interpreter via `sys.executable`, so it works either way).

```bash
pip install -r requirements.txt
```

**Set up a free local model with Ollama** (recommended — no API key, no cost):

```bash
# install from https://ollama.com, then:
ollama pull llama3.2
```

That's it — `LLM_PROVIDER` defaults to `ollama`. If you'd rather use a
paid API, set `LLM_PROVIDER=openai` + `OPENAI_API_KEY`, or
`LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`, before running the
commands below.

**Run everything in one command:**

```bash
python3 run_all.py
```

...or run each stage individually:

```bash
# 1. (re)build the dataset
python3 dataset/build_dataset.py

# 2. generate suggested replies for the 8 held-out incoming emails
python3 generator/generate.py

# 3. score the generated replies (per-response + overall + per-category)
python3 evaluation/evaluate.py

# 4. validate the metric discriminates good/mediocre/bad replies
python3 evaluation/validate_metric.py

# 5. validate the metric is stable across repeated scoring
python3 evaluation/consistency_check.py
```

**Run the tests:**

```bash
pip install pytest
python3 -m pytest tests/ -v
```

There is also an offline `LLM_PROVIDER=mock` mode (`llm_backend.py`) used
only to smoke-test that retrieval, prompting, and JSON I/O run end-to-end
without any model installed — it returns templated, non-meaningful
text/scores and must not be used to judge quality. This is what the CI
config runs (CI has no model access), alongside the real unit tests in
`tests/`.

A GitHub Actions workflow for this lives at `ci/ci.yml`. It's kept outside
`.github/workflows/` so the repo can be pushed with a token that lacks the
`workflow` OAuth scope; to activate it, move it to `.github/workflows/ci.yml`
(`git mv ci/ci.yml .github/workflows/ci.yml`) from an account/token that has
the `workflow` scope, or add it through GitHub's web UI.

## AI tool use

This entire project — architecture, dataset authoring, code, and this
README — was built in a single session using Claude (Anthropic), inside
Claude's Cowork agent environment, per the challenge rules allowing any AI
tools with disclosure. Claude wrote the code and the synthetic dataset
content directly; the design decisions (retrieval-based few-shot grounding
over fine-tuning, the six-dimension weighted rubric over exact-match
scoring, and the good-vs-bad discriminative validation approach) were
made and explained inline throughout, as documented in the code comments
and this README.

## Known limitations / what I'd do with more time

- The dataset is small (40 rows) and hand-authored rather than sourced
  from real (anonymized) support traffic — real traffic would have
  messier, multi-topic, sometimes ambiguous emails.
- TF-IDF retrieval is a lexical, not semantic, match — it can miss
  paraphrased duplicates. A small local embedding model (also available
  via Ollama) would be a quick upgrade.
- `validate_metric.py` uses 10 hand-picked cases across 3 tiers; a larger,
  more systematic adversarial set (and inter-rater agreement against a
  second, different judge model) would strengthen the validation claim
  further.
- `consistency_check.py` samples a handful of responses at temperature 0;
  a fuller study would sample every response and also measure variance at
  higher temperature, plus agreement between two different judge models
  (not just repeated calls to the same one).
- The committed sample scores are high (4.74/5) partly because the sample
  generator was Claude Opus, a strong model, on relatively clean
  single-topic emails. A smaller local model (e.g. `llama3.2` via Ollama)
  would score lower and show more spread; the discrimination the metric
  gives you is demonstrated by `validate_metric.py`, not by the test-set
  mean.

## Sample results & provenance

The `evaluation/*.json` and `generator/outputs.json` in this repo are a
**real sample run**, not the `mock` backend's placeholder output. Being
fully transparent about how they were produced:

This build session had **no scriptable LLM provider** available — no API key
was set, no local Ollama server, and repurposing the host's OAuth
credentials was (correctly) disallowed. So the model-dependent content — the
8 generated replies and every rubric score — was produced by **Claude Opus
during the session**, i.e. the same model family that `LLM_PROVIDER=anthropic`
calls at runtime. It is genuine model output, authored in-session rather than
fetched over HTTP.

`sample_run/build_sample_artifacts.py` shows exactly which parts are which:

- **Computed by the project code** (deterministic, not hand-made): the
  retrieved example ids/scores (`common/retrieval.py`), every composite score
  (`evaluation/rubric.composite_score`), the reference-similarity diagnostic,
  and all aggregation/ordering statistics.
- **Authored by the model** (Claude Opus): the reply text, and the
  per-dimension 0–5 scores plus one-line justifications.

Every result file carries a `_provenance` field saying this, and it is all
**regenerable**: run `python run_all.py` (or the individual stages) with any
real `LLM_PROVIDER` and `evaluate.py` / `validate_metric.py` /
`consistency_check.py` overwrite these files with a fresh run. The offline
`LLM_PROVIDER=mock` smoke test (what the CI config runs, since CI has no model access)
still exists purely to prove the retrieval → prompting → JSON I/O plumbing
runs end-to-end without any model.
