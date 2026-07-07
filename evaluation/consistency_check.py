"""
Checks judge self-consistency: how much does a score change if you ask the
same judge to score the same response multiple times?

Why this matters: a rubric score is only trustworthy for the "overall
system score" and per-response reporting the challenge asks for if it's
reasonably stable - if scoring the exact same reply twice gives wildly
different numbers, the composite score is mostly noise dressed up as
precision, no matter how sensible the rubric dimensions look on paper.
This is a second, distinct kind of validation from validate_metric.py:
that script checks the metric points in the right DIRECTION (discriminates
good from bad); this one checks the metric is STABLE (low variance) at a
fixed quality level.

Method: pick a handful of already-generated replies (from
generator/outputs.json) and score each one N times (default 3) at
temperature 0. Report the standard deviation of the composite score per
response, and the overall mean std dev across all sampled responses.

Run:
  python3 consistency_check.py
Produces:
  evaluation/consistency_report.json
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.retrieval import load_dataset
from generator.llm_backend import LLMError, PROVIDER
from rubric import composite_score
from evaluate import score_one

N_RUNS = 3
N_RESPONSES_TO_SAMPLE = 4


def main():
    dataset = load_dataset()
    by_id = {row["id"]: row for row in dataset}

    gen_path = os.path.join(os.path.dirname(__file__), "..", "generator", "outputs.json")
    with open(gen_path) as f:
        generated = json.load(f)

    if not generated:
        print("No generated replies found - run generator/generate.py first.")
        return

    sample = generated[:N_RESPONSES_TO_SAMPLE]
    per_response = []
    print(f"Using LLM_PROVIDER={PROVIDER}; scoring {len(sample)} responses x{N_RUNS} runs each\n")

    for row in sample:
        ref_examples = [by_id[i] for i in row["retrieved_example_ids"] if i in by_id]
        scores = []
        for run in range(N_RUNS):
            try:
                dims = score_one(row["subject"], row["incoming_email"], ref_examples, row["generated_reply"])
            except (LLMError, ValueError) as e:
                print(f"  ! Judge call failed on {row['id']} run {run+1} ({e}). Skipping run.")
                continue
            scores.append(composite_score(dims))
        if len(scores) < 2:
            print(f"{row['id']}: not enough successful runs to measure variance, skipping.")
            continue
        stdev = round(statistics.pstdev(scores), 3)
        per_response.append({"id": row["id"], "scores": scores, "mean": round(statistics.mean(scores), 3), "stdev": stdev})
        print(f"{row['id']}: scores={scores} stdev={stdev}")

    if not per_response:
        print("\nNo consistency data collected - check judge/LLM backend connectivity.")
        return

    overall_mean_stdev = round(statistics.mean(r["stdev"] for r in per_response), 3)
    report = {
        "n_runs_per_response": N_RUNS,
        "per_response": per_response,
        "overall_mean_stdev": overall_mean_stdev,
        "interpretation": (
            "Lower is more stable. A mean stdev near 0 on the 0-5 composite "
            "scale means repeated scoring of the same reply lands on ~the "
            "same number; a mean stdev above ~0.5 would mean the composite "
            "score isn't precise enough to trust small differences between "
            "responses."
        ),
    }

    out_path = os.path.join(os.path.dirname(__file__), "consistency_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOverall mean stdev across sampled responses: {overall_mean_stdev}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
