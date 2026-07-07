"""
Accuracy / evaluation system.

Scores each generated reply (from generator/outputs.json) using the LLM-
judge rubric in rubric.py, plus a secondary reference-similarity signal
purely as a diagnostic (see rubric.py docstring for why similarity is not
the primary metric).

Run:
  python3 evaluate.py
Produces:
  evaluation/scores.json - per-response dimension scores + composite,
  plus an overall aggregate across all responses.
"""

import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.retrieval import load_dataset
from generator.llm_backend import chat, LLMError, PROVIDER
from rubric import DIMENSIONS, WEIGHTS, JUDGE_SYSTEM_PROMPT, build_judge_prompt, composite_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_json(text):
    """LLMs occasionally wrap JSON in prose or code fences; pull out the
    first {...} block and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:300]!r}")
    return json.loads(match.group(0))


def reference_similarity(generated_reply, reference_replies):
    """Diagnostic-only cosine similarity to the closest retrieved
    reference reply. Not part of the composite score - see rubric.py."""
    if not reference_replies:
        return None
    corpus = reference_replies + [generated_reply]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(corpus)
    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    return round(float(max(sims)), 3)


def score_one(incoming_subject, incoming_email, reference_examples, suggested_reply):
    prompt = build_judge_prompt(incoming_subject, incoming_email, reference_examples, suggested_reply)
    raw = chat(JUDGE_SYSTEM_PROMPT, prompt, temperature=0.0)
    parsed = extract_json(raw)
    for d in DIMENSIONS:
        if d not in parsed or "score" not in parsed[d]:
            raise ValueError(f"Judge response missing dimension '{d}': {parsed}")
        parsed[d]["score"] = max(0, min(5, int(parsed[d]["score"])))
    return parsed


def main():
    dataset = load_dataset()
    by_id = {row["id"]: row for row in dataset}

    gen_path = os.path.join(os.path.dirname(__file__), "..", "generator", "outputs.json")
    with open(gen_path) as f:
        generated = json.load(f)

    if not generated:
        print("No generated replies found - run generator/generate.py first.")
        return

    results = []
    print(f"Using LLM_PROVIDER={PROVIDER} for judging\n")
    for row in generated:
        ref_examples = [by_id[i] for i in row["retrieved_example_ids"] if i in by_id]
        ref_replies = [ex["reply"] for ex in ref_examples]

        print(f"Scoring {row['id']}: {row['subject']!r} ...")
        try:
            dims = score_one(row["subject"], row["incoming_email"], ref_examples, row["generated_reply"])
        except (LLMError, ValueError) as e:
            print(f"  ! Judge call failed ({e}). Skipping.")
            continue

        comp = composite_score(dims)
        sim = reference_similarity(row["generated_reply"], ref_replies)

        results.append({
            "id": row["id"],
            "subject": row["subject"],
            "generated_reply": row["generated_reply"],
            "dimension_scores": {d: dims[d]["score"] for d in DIMENSIONS},
            "dimension_reasons": {d: dims[d]["why"] for d in DIMENSIONS},
            "composite_score_0_to_5": comp,
            "composite_score_pct": round(comp / 5 * 100, 1),
            "reference_similarity_diagnostic": sim,
        })

    if not results:
        print("\nNo scores produced - check judge/LLM backend connectivity.")
        return

    overall = {}
    for d in DIMENSIONS:
        overall[d + "_mean"] = round(sum(r["dimension_scores"][d] for r in results) / len(results), 2)
    overall["composite_mean_0_to_5"] = round(sum(r["composite_score_0_to_5"] for r in results) / len(results), 3)
    overall["composite_mean_pct"] = round(overall["composite_mean_0_to_5"] / 5 * 100, 1)
    overall["n_responses"] = len(results)
    overall["weights"] = WEIGHTS

    # Per-incoming-email-category breakdown. We look up each generated
    # reply's category via the closest-retrieved reference example's
    # category (majority vote among the top-k retrieved), since the new
    # incoming emails themselves aren't pre-labeled with a category.
    by_category = {}
    for row in generated:
        scored = next((r for r in results if r["id"] == row["id"]), None)
        if scored is None:
            continue
        ref_cats = [by_id[i]["category"] for i in row["retrieved_example_ids"] if i in by_id]
        top_cat = max(set(ref_cats), key=ref_cats.count) if ref_cats else "unknown"
        by_category.setdefault(top_cat, []).append(scored["composite_score_0_to_5"])

    overall["by_category_mean_composite"] = {
        cat: round(sum(scores) / len(scores), 3) for cat, scores in sorted(by_category.items())
    }
    overall["by_category_n"] = {cat: len(scores) for cat, scores in sorted(by_category.items())}

    # Lowest and highest scoring responses, so a reader can jump straight
    # to the interesting cases instead of scanning the full list.
    ranked = sorted(results, key=lambda r: r["composite_score_0_to_5"])
    overall["lowest_scoring_id"] = ranked[0]["id"] if ranked else None
    overall["highest_scoring_id"] = ranked[-1]["id"] if ranked else None

    output = {"per_response": results, "overall": overall}

    out_path = os.path.join(os.path.dirname(__file__), "scores.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote scores for {len(results)} responses to {out_path}")
    print(f"Overall composite score: {overall['composite_mean_0_to_5']}/5 ({overall['composite_mean_pct']}%)")
    for d in DIMENSIONS:
        print(f"  {d}: {overall[d + '_mean']}/5")
    print("By category:")
    for cat, mean in overall["by_category_mean_composite"].items():
        print(f"  {cat} (n={overall['by_category_n'][cat]}): {mean}/5")
    print(f"Lowest scoring: {overall['lowest_scoring_id']}  |  Highest scoring: {overall['highest_scoring_id']}")


if __name__ == "__main__":
    main()
