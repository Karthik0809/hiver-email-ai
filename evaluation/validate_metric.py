"""
Validates that the scoring rubric actually discriminates good replies from
bad ones, rather than just producing plausible-looking numbers.

Method: for a set of dataset rows, we score THREE replies per row:
  - GOOD:     the real "sent" reply from the dataset.
  - MEDIOCRE: a generic, technically-not-wrong reply that avoids the
              specific failure modes below but also doesn't really engage
              with the specifics of the request (the realistic "meh, this
              might actually pass a lazy human review" case).
  - BAD:      a reply with one deliberate, named failure mode (ignores the
              question, invents facts, rude tone, no resolution, padded/
              rambling text, or a flatly wrong policy answer).

We run the exact same judge used in evaluate.py on all three tiers. If the
metric is meaningful we expect GOOD > MEDIOCRE > BAD, especially on
`relevance` and `correctness` (the two heaviest-weighted, most safety-
relevant dimensions). Testing three tiers instead of two is a stronger
check than good-vs-bad alone: a judge could trivially separate "great" from
"obviously terrible" while still being unable to distinguish genuinely good
work from merely adequate work, which is the harder and more useful
distinction in practice.

This is a lightweight discriminative-validity check, not proof the rubric
is perfect - but it's concrete, falsifiable evidence the score tracks real
quality rather than surface fluency.

Run:
  python3 validate_metric.py
Produces:
  evaluation/validation_report.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.retrieval import load_dataset
from generator.llm_backend import LLMError, PROVIDER
from rubric import DIMENSIONS, composite_score
from evaluate import score_one

# Each entry: dataset row id -> (failure_mode_label, mediocre_reply, bad_reply)
CASES = {
    "os-001": (
        "ignores_the_question",
        "Hi Priya,\n\nThanks for your email, we're on it. We'll keep you posted.\n\nBest,\nSupport Team",
        "Hi, thanks for your email. We appreciate your business! Let us know if there's anything else we can help with.\n\nBest,\nSupport Team",
    ),
    "rf-001": (
        "hallucinated_facts",
        "Hi James,\n\nSorry about the lamp. We'll look into a refund or replacement and follow up soon.\n\nBest,\nSupport Team",
        "Hi James,\n\nI've processed your refund of $129.99 and also upgraded you to our Platinum loyalty tier with free shipping for life. Your replacement lamp ships from our Chicago warehouse tomorrow via SpeedEx, tracking ZX881122.\n\nBest,\nSupport Team",
    ),
    "es-001": (
        "generic_apology_no_fix",
        "Hi Patricia,\n\nWe're sorry for the trouble and appreciate your patience. Someone will look into your refund soon.\n\nBest,\nSupport Team",
        "We already told you the refund is coming. Please stop emailing about this, there is nothing more we can do right now. Refunds just take time.\n\nBest,\nSupport Team",
    ),
    "pd-001": (
        "no_actionable_resolution",
        "Hi Wendy,\n\nThanks for letting us know about the blender. We'll review your case and get back to you with next steps.\n\nBest,\nSupport Team",
        "Hi Wendy,\n\nSorry to hear about your blender. That does sound frustrating, appliances breaking early is never fun. We take quality seriously and always want customers to be happy.\n\nBest,\nSupport Team",
    ),
    "bd-001": (
        "vague_acknowledgement",
        "Hi Helena,\n\nThanks for flagging the double charge. We're looking into your billing and will resolve it shortly.\n\nBest,\nSupport Team",
        "Hi Helena,\n\nThank you so much for reaching out to us today, we truly value your business and feedback! Our team works hard every day to make sure every customer has a wonderful experience with our products and services. We understand that billing issues can sometimes be confusing and we want you to know that we hear you and we're here for you every step of the way. Regarding your question about the charges, we will look into it and get back to you as soon as possible, thank you again for your patience and understanding during this process!\n\nBest,\nSupport Team",
    ),
    "cn-003": (
        "no_clear_decision",
        "Hi Kwame,\n\nThanks for your message about the $25 fee. We're reviewing your account and will follow up on whether anything can be done.\n\nBest,\nSupport Team",
        "Hi Kwame,\n\nUnfortunately that $25 fee is non-refundable per our terms of service, there's nothing we can do. All cancellation fees are final.\n\nBest,\nSupport Team",
    ),
    "aa-001": (
        "generic_troubleshoot",
        "Hi Isabel,\n\nSorry you're locked out. Please try resetting your password again and let us know if it still doesn't work.\n\nBest,\nSupport Team",
        "Hi Isabel,\n\nHave you tried turning your computer off and on again? That usually fixes login issues. Let us know if you're still having trouble after that.\n\nBest,\nSupport Team",
    ),
    "sd-001": (
        "generic_wait_and_see",
        "Hi Nadia,\n\nThanks for the update on your tracking. We'll monitor the shipment and reach out if anything changes.\n\nBest,\nSupport Team",
        "Hi, packages can take a while sometimes, it's probably still coming. Nothing we can really do on our end, just keep waiting and check the tracking page.\n\nBest,\nSupport Team",
    ),
    "rf-004": (
        "half_answer",
        "Hi Grace,\n\nThanks for your patience, we can look into a partial refund option for you. We'll follow up with specifics soon.\n\nBest,\nSupport Team",
        "Hi Grace,\n\nUnfortunately we don't offer partial refunds under any circumstances, you'll need to do a full return to get anything back.\n\nBest,\nSupport Team",
    ),
    "fr-003": (
        "deflects_no_info",
        "Hi Hugo,\n\nThanks for the feedback on rate limits. We're always looking at ways to improve the API, I'll pass this along.\n\nBest,\nSupport Team",
        "Hi Hugo, rate limits are fixed and can't be changed for any account, sorry.\n\nBest,\nSupport Team",
    ),
}


def _run_tier(row, reply, tier_name):
    dims = score_one(row["subject"], row["incoming_email"], [], reply)
    return {"tier": tier_name, "composite": composite_score(dims), "dims": {d: dims[d]["score"] for d in DIMENSIONS}}


def main():
    dataset = load_dataset()
    by_id = {row["id"]: row for row in dataset}

    triples = []
    print(f"Using LLM_PROVIDER={PROVIDER} for judging\n")

    for row_id, (failure_mode, mediocre_reply, bad_reply) in CASES.items():
        row = by_id[row_id]
        print(f"Validating on {row_id} ({failure_mode}) ...")
        try:
            good = _run_tier(row, row["reply"], "good")
            mediocre = _run_tier(row, mediocre_reply, "mediocre")
            bad = _run_tier(row, bad_reply, "bad")
        except (LLMError, ValueError) as e:
            print(f"  ! Judge call failed ({e}). Skipping.")
            continue
        triples.append({"id": row_id, "failure_mode": failure_mode, "good": good, "mediocre": mediocre, "bad": bad})

    if not triples:
        print("\nNo validation triples scored - check judge/LLM backend connectivity.")
        return

    n = len(triples)
    fully_ordered = sum(1 for t in triples if t["good"]["composite"] > t["mediocre"]["composite"] > t["bad"]["composite"])
    good_beats_bad = sum(1 for t in triples if t["good"]["composite"] > t["bad"]["composite"])
    good_beats_mediocre = sum(1 for t in triples if t["good"]["composite"] > t["mediocre"]["composite"])
    mediocre_beats_bad = sum(1 for t in triples if t["mediocre"]["composite"] > t["bad"]["composite"])

    mean_good = round(sum(t["good"]["composite"] for t in triples) / n, 3)
    mean_mediocre = round(sum(t["mediocre"]["composite"] for t in triples) / n, 3)
    mean_bad = round(sum(t["bad"]["composite"] for t in triples) / n, 3)

    per_dim_gap = {}
    for d in DIMENSIONS:
        mg = sum(t["good"]["dims"][d] for t in triples) / n
        mm = sum(t["mediocre"]["dims"][d] for t in triples) / n
        mb = sum(t["bad"]["dims"][d] for t in triples) / n
        per_dim_gap[d] = {"good_minus_mediocre": round(mg - mm, 2), "mediocre_minus_bad": round(mm - mb, 2)}

    report = {
        "n_triples": n,
        "fully_ordered_good_gt_mediocre_gt_bad": fully_ordered,
        "good_beats_bad": good_beats_bad,
        "good_beats_mediocre": good_beats_mediocre,
        "mediocre_beats_bad": mediocre_beats_bad,
        "pass": fully_ordered == n,
        "mean_composite_good": mean_good,
        "mean_composite_mediocre": mean_mediocre,
        "mean_composite_bad": mean_bad,
        "per_dimension_gap": per_dim_gap,
        "triples": triples,
    }

    out_path = os.path.join(os.path.dirname(__file__), "validation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{fully_ordered}/{n} cases: fully ordered good > mediocre > bad")
    print(f"  good beats bad: {good_beats_bad}/{n}")
    print(f"  good beats mediocre: {good_beats_mediocre}/{n}")
    print(f"  mediocre beats bad: {mediocre_beats_bad}/{n}")
    print(f"Mean composite - good: {mean_good}/5, mediocre: {mean_mediocre}/5, bad: {mean_bad}/5")
    print(f"\nWrote {out_path}")
    if not report["pass"]:
        print("\nNOTE: not every case was fully ordered - see triples in the report for which ones, and why (dimension-level detail included).")


if __name__ == "__main__":
    main()
