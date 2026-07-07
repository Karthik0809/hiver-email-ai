"""
Scoring rubric for suggested email replies.

Why not exact-match / BLEU / ROUGE against the "reference" reply:
There is no single correct reply to a customer email - two agents could
write differently worded replies that are both excellent (or both bad in
different ways). Scoring against string overlap with one reference reply
would penalize perfectly good responses that just use different wording,
and would reward responses that copy phrasing without actually being
correct or useful. So "accurate" here has to mean something like "would a
support lead approve sending this," not "matches the reference text."

Instead we score each generated reply directly, on the dimensions that
actually determine whether a support lead would approve sending it:

  relevance      - Does it address what the customer actually asked/needs,
                    not a generic non-answer?
  completeness   - If the customer raised multiple points, are all of them
                    addressed?
  correctness    - Does it avoid inventing facts it couldn't know (specific
                    order numbers, amounts, dates not present in the email)
                    and avoid making promises inconsistent with how this
                    team actually operates (per the retrieved examples)?
                    This is the dimension most tied to real-world harm if
                    wrong, so it is weighted most heavily.
  tone           - Is it professional, empathetic where warranted, and
                    appropriately matched to the customer's emotional state?
  actionability  - Does it give a concrete next step / resolution rather
                    than just acknowledging the problem?
  conciseness    - Is it appropriately brief, without padding or repetition?

Each dimension is scored 0-5 by an LLM judge with a short justification,
so a human can audit *why* a score was given, not just see a number.
The composite score is a weighted average, weighted toward relevance and
correctness because a fluent, well-toned reply that answers the wrong
question or invents facts is actively worse than a plain but correct one.
"""

DIMENSIONS = ["relevance", "completeness", "correctness", "tone", "actionability", "conciseness"]

WEIGHTS = {
    "relevance": 0.25,
    "correctness": 0.25,
    "completeness": 0.15,
    "actionability": 0.15,
    "tone": 0.10,
    "conciseness": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

JUDGE_SYSTEM_PROMPT = """You are a meticulous support-team lead auditing a suggested email reply before it goes out to a customer.

Score the SUGGESTED REPLY (not the reference examples) on these six dimensions, each from 0 to 5:
- relevance: does it address what the customer actually asked or is upset about?
- completeness: if the customer raised multiple distinct points, are all addressed?
- correctness: does it avoid inventing specific facts (order numbers, dollar amounts, dates, tracking numbers) that were not given, and avoid promises inconsistent with how this team operates based on the reference examples?
- tone: is it professional and appropriately empathetic for the situation?
- actionability: does it give a concrete next step or resolution, not just an acknowledgement?
- conciseness: is it appropriately brief without padding, repetition, or filler?

Be strict and specific. A reply that sounds polished but is vague, evasive, or invents details should score LOW on relevance/correctness even if tone is good.

Respond with ONLY valid JSON, no other text, in exactly this shape:
{
  "relevance": {"score": <0-5 integer>, "why": "<one sentence>"},
  "completeness": {"score": <0-5 integer>, "why": "<one sentence>"},
  "correctness": {"score": <0-5 integer>, "why": "<one sentence>"},
  "tone": {"score": <0-5 integer>, "why": "<one sentence>"},
  "actionability": {"score": <0-5 integer>, "why": "<one sentence>"},
  "conciseness": {"score": <0-5 integer>, "why": "<one sentence>"}
}"""


def build_judge_prompt(incoming_subject, incoming_email, reference_examples, suggested_reply):
    parts = []
    if reference_examples:
        parts.append("Reference examples of how this team has replied to similar past emails (for tone/policy context only):\n")
        for ex in reference_examples:
            parts.append(f"- Customer said: {ex['incoming_email'][:200]}\n  Team replied: {ex['reply'][:200]}\n")
    parts.append(
        f"\nCUSTOMER EMAIL:\nSubject: {incoming_subject}\n{incoming_email}\n\n"
        f"SUGGESTED REPLY TO SCORE:\n{suggested_reply}\n"
    )
    return "\n".join(parts)


def composite_score(dimension_scores: dict) -> float:
    total = sum(dimension_scores[d]["score"] * WEIGHTS[d] for d in DIMENSIONS)
    return round(total, 3)
