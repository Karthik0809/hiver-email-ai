"""
Gen-AI suggested-reply generator.

For each new incoming email:
  1. Retrieve the k most similar past email/reply pairs from the dataset
     (TF-IDF cosine similarity over subject+body).
  2. Build a few-shot prompt: those retrieved pairs act as grounded
     examples of "how this team actually replies" (tone, level of detail,
     what commitments are reasonable to make).
  3. Ask the LLM to draft a reply to the new email in that style.

Why few-shot retrieval instead of fine-tuning or a bigger RAG stack:
  - The dataset here is small (~40 rows) — nowhere near enough to
    fine-tune a model responsibly; it would badly overfit.
  - Prompting with retrieved real examples gets most of the benefit
    (style/tone grounding, policy consistency) with none of the
    infrastructure cost, and updates instantly if the dataset changes
    (no retraining step) — important for a support inbox where policies
    change often.
  - Trade-off: this leans on the base model's general reasoning ability
    for the actual content of the reply (facts about THIS customer's
    order) — same as a human agent would, using the dataset as a style/
    policy guide rather than a source of ground-truth account data.

Run:
  python3 generate.py
Produces:
  generator/outputs.json — one generated reply per row in
  dataset/incoming_emails.json, plus the retrieved examples used.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.retrieval import load_dataset, EmailRetriever
from llm_backend import chat, LLMError, PROVIDER

K = 3

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce/SaaS company, writing email replies in a shared support inbox.

Write replies that:
- Directly address what the customer actually asked or is upset about
- Are warm but professional, and concise (not padded with filler)
- Only make commitments you can plausibly keep (don't invent policies not implied by the examples)
- Include a clear next step or resolution, not just an acknowledgement
- Follow the tone and structure of the example replies below, which are real replies this team has sent for similar situations

Sign off as "Support Team". Do not include a subject line, just the email body."""


def build_user_prompt(incoming, examples):
    parts = ["Here are examples of past emails this team received and how they replied:\n"]
    for ex, score in examples:
        parts.append(
            f"---\nCustomer email:\nSubject: {ex['subject']}\n{ex['incoming_email']}\n\n"
            f"Reply that was sent:\n{ex['reply']}\n"
        )
    parts.append(
        "---\nNow write a reply to this new customer email, in the same style:\n"
        f"Subject: {incoming['subject']}\n{incoming['incoming_email']}\n"
    )
    return "\n".join(parts)


def generate_reply(incoming, retriever):
    examples = retriever.top_k(incoming["subject"], incoming["incoming_email"], k=K)
    user_prompt = build_user_prompt(incoming, examples)
    reply = chat(SYSTEM_PROMPT, user_prompt, temperature=0.4)
    return reply.strip(), examples


def main():
    dataset = load_dataset()
    retriever = EmailRetriever(dataset)

    incoming_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "incoming_emails.json")
    with open(incoming_path) as f:
        incoming_emails = json.load(f)

    results = []
    print(f"Using LLM_PROVIDER={PROVIDER}\n")
    for incoming in incoming_emails:
        print(f"Generating reply for {incoming['id']}: {incoming['subject']!r} ...")
        try:
            reply, examples = generate_reply(incoming, retriever)
        except LLMError as e:
            print(f"  ! LLM call failed ({e}). Is Ollama running? "
                  f"See README for setup. Skipping.")
            continue
        results.append({
            "id": incoming["id"],
            "subject": incoming["subject"],
            "incoming_email": incoming["incoming_email"],
            "generated_reply": reply,
            "retrieved_example_ids": [ex["id"] for ex, _ in examples],
            "retrieved_scores": [round(score, 3) for _, score in examples],
        })

    out_path = os.path.join(os.path.dirname(__file__), "outputs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} generated replies to {out_path}")
    if not results:
        print("\nNo replies were generated — check that your LLM backend is reachable "
              "(see README 'How to run').")


if __name__ == "__main__":
    main()
