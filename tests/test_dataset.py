import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "emails.json")
INCOMING_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "incoming_emails.json")

REQUIRED_KEYS = {"id", "category", "customer_name", "subject", "incoming_email", "reply"}


def test_dataset_exists_and_parses():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 30, "expected at least 30 email/reply pairs"


def test_dataset_rows_have_required_fields():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    for row in data:
        missing = REQUIRED_KEYS - set(row.keys())
        assert not missing, f"row {row.get('id')} missing fields: {missing}"
        assert len(row["incoming_email"]) > 10
        assert len(row["reply"]) > 10


def test_dataset_ids_are_unique():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    ids = [row["id"] for row in data]
    assert len(ids) == len(set(ids)), "duplicate ids found in dataset"


def test_dataset_has_multiple_categories():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    categories = {row["category"] for row in data}
    assert len(categories) >= 5, "expected reasonable category diversity"


def test_incoming_emails_exist_and_parse():
    with open(INCOMING_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 5
    for row in data:
        assert "incoming_email" in row and "subject" in row and "id" in row


def test_incoming_emails_not_in_training_dataset():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    with open(INCOMING_PATH) as f:
        incoming = json.load(f)
    dataset_ids = {row["id"] for row in dataset}
    incoming_ids = {row["id"] for row in incoming}
    assert dataset_ids.isdisjoint(incoming_ids), "incoming test emails must not overlap dataset ids (would defeat the point of a held-out test set)"
