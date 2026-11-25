import pytest

import admin_store


def test_messages_persist_and_can_be_organized_and_deleted(tmp_path):
    db_path = tmp_path / "admin" / "messages.sqlite3"
    first_id = admin_store.add_message(
        "Check this link", "BDO", "Smishing", {"LR": 0.97}, db_path=db_path
    )
    second_id = admin_store.add_message(
        "Your OTP is 123", "", "Legitimate", {"LR": 0.02}, db_path=db_path
    )

    admin_store.update_reviews(
        [{"id": first_id, "category": "Needs review", "notes": "Check the URL"}],
        db_path=db_path,
    )

    assert admin_store.list_messages(
        search="BDO", category="Needs review", db_path=db_path
    )[0]["notes"] == "Check the URL"
    assert admin_store.delete_messages([first_id, first_id], db_path=db_path) == 1
    remaining = admin_store.list_messages(db_path=db_path)
    assert [message["id"] for message in remaining] == [second_id]
    assert remaining[0]["probabilities"] == {"LR": 0.02}


def test_update_rejects_unknown_categories(tmp_path):
    with pytest.raises(ValueError, match="Unknown category"):
        admin_store.update_reviews(
            [{"id": 1, "category": "Delete me", "notes": ""}],
            db_path=tmp_path / "messages.sqlite3",
        )


def test_delete_all_messages(tmp_path):
    db_path = tmp_path / "messages.sqlite3"
    admin_store.add_message("first", "", "Uncertain", {}, db_path=db_path)
    admin_store.add_message("second", "", "Uncertain", {}, db_path=db_path)

    assert admin_store.delete_all_messages(db_path=db_path) == 2
    assert admin_store.list_messages(db_path=db_path) == []


def test_flags_persist_and_a_new_flag_replaces_the_old_label(tmp_path):
    db_path = tmp_path / "messages.sqlite3"
    admin_store.add_flag("Claim your prize", "09171234567", 1, db_path=db_path)
    admin_store.add_flag("Your OTP is 123", "BDO", 1, db_path=db_path)
    admin_store.add_flag("Your OTP is 123", "BDO", 0, db_path=db_path)  # changed their mind

    flags = {(f["text"], f["sender"]): f["label"] for f in admin_store.list_flags(db_path=db_path)}
    assert flags == {("Claim your prize", "09171234567"): 1, ("Your OTP is 123", "BDO"): 0}
    assert admin_store.clear_flags(db_path=db_path) == 2
    assert admin_store.list_flags(db_path=db_path) == []


def test_add_flag_rejects_other_labels(tmp_path):
    with pytest.raises(ValueError):
        admin_store.add_flag("text", "", 2, db_path=tmp_path / "messages.sqlite3")


def test_only_smishing_and_legitimate_decisions_are_used_for_training(tmp_path):
    db_path = tmp_path / "messages.sqlite3"
    ids = [admin_store.add_message(text, sender, "Uncertain", {}, db_path=db_path)
           for text, sender in [("scam text", "GCash"), ("bank text", "BDO"), ("unsure text", ""), ("new text", "")]]
    admin_store.update_reviews(
        [{"id": ids[0], "category": "Smishing", "notes": ""},
         {"id": ids[1], "category": "Legitimate", "notes": ""},
         {"id": ids[2], "category": "Needs review", "notes": ""}],
        db_path=db_path,
    )
    rows = admin_store.reviewed_for_training(db_path=db_path)
    assert sorted((r["text"], r["sender"], r["label"]) for r in rows) == [
        ("bank text", "BDO", 0), ("scam text", "GCash", 1)]


def test_a_message_submitted_twice_uses_the_latest_decision(tmp_path):
    db_path = tmp_path / "messages.sqlite3"
    first = admin_store.add_message("same text", "", "Smishing", {}, db_path=db_path)
    second = admin_store.add_message("same text", "", "Smishing", {}, db_path=db_path)
    admin_store.update_reviews(
        [{"id": first, "category": "Smishing", "notes": ""},
         {"id": second, "category": "Legitimate", "notes": ""}],
        db_path=db_path,
    )
    assert admin_store.reviewed_for_training(db_path=db_path) == [{"text": "same text", "sender": "", "label": 0}]
