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
