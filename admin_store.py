"""Persistent storage for messages submitted to the analyzer."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).parent / "admin_messages.sqlite3"
CATEGORIES = ("Uncategorized", "Needs review", "Smishing", "Legitimate")
# Admin categories that count as a decision on the label, and are used when the models are retrained
TRAINING_LABELS = {"Smishing": 1, "Legitimate": 0}


def _connect(db_path=None):
    path = Path(db_path or os.getenv("SMISHGUARD_DB_PATH", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection(db_path=None):
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize(db_path=None):
    with _connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyzed_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sender TEXT NOT NULL,
                text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                probabilities TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'Uncategorized',
                notes TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Results that users marked as wrong in the Analyze tab, with the label they say is correct.
        # One row per message and sender; marking the same message again replaces the label
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sender TEXT NOT NULL,
                text TEXT NOT NULL,
                label INTEGER NOT NULL CHECK (label IN (0, 1)),
                UNIQUE (text, sender)
            )
            """
        )


def add_message(text, sender, verdict, probabilities, db_path=None):
    initialize(db_path)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyzed_messages
                (created_at, sender, text, verdict, probabilities)
            VALUES (?, ?, ?, ?, ?)
            """,
            (created_at, sender, text, verdict, json.dumps(probabilities, sort_keys=True)),
        )
        return cursor.lastrowid


def list_messages(search="", verdict="All", category="All", db_path=None):
    initialize(db_path)
    clauses, values = [], []
    if search:
        clauses.append("(text LIKE ? OR sender LIKE ?)")
        values.extend((f"%{search}%", f"%{search}%"))
    if verdict != "All":
        clauses.append("verdict = ?")
        values.append(verdict)
    if category != "All":
        clauses.append("category = ?")
        values.append(category)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM analyzed_messages{where} ORDER BY created_at DESC, id DESC",
            values,
        ).fetchall()
    messages = [dict(row) for row in rows]
    for message in messages:
        message["probabilities"] = json.loads(message["probabilities"])
    return messages


def update_reviews(reviews, db_path=None):
    updates = []
    for review in reviews:
        category = review["category"]
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        updates.append((category, str(review["notes"]), int(review["id"])))
    initialize(db_path)
    with _connection(db_path) as connection:
        connection.executemany(
            "UPDATE analyzed_messages SET category = ?, notes = ? WHERE id = ?",
            updates,
        )


def delete_messages(message_ids, db_path=None):
    ids = list(dict.fromkeys(int(message_id) for message_id in message_ids))
    if not ids:
        return 0
    initialize(db_path)
    placeholders = ", ".join("?" for _ in ids)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            f"DELETE FROM analyzed_messages WHERE id IN ({placeholders})",
            ids,
        )
        return cursor.rowcount


def delete_all_messages(db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM analyzed_messages")
        return cursor.rowcount


def add_flag(text, sender, label, db_path=None):
    """Save a user's correction (1 = smishing, 0 = legitimate) for a message."""
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, not {label!r}")
    initialize(db_path)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO flags (created_at, sender, text, label) VALUES (?, ?, ?, ?)
            ON CONFLICT (text, sender) DO UPDATE SET label = excluded.label, created_at = excluded.created_at
            """,
            (created_at, sender, text, int(label)),
        )


def list_flags(db_path=None):
    """All flags, oldest first, as dicts with text, sender, label and created_at."""
    initialize(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute(
            "SELECT text, sender, label, created_at FROM flags ORDER BY created_at, id"
        ).fetchall()
    return [dict(row) for row in rows]


def clear_flags(db_path=None):
    initialize(db_path)
    with _connection(db_path) as connection:
        return connection.execute("DELETE FROM flags").rowcount


def reviewed_for_training(db_path=None):
    """Messages an admin has categorized as Smishing or Legitimate, as dicts with text, sender and
    label. A message submitted more than once counts once, with its most recent decision."""
    initialize(db_path)
    placeholders = ", ".join("?" for _ in TRAINING_LABELS)
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT text, sender, category FROM analyzed_messages WHERE category IN ({placeholders}) "
            "ORDER BY created_at DESC, id DESC",
            list(TRAINING_LABELS),
        ).fetchall()
    latest = {}
    for row in rows:
        latest.setdefault((row["text"], row["sender"]), TRAINING_LABELS[row["category"]])
    return [{"text": text, "sender": sender, "label": label} for (text, sender), label in latest.items()]
