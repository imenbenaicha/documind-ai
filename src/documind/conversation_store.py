
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# Database stored in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "documind.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create conversation and message tables."""

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation
            ON messages(conversation_id)
        """)


def create_conversation(document_id, document_name):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO conversations (
                document_id,
                document_name,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            document_id,
            document_name,
            "New conversation",
            now,
            now
        ))

        return cursor.lastrowid


def list_conversations(document_id):
    with get_connection() as connection:

        rows = connection.execute("""
            SELECT *
            FROM conversations
            WHERE document_id = ?
            ORDER BY updated_at DESC
        """, (document_id,)).fetchall()

        return [dict(row) for row in rows]


def add_message(conversation_id, role, content):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO messages (
                conversation_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            conversation_id,
            role,
            content,
            now
        ))

        # Update conversation title using its first user message
        if role == "user":

            row = connection.execute("""
                SELECT title
                FROM conversations
                WHERE id = ?
            """, (conversation_id,)).fetchone()

            if row and row["title"] == "New conversation":

                title = content[:60]

                connection.execute("""
                    UPDATE conversations
                    SET title = ?
                    WHERE id = ?
                """, (title, conversation_id))

        connection.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
        """, (now, conversation_id))


def get_messages(conversation_id):
    with get_connection() as connection:

        rows = connection.execute("""
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

        return [dict(row) for row in rows]


def delete_conversation(conversation_id):
    with get_connection() as connection:

        connection.execute("PRAGMA foreign_keys = ON")

        connection.execute("""
            DELETE FROM conversations
            WHERE id = ?
        """, (conversation_id,))
