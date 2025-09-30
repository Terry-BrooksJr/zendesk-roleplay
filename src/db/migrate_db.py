#!/usr/bin/env python3
"""
Database migration script to update existing SQLite database schema.

Adds missing columns and ensures compatibility with the current codebase.
"""

import os
import sqlite3
from pathlib import Path


def migrate_database():
    """Migrate existing database to current schema."""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/data.db")

    # Extract database path from SQLite URL
    if database_url.startswith("sqlite:///"):
        db_path = database_url[10:]  # Remove "sqlite:///" prefix
    else:
        print(f"Unsupported database URL: {database_url}")
        return

    if not Path(db_path).exists():
        print(f"Database file {db_path} does not exist. Run init_db.py first.")
        return

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if last_updated column exists
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]

        if "last_updated" not in columns:
            print("Adding last_updated column to sessions table...")
            # SQLite doesn't support DEFAULT CURRENT_TIMESTAMP in ALTER TABLE
            # Add with NULL default, then update existing rows
            cursor.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN last_updated DATETIME
            """
            )
            cursor.execute(
                """
                UPDATE sessions
                SET last_updated = CURRENT_TIMESTAMP
                WHERE last_updated IS NULL
            """
            )
            print("✓ Added last_updated column")
        else:
            print("✓ last_updated column already exists")

        # Check if state_data column exists
        if "state_data" not in columns:
            print("Adding state_data column to sessions table...")
            cursor.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN state_data TEXT
            """
            )
            print("✓ Added state_data column")
        else:
            print("✓ state_data column already exists")

        conn.commit()
        print("Database migration completed successfully!")

    except sqlite3.Error as e:
        print(f"Migration error: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_database()
