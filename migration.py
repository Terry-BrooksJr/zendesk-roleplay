#!/usr/bin/env python3
"""
Database migration script for adding session state management.

This script safely adds the new columns needed for session state storage
and migrates existing data.
"""

import json
from datetime import datetime
from pathlib import Path

import psycopg
from loguru import logger
from psycopg.rows import dict_row


def run_migration():
    """Run the database migration to add session state columns."""

    db_path = Path("data.db")
    if not db_path.exists():
        logger.info("Database not found. Creating new database with updated schema.")
        # If no database exists, the new schema will be created automatically
        return True

    logger.info("Starting database migration...")
    with psycopg.connect(
        conninfo="postgresql://assessment_user:tep4XMU8efu*ydv!yqt@db.blackberry-py.dev:5432/assessment_db",
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cur:
            try:
                return _migrate(cur, conn)
            except Exception as e:
                logger.exception(f"Migration failed: {e}")
                if "conn" in locals():
                    conn.rollback()
                return False


def _migrate(cur, conn):
    """Performs the database schema migration for session state management.

    This function adds new columns to the sessions table, creates the milestones table if needed, and migrates existing session data to the new schema.

    Args:
        cur: The database cursor for executing SQL commands.
        conn: The database connection object.

    Returns:
        True if the migration was successful or already completed.
    """
    # Check if migration is needed
    introspection = cur.execute(
        """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM
                    information_schema.columns
                WHERE
                    table_name = 'sessions'
                    AND table_schema = 'public';
                """
    ).fetchall()
    logger.debug(f"Introspection result: {introspection}")
    columns = [col["column_name"] for col in introspection]
    print(columns)

    if "state_data" in columns and "last_updated" in columns:
        logger.success("Migration already completed. Database is up to date.")
        conn.close()
        return True

    logger.info("Adding new columns to sessions table...")

    # Add new columns if they don't exist
    if "state_data" not in columns:
        cur.execute(
            """
                        ALTER TABLE sessions 
                        ADD COLUMN state_data TEXT
                    """
        )
        logger.success("Added state_data column")

    if "last_updated" not in columns:
        cur.execute(
            """
                        ALTER TABLE sessions 
                        ADD COLUMN last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    """
        )
        logger.success("Added last_updated column")

    # Create milestones table if it doesn't exist
    cur.execute(
        """
                    CREATE TABLE IF NOT EXISTS milestones (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        milestone_id TEXT,
                        achieved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        turn_number INTEGER
                    )
                """
    )
    logger.success("Created milestones table")

    # Migrate existing sessions to have default state data
    logger.info("Migrating existing sessions...")
    cur.execute("SELECT id, started_at FROM sessions WHERE state_data IS NULL")
    sessions_to_migrate = cur.fetchall()

    for session_id, started_at in sessions_to_migrate:
        default_state = {
            "milestones": [],
            "penalty": 0.0,
            "transcript": [],
            "last_bot": "",
            "started_at": started_at or datetime.now().isoformat(),
        }

        cur.execute(
            """
                        UPDATE sessions 
                        SET state_data = ?, last_updated = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """,
            (json.dumps(default_state), session_id),
        )

    logger.success(f"Migrated {len(sessions_to_migrate)} existing sessions")

    # Commit all changes
    conn.commit()
    logger.success("Migration completed successfully!")
    return True


def backup_database():
    """Create a backup of the existing database before migration."""
    import shutil

    db_path = Path("data.db")
    if db_path.exists():
        backup_path = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to {backup_path}")
        return backup_path
    return None


if __name__ == "__main__":
    print("=== Database Migration Tool ===")

    # Create backup first
    backup_file = backup_database()
    if backup_file:
        print(f"Backup created: {backup_file}")

    if success := run_migration():
        logger.success("\n✅ Migration completed successfully!")
        print("Your application is now ready to use the enhanced session management.")
    else:
        print("\n❌ Migration failed!")
        if backup_file:
            logger.info(f"You can restore from backup: {backup_file}")
        exit(1)
