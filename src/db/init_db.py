#!/usr/bin/env python3
"""
Database initialization script for SQLite setup.

Creates the necessary tables for the chatbot application.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine

from ..common.storage import md


def init_database():
    """Initialize the SQLite database with required tables."""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/data.db")

    print(f"Initializing database: {database_url}")

    # Extract database path to check if it exists
    if database_url.startswith("sqlite:///"):
        db_path = database_url[10:]
        if Path(db_path).exists():
            print(f"Database file {db_path} already exists.")
            print(
                "If you need to recreate it, delete the file first or run migrate_db.py to update schema."
            )
            return

    # Create engine
    engine = create_engine(database_url, future=True, echo=True)

    # Create all tables
    md.create_all(engine)

    print("Database initialized successfully!")
    print("Tables created:")
    for table_name in md.tables.keys():
        print(f"  - {table_name}")


if __name__ == "__main__":
    init_database()
