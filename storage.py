"""
module: storage

This module provides functions to ensure data persistence, It includes functionality to create new sessions, log conversation turns, and mark sessions as ended.
Enhanced with session state management for milestones, penalties, and application state.
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.sql import func

engine = create_engine(
    "postgresql+psycopg://assessment_user:tep4XMU8efu*ydv!yqt@db.blackberry-py.dev:5432/assessment_db",
    future=True,
)
md = MetaData()

sessions = Table(
    "sessions",
    md,
    Column("id", String, primary_key=True),
    Column("candidate_hash", String),
    Column("scenario_id", String),
    Column("started_at", DateTime, server_default=func.now()),
    Column("ended_at", DateTime, nullable=True),
    Column("elapsed_sec", Float, default=0.0),
    # NEW: Store session state as JSON
    Column(
        "state_data", Text, nullable=True
    ),  # JSON blob for milestones, penalty, etc.
    Column("last_updated", DateTime, server_default=func.now()),
)

turns = Table(
    "turns",
    md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String),
    Column("speaker", String),  # user|bot
    Column("text", Text),
    Column("ts", DateTime, server_default=func.now()),
)

# NEW: Optional table for structured milestone tracking
milestones = Table(
    "milestones",
    md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String),
    Column("milestone_id", String),  # e.g., "M1_goal", "M2_logs"
    Column("achieved_at", DateTime, server_default=func.now()),
    Column("turn_number", Integer),  # Which turn this was achieved on
)

md.create_all(engine)


def new_session(candidate_label: str, scenario_id: str) -> str:
    """Creates a new session entry in the database and returns its unique ID.

    This function generates a pseudo-anonymized candidate hash and stores the session with the provided scenario ID.

    Args:
        candidate_label (str): The label identifying the candidate.
        scenario_id (str): The identifier for the scenario.

    Returns:
        str: The unique session ID for the new session.
    """
    sid = str(uuid.uuid4())
    h = hashlib.sha256(f"{candidate_label}".encode()).hexdigest()[:16]

    # Initialize empty session state
    initial_state = {
        "milestones": [],
        "penalty": 0.0,
        "transcript": [],
        "last_bot": "",
        "started_at": datetime.now().isoformat(),
    }

    with engine.begin() as c:
        c.execute(
            sessions.insert().values(
                id=sid,
                candidate_hash=h,
                scenario_id=scenario_id,
                state_data=json.dumps(initial_state),
            )
        )
    return sid


def log_turn(sid: str, speaker: str, text: str):
    """Logs a conversation turn for a session in the database.

    This function records the speaker and their message for the specified session.

    Args:
        sid (str): The session ID.
        speaker (str): The role of the speaker, such as 'user' or 'bot'.
        text (str): The content of the message.
    """
    with engine.begin() as c:
        c.execute(turns.insert().values(session_id=sid, speaker=speaker, text=text))


def end_session(sid: str, elapsed: float):
    """Marks a session as ended and updates its elapsed time in the database.

    This function sets the end timestamp and the total elapsed seconds for the specified session.

    Args:
        sid (str): The session ID to end.
        elapsed (float): The total elapsed time in seconds for the session.
    """
    with engine.begin() as c:
        c.execute(
            update(sessions)
            .where(sessions.c.id == sid)
            .values(ended_at=datetime.now(), elapsed_sec=elapsed)
        )


def get_session(sid: str) -> dict | None:
    """Retrieves session details from the database.

    This function fetches the session information for the specified session ID.
    Now includes parsed state data for application use.

    Args:
        sid (str): The session ID to retrieve.

    Returns:
        dict | None: A dictionary containing session details or None if not found.
    """
    with engine.begin() as c:
        result = c.execute(select(sessions).where(sessions.c.id == sid)).first()

        if not result:
            return None

        session_data = result._asdict()

        # Parse JSON state data
        if session_data.get("state_data"):
            try:
                state = json.loads(session_data["state_data"])
                session_data.update(state)  # Merge state data into main dict
            except (json.JSONDecodeError, TypeError) as e:
                # Handle corrupted state data gracefully
                print(f"Warning: Failed to parse state data for session {sid}: {e}")
                session_data.update(
                    {
                        "milestones": [],
                        "penalty": 0.0,
                        "transcript": [],
                        "last_bot": "",
                    }
                )
        else:
            # Provide defaults if no state data exists
            session_data.update(
                {
                    "milestones": [],
                    "penalty": 0.0,
                    "transcript": [],
                    "last_bot": "",
                }
            )

        return session_data


def update_session(sid: str, state_data: Dict[str, Any]) -> bool:
    """Updates session state data in the database.

    This function stores the current session state (milestones, penalty, transcript, etc.)
    as JSON in the database.

    Args:
        sid (str): The session ID to update.
        state_data (Dict[str, Any]): Dictionary containing session state to store.

    Returns:
        bool: True if update successful, False otherwise.
    """
    try:
        # Ensure we have the required fields
        required_fields = ["milestones", "penalty", "transcript", "last_bot"]
        for field in required_fields:
            if field not in state_data:
                state_data[field] = (
                    []
                    if field in ["milestones", "transcript"]
                    else (0.0 if field == "penalty" else "")
                )

        # Convert datetime objects to ISO strings for JSON serialization
        json_safe_data = {}
        for key, value in state_data.items():
            if isinstance(value, datetime):
                json_safe_data[key] = value.isoformat()
            else:
                json_safe_data[key] = value

        with engine.begin() as c:
            result = c.execute(
                update(sessions)
                .where(sessions.c.id == sid)
                .values(
                    state_data=json.dumps(json_safe_data), last_updated=datetime.now()
                )
            )
            return result.rowcount > 0
    except Exception as e:
        print(f"Error updating session {sid}: {e}")
        return False


def add_milestone(sid: str, milestone_id: str, turn_number: int = None) -> bool:
    """Adds a milestone achievement to the database.

    This function records when a specific milestone was achieved during the session.
    Also updates the main session state.

    Args:
        sid (str): The session ID.
        milestone_id (str): The milestone identifier (e.g., "M1_goal").
        turn_number (int, optional): The turn number when achieved.

    Returns:
        bool: True if milestone was added successfully, False otherwise.
    """
    try:
        # First, check if milestone already exists
        with engine.begin() as c:
            existing = c.execute(
                select(milestones).where(
                    (milestones.c.session_id == sid)
                    & (milestones.c.milestone_id == milestone_id)
                )
            ).first()

            if existing:
                return False  # Milestone already exists

            # Add milestone record
            c.execute(
                milestones.insert().values(
                    session_id=sid, milestone_id=milestone_id, turn_number=turn_number
                )
            )

            # Update session state data
            session = get_session(sid)
            if session:
                current_milestones = session.get("milestones", [])
                if milestone_id not in current_milestones:
                    current_milestones.append(milestone_id)

                state_update = {
                    "milestones": current_milestones,
                    "penalty": session.get("penalty", 0.0),
                    "transcript": session.get("transcript", []),
                    "last_bot": session.get("last_bot", ""),
                }
                update_session(sid, state_update)

        return True
    except Exception as e:
        print(f"Error adding milestone {milestone_id} to session {sid}: {e}")
        return False


def get_session_milestones(sid: str) -> List[Dict[str, Any]]:
    """Retrieves all milestones for a session with timestamps.

    Args:
        sid (str): The session ID.

    Returns:
        List[Dict[str, Any]]: List of milestone records with timestamps.
    """
    try:
        with engine.begin() as c:
            results = c.execute(
                select(milestones)
                .where(milestones.c.session_id == sid)
                .order_by(milestones.c.achieved_at)
            ).fetchall()

            return [result._asdict() for result in results]
    except Exception as e:
        print(f"Error retrieving milestones for session {sid}: {e}")
        return []


def get_session_transcript(sid: str) -> List[Dict[str, Any]]:
    """Retrieves the full conversation transcript for a session.

    Args:
        sid (str): The session ID.

    Returns:
        List[Dict[str, Any]]: List of turn records with timestamps.
    """
    try:
        with engine.begin() as c:
            results = c.execute(
                select(turns).where(turns.c.session_id == sid).order_by(turns.c.ts)
            ).fetchall()

            return [result._asdict() for result in results]
    except Exception as e:
        print(f"Error retrieving transcript for session {sid}: {e}")
        return []


def cleanup_old_sessions(days_old: int = 30) -> int:
    """Removes session data older than specified days.

    Args:
        days_old (int): Number of days after which to remove sessions.

    Returns:
        int: Number of sessions cleaned up.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)

        with engine.begin() as c:
            # Delete old turns first (foreign key constraint)
            c.execute(
                turns.delete().where(
                    turns.c.session_id.in_(
                        select(sessions.c.id).where(sessions.c.started_at < cutoff_date)
                    )
                )
            )

            # Delete old milestones
            c.execute(
                milestones.delete().where(
                    milestones.c.session_id.in_(
                        select(sessions.c.id).where(sessions.c.started_at < cutoff_date)
                    )
                )
            )

            # Delete old sessions
            result = c.execute(
                sessions.delete().where(sessions.c.started_at < cutoff_date)
            )

            return result.rowcount
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return 0


# Migration function to update existing sessions
def migrate_existing_sessions():
    """Migrates existing sessions to include state_data column.

    This function should be run once to populate state_data for existing sessions.
    """
    try:
        with engine.begin() as c:
            # Get all sessions without state_data
            results = c.execute(
                select(sessions).where(sessions.c.state_data.is_(None))
            ).fetchall()

            for session in results:
                # Create default state data
                default_state = {
                    "milestones": [],
                    "penalty": 0.0,
                    "transcript": [],
                    "last_bot": "",
                    "started_at": (
                        session.started_at.isoformat()
                        if session.started_at
                        else datetime.now().isoformat()
                    ),
                }

                # Update session with default state
                c.execute(
                    update(sessions)
                    .where(sessions.c.id == session.id)
                    .values(state_data=json.dumps(default_state))
                )

            print(f"Migrated {len(results)} sessions")
    except Exception as e:
        print(f"Migration error: {e}")
