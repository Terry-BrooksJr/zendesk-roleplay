from sqlalchemy import create_engine, Table, Column, String, Text, Float, Integer, MetaData, DateTime
from sqlalchemy.sql import func
import uuid, hashlib, time

engine = create_engine("sqlite:///data.db", future=True)
md = MetaData()
sessions = Table("sessions", md,
    Column("id", String, primary_key=True),
    Column("candidate_hash", String),
    Column("scenario_id", String),
    Column("started_at", DateTime, server_default=func.now()),
    Column("ended_at", DateTime, nullable=True),
    Column("elapsed_sec", Float, default=0.0))
turns = Table("turns", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String),
    Column("speaker", String),  # user|bot
    Column("text", Text),
    Column("ts", DateTime, server_default=func.now()))
md.create_all(engine)

def new_session(candidate_label:str, scenario_id:str)->str:
    # pseudo-anonymize (hash label + salt)
    sid = str(uuid.uuid4())
    h = hashlib.sha256(f"{candidate_label}".encode()).hexdigest()[:16]
    with engine.begin() as c:
        c.execute(sessions.insert().values(id=sid, candidate_hash=h, scenario_id=scenario_id))
    return sid

def log_turn(sid:str, speaker:str, text:str):
    with engine.begin() as c:
        c.execute(turns.insert().values(session_id=sid, speaker=speaker, text=text))

def end_session(sid:str, elapsed:float):
    from sqlalchemy import update
    with engine.begin() as c:
        c.execute(update(sessions).where(sessions.c.id==sid).values(ended_at=func.now(), elapsed_sec=elapsed))