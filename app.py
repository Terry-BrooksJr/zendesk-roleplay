import os, time, yaml, asyncio
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from intents import detect
from providers import make_provider
from scoring import compute_score
from storage import new_session, log_turn, end_session
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()
with open("scenario.yml") as f:
    SCEN = yaml.safe_load(f)

class Msg(BaseModel):
    session_id: str
    text: str

class StartReq(BaseModel):
    candidate_label: str = "anon"

app = FastAPI(title="Zendesk Roleplay (LangGraph)")

# --- Graph State ---
class State(dict):
    # state keys: session_id, transcript(list), milestones(set), started_at, penalty(float)
    pass

def node_router(state:State):
    return "reply_node"

async def reply_node_fn(state:State)->State:
    user_text = state["last_user"]
    intents = detect(user_text)
    milestones = set(state["milestones"])
    reply_parts = []

    # triggers
    if intents.get("ask_for_goal"): milestones.add("M1_goal")
    if intents.get("ask_for_context"): milestones.add("M3_context")
    if intents.get("ask_for_logs") or intents.get("ask_for_repro"):
        milestones.add("M2_logs")
        reply_parts.append("I’ve attached the HAR from my device.")

    # off-track heuristic: asked unrelated config without discovery/evidence
    off_track = ("propose_fix" in intents and "M1_goal" not in milestones and "M2_logs" not in milestones)
    if off_track:
        reply_parts.append(SCEN["challenge_injections"][0]["text"])
        state["penalty"] = state.get("penalty",0) + abs(SCEN["challenge_injections"][0]["score_effect"]["communication"])

    # assemble bot response (LLM for style, deterministic settings)
    provider = make_provider(
        system_prompt=SCEN["determinism"]["style"],
        temperature=float(SCEN["determinism"]["temperature"]),
        top_p=float(SCEN["determinism"]["top_p"])
    )
    # Compose a grounded prompt so the LLM can't leak the answer:
    content = ("User said: " + user_text + "\n"
               "Respond as the teacher. If HAR was requested, mention it's attached. "
               "Keep answers short and realistic. Do not propose the fix; answer only what was asked.")
    bot_text = await provider.chat([{"role":"user","content": content}])

    state["milestones"] = list(milestones)
    state["last_bot"] = ("\n\n".join(reply_parts + [bot_text])).strip()
    state["transcript"].append({"role":"user","text":user_text})
    state["transcript"].append({"role":"assistant","text":state["last_bot"]})
    return state

# wire graph
g = StateGraph(State)
g.add_node("reply_node", reply_node_fn)
g.set_entry_point("reply_node")
graph = g.compile()

# --- API ---
@app.post("/start")
async def start(req: StartReq):
    sid = new_session(req.candidate_label, SCEN["id"])
    state:State = {"session_id":sid, "transcript":[], "milestones":[], "started_at": time.time(), "penalty":0.0}
    seed = SCEN["seed_message"]
    state["last_bot"] = seed
    state["transcript"].append({"role":"assistant","text":seed})
    log_turn(sid, "bot", seed)
    return {"session_id": sid, "message": seed, "artifacts":[]}

@app.post("/reply")
async def reply(msg: Msg):
    # load state from a cache or reconstruct minimal (for demo, stateless in memory per req)
    # In prod, you'd persist State; here we recompute transcript via logs if needed.
    state:State = {"session_id": msg.session_id, "transcript":[], "milestones":[], "started_at": time.time(), "penalty":0.0}
    state["last_user"] = msg.text
    log_turn(msg.session_id, "user", msg.text)
    out:State = await graph.ainvoke(state)
    log_turn(msg.session_id, "bot", out["last_bot"])
    attachments = []
    if "M2_logs" in out["milestones"]:
        attachments.append(SCEN["artifacts"]["har"])
    return {"message": out["last_bot"], "attachments": attachments, "milestones": out["milestones"]}

@app.post("/finish/{session_id}")
async def finish(session_id:str):
    elapsed = 0.0  # if you persisted started_at, compute here
    end_session(session_id, elapsed)
    return {"ok": True}

@app.post("/score")
async def score(payload: Dict[str,Any]):
    # payload: {"milestones":["M1_goal","M2_logs","M3_context","M4_solution"], "penalty": 0.05}
    ms = set(payload.get("milestones",[]))
    rub = SCEN["rubric"]
    res = compute_score(ms, rub)
    comm_pen = payload.get("penalty",0.0)
    res["by_dimension"]["communication"] = max(0.0, res["by_dimension"]["communication"] - comm_pen)
    res["total"] = round(
        res["by_dimension"]["discovery"]*rub["weights"]["discovery"] +
        res["by_dimension"]["evidence"]*rub["weights"]["evidence"] +
        res["by_dimension"]["reasoning"]*rub["weights"]["reasoning"] +
        res["by_dimension"]["communication"]*rub["weights"]["communication"], 3
    )
    res["pass"] = res["total"] >= rub["pass_threshold"]
    return res