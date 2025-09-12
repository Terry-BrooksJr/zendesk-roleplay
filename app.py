import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel, Field, validator

try:
    import structlog  # type: ignore
    _STRUCTLOG_AVAILABLE = True
except ImportError:  # Fallback if structlog isn't installed
    _STRUCTLOG_AVAILABLE = False
    structlog = None  # sentinel for type checkers

from intents import detect
from providers import make_provider
from scoring import compute_score
from storage import (
    get_session, 
    end_session, 
    log_turn, 
    new_session, 
    update_session,
    add_milestone,
    get_session_milestones,
    get_session_transcript
)
# Load environment and scenario configuration
load_dotenv()

# Cache scenario configuration at startup
SCEN: Optional[Dict[str, Any]] = None
PROVIDER_CACHE: Dict[str, Any] = {}
# Configure structured logging (with fallback if structlog is missing)
if _STRUCTLOG_AVAILABLE:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    struct_logger = structlog.get_logger()
else:
    # Minimal fallback: use loguru for structured-ish logs
    class _StructLoggerFallback:
        def __getattr__(self, name):
            # Map structlog method names to loguru methods; default to info
            def _log(*args, **kwargs):
                message = args[0] if args else name
                # Merge key/values into a string for visibility
                if kwargs:
                    message = f"{message} | " + ", ".join(
                        f"{k}={v}" for k, v in kwargs.items()
                    )
                getattr(
                    logger,
                    name if name in {"debug", "info", "warning", "error"} else "info",
                )(message)
            return _log

    struct_logger = _StructLoggerFallback()





@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    global SCEN, PROVIDER_CACHE

    logger.info("Starting FastAPI application")
    struct_logger.info("application.startup", action="loading_scenario")

    try:
        with open("scenario.yml", mode="r", encoding="utf-8") as f:
            SCEN = yaml.safe_load(f)

        # Validate required fields before proceeding
        required_fields = ["determinism", "id", "seed_message", "artifacts", "rubric"]
        for field in required_fields:
            if field not in SCEN:
                raise ValueError(f"Missing required field in scenario.yml: {field}")

        # Pre-warm provider cache
        PROVIDER_CACHE["default"] = make_provider(
            system_prompt=SCEN["determinism"]["style"],
            temperature=float(SCEN["determinism"]["temperature"]),
            top_p=float(SCEN["determinism"]["top_p"]),
        )

        struct_logger.info(
            "application.startup.complete",
            scenario_id=SCEN.get("id"),
            determinism_config=SCEN.get("determinism"),
        )

    except (KeyError, ValueError, FileNotFoundError) as e:
        struct_logger.error("application.startup.failed", error=str(e), exc_info=True)
        logger.error(f"Failed to start application: {str(e)}")
        import sys
        sys.exit(1)
    except Exception as e:
        struct_logger.error(
            "application.startup.unknown_error", error=str(e), exc_info=True
        )
        logger.error(f"Unknown error during startup: {str(e)}")
        import sys
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application")
    struct_logger.info("application.shutdown")


class Msg(BaseModel):
    """Represents a user message within a session.

    This model contains the session identifier and the text of the user's message.
    """

    session_id: str = Field(
        ..., min_length=1, description="The unique identifier for the session"
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The content of the user's message",
    )

    @validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Message text cannot be empty or whitespace only")
        return v.strip()


class StartReq(BaseModel):
    """Represents the request body for starting a new session."""

    candidate_label: str = Field(
        default="anon",
        min_length=1,
        max_length=100,
        description="The label identifying the candidate",
    )


app = FastAPI(
    title=os.environ.get("TITLE", "Zendesk Roleplay (LangGraph)"), lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure appropriately for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Enhanced Graph State ---


class State(dict):
    """Represents the state of a session during the conversation.

    This class extends the dictionary to store session information such as
    transcript, milestones, and timing with better type safety.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure default keys exist
        self.setdefault("milestones", [])
        self.setdefault("transcript", [])
        self.setdefault("penalty", 0.0)
        self.setdefault("started_at", time.time())

    def get_session_id(self) -> str:
        """Get session ID with validation."""
        if session_id := self.get("session_id"):
            return session_id
        else:
            raise ValueError("Session ID not found in state")

    def add_milestone(self, milestone: str) -> None:
        """Add milestone if not already present, preserving order."""
        milestones = self.get("milestones", [])
        if milestone not in milestones:
            milestones.append(milestone)
            self["milestones"] = milestones

    def add_penalty(self, penalty: float) -> None:
        """Add penalty to current total."""
        self["penalty"] = self.get("penalty", 0.0) + penalty

    def append_to_transcript(self, role: str, text: str) -> None:
        """Add message to transcript."""
        if "transcript" not in self:
            self["transcript"] = []
        self["transcript"].append({"role": role, "text": text})


# --- Safe helpers that work even if LangGraph hands us a plain dict ---


def _append_transcript(state: dict, role: str, text: str) -> None:
    transcript = state.get("transcript")
    if not isinstance(transcript, list):
        logger.warning(
            "Transcript in state is not a list (type: %s). Overwriting and backing up old value.",
            type(transcript).__name__,
        )
        logger.debug(f"Old transcript value:{transcript}")
        state["_transcript_backup"] = transcript
        state["transcript"] = []
    state["transcript"].append({"role": role, "text": text})


def node_router(state: State) -> str:
    """Determines the next node to execute in the conversation graph.

    Args:
        state (State): The current session state.

    Returns:
        str: The name of the next node to execute.
    """
    struct_logger.debug("graph.node_router", session_id=state.get("session_id"))
    return "reply_node"


async def reply_node_fn(state: State) -> State:
    """Generates reply based on user input and updates the session state.

    This function analyzes user intent, updates milestones, applies penalties
    if necessary, and produces a contextually appropriate bot response.

    Args:
        state (State): The current session state containing user input and progress.

    Returns:
        State: The updated session state with the bot's reply and milestone changes.
    """
    # Ensure we're working with proper State object
    if not isinstance(state, State):
        state = State(state)
    
    session_id = state.get("session_id", "unknown")
    start_time = time.time()

    struct_logger.info(
        "graph.reply_node.start",
        session_id=session_id,
        milestones_count=len(state.get("milestones", [])),
        transcript_length=len(state.get("transcript", [])),
    )

    try:
        user_text = state.get("last_user", "")

        if not user_text:
            # Defensive guard with logging
            struct_logger.warning("graph.reply_node.no_input", session_id=session_id)
            fallback_message = "I'm ready when you are—what's the customer's message?"
            state["last_bot"] = fallback_message
            _append_transcript(state, "assistant", fallback_message)
            return state

        # Intent detection with timeout and error handling
        intent_start = time.time()
        try:
            intents = detect(user_text)
        except Exception as intent_error:
            struct_logger.error(
                "intent.detection.failed",
                session_id=session_id,
                error=str(intent_error),
                exc_info=True
            )
            intents = {}  # Fallback to empty intents
            
        intent_duration = time.time() - intent_start

        struct_logger.info(
            "graph.intent_detection",
            session_id=session_id,
            intents=intents,
            duration_ms=round(intent_duration * 1000, 2),
            user_text_length=len(user_text),
        )

        milestones = set(state.get("milestones", []))
        initial_milestone_count = len(milestones)
        reply_parts = []

        # Process intents and update milestones
        if intents.get("ask_for_goal"):
            milestones.add("M1_goal")
            struct_logger.info(
                "milestone.achieved", session_id=session_id, milestone="M1_goal"
            )

        if intents.get("ask_for_context"):
            milestones.add("M3_context")
            struct_logger.info(
                "milestone.achieved", session_id=session_id, milestone="M3_context"
            )

        if intents.get("ask_for_logs") or intents.get("ask_for_repro"):
            milestones.add("M2_logs")
            reply_parts.append("I've attached the HAR from my device.")
            struct_logger.info(
                "milestone.achieved", session_id=session_id, milestone="M2_logs"
            )

        # Off-track detection with detailed logging
        off_track = (
            "propose_fix" in intents
            and "M1_goal" not in milestones
            and "M2_logs" not in milestones
        )

        if off_track:
            try:
                challenge_text = SCEN["challenge_injections"][0]["text"]
                penalty_amount = abs(
                    SCEN["challenge_injections"][0]["score_effect"]["communication"]
                )
                reply_parts.append(challenge_text)
                current_penalty = state.get("penalty", 0.0)
                state["penalty"] = current_penalty + penalty_amount

                struct_logger.warning(
                    "conversation.off_track",
                    session_id=session_id,
                    penalty_applied=penalty_amount,
                    total_penalty=state["penalty"],
                    missing_milestones=["M1_goal", "M2_logs"],
                )
            except (KeyError, IndexError) as config_error:
                struct_logger.error(
                    "challenge_injection.config_error",
                    session_id=session_id,
                    error=str(config_error)
                )

        # Generate LLM response with better error handling
        llm_start = time.time()
        try:
            provider = PROVIDER_CACHE.get("default")
            if not provider:
                # Fallback provider creation
                provider = make_provider(
                    system_prompt=SCEN["determinism"]["style"],
                    temperature=float(SCEN["determinism"]["temperature"]),
                    top_p=float(SCEN["determinism"]["top_p"]),
                )

            content = (
                f"User said: {user_text}\n"
                "Respond as the teacher. If HAR was requested, mention it's attached. "
                "Keep answers short and realistic. Do not propose the fix; answer only what was asked."
            )

            bot_text = await provider.chat([{"role": "user", "content": content}])
            llm_duration = time.time() - llm_start

            struct_logger.info(
                "llm.response_generated",
                session_id=session_id,
                duration_ms=round(llm_duration * 1000, 2),
                response_length=len(bot_text),
                prompt_length=len(content),
            )

        except Exception as e:
            llm_duration = time.time() - llm_start
            bot_text = "Got it. I'll take a closer look—could you clarify the customer's goal or share any logs/HAR if available?"

            struct_logger.error(
                "llm.response_failed",
                session_id=session_id,
                error=str(e),
                duration_ms=round(llm_duration * 1000, 2),
                fallback_used=True,
                exc_info=True,
            )

        # Update state
        state["milestones"] = list(milestones)
        state["last_bot"] = ("\n\n".join(reply_parts + [bot_text])).strip()
        _append_transcript(state, "user", user_text)
        _append_transcript(state, "assistant", state["last_bot"])

        total_duration = time.time() - start_time
        new_milestones = len(milestones) - initial_milestone_count

        struct_logger.info(
            "graph.reply_node.complete",
            session_id=session_id,
            total_duration_ms=round(total_duration * 1000, 2),
            new_milestones_achieved=new_milestones,
            total_milestones=len(milestones),
            response_length=len(state["last_bot"]),
        )

        return state

    except Exception as e:
        struct_logger.error(
            "graph.reply_node.error", session_id=session_id, error=str(e), exc_info=True
        )
        # Ensure we return a valid state even on error
        state["last_bot"] = (
            "I apologize, but I encountered an issue. Could you please try again?"
        )
        _append_transcript(state, "assistant", state["last_bot"])
        return state


# Build and compile the graph
g = StateGraph(State)
g.add_node("reply_node", reply_node_fn)
g.set_entry_point("reply_node")
graph = g.compile()


# --- API Endpoints with Enhanced Logging ---


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    struct_logger.info(
        "http.request.start",
        method=request.method,
        url=str(request.url),
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent", "unknown"),
    )

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        struct_logger.info(
            "http.request.complete",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
            client_ip=client_ip,
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        struct_logger.error(
            "http.request.error",
            method=request.method,
            url=str(request.url),
            error=str(e),
            duration_ms=round(duration * 1000, 2),
            client_ip=client_ip,
            exc_info=True,
        )
        raise


@app.post("/start")
async def start(req: StartReq):
    """Initialize a new session and return session details.

    Args:
        req (StartReq): The request containing the candidate label.

    Returns:
        dict: Session ID, initial message, and artifacts.
    """
    struct_logger.info("session.start.request", candidate_label=req.candidate_label)

    try:
        sid = new_session(req.candidate_label, SCEN["id"])

        seed = SCEN["seed_message"]
        
        # Update session with initial seed message
        initial_state = {
            "milestones": [],
            "penalty": 0.0,
            "transcript": [{"role": "assistant", "text": seed}],
            "last_bot": seed,
            "started_at": datetime.now().isoformat(),
        }
        
        update_success = update_session(sid, initial_state)
        if not update_success:
            struct_logger.warning("session.start.update_failed", session_id=sid)

        # Log the turn
        log_turn(sid, "bot", seed)

        struct_logger.info(
            "session.start.success",
            session_id=sid,
            candidate_label=req.candidate_label,
            seed_message_length=len(seed),
        )

        return {"session_id": sid, "message": seed, "artifacts": []}

    except Exception as e:
        struct_logger.error(
            "session.start.error",
            candidate_label=req.candidate_label,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to start session")


@app.post("/reply")
async def reply(msg: Msg):
    """Process user message and return assistant reply.

    Args:
        msg (Msg): The user message and session ID.

    Returns:
        dict: Assistant reply, attachments, and milestones.
    """
    struct_logger.info(
        "message.reply.request", session_id=msg.session_id, message_length=len(msg.text)
    )

    try:
        # CRITICAL FIX: Load existing session data instead of creating empty state
        existing_session = get_session(msg.session_id)
        if not existing_session:
            struct_logger.error("session.not_found", session_id=msg.session_id)
            raise HTTPException(status_code=404, detail="Session not found")

        # Create state with existing session data
        state = State({
            "session_id": msg.session_id,
            "transcript": existing_session.get("transcript", []),
            "milestones": existing_session.get("milestones", []),
            "started_at": existing_session.get("started_at", time.time()),
            "penalty": existing_session.get("penalty", 0.0),
            "last_user": msg.text,
        })

        struct_logger.debug(
            "session.loaded",
            session_id=msg.session_id,
            existing_milestones=len(state.get("milestones", [])),
            existing_transcript_length=len(state.get("transcript", []))
        )

        # Log user turn
        log_turn(msg.session_id, "user", msg.text)

        # Process through graph
        graph_start = time.time()
        out: State = await graph.ainvoke(state)
        graph_duration = time.time() - graph_start
        
        if not isinstance(out, dict):
            struct_logger.error(
                "graph.invalid_output",
                session_id=msg.session_id,
                output_type=type(out).__name__,
            )
            raise HTTPException(status_code=500, detail="Graph returned invalid state")

        # Ensure bot response exists
        if "last_bot" not in out:
            out["last_bot"] = "I'm ready when you are—what's the customer's message?"
            struct_logger.warning(
                "message.reply.fallback_response", session_id=msg.session_id
            )

        # CRITICAL FIX: Save updated session state back to storage
        try:
            # Convert started_at to string if it's a datetime or timestamp
            started_at_value = existing_session.get("started_at", time.time())
            if isinstance(started_at_value, datetime):
                started_at_str = started_at_value.isoformat()
            elif isinstance(started_at_value, (int, float)):
                started_at_str = datetime.fromtimestamp(started_at_value).isoformat()
            else:
                started_at_str = str(started_at_value)
            
            updated_session_data = {
                "transcript": out.get("transcript", []),
                "milestones": out.get("milestones", []),
                "penalty": out.get("penalty", 0.0),
                "started_at": started_at_str,
                "last_bot": out.get("last_bot", ""),
            }
            
            update_success = update_session(msg.session_id, updated_session_data)
            if not update_success:
                struct_logger.warning(
                    "session.update_failed", session_id=msg.session_id
                )
            
            struct_logger.debug(
                "session.updated",
                session_id=msg.session_id,
                milestones_count=len(out.get("milestones", [])),
                transcript_length=len(out.get("transcript", []))
            )
            
            # Also save individual milestones for tracking
            new_milestones = set(out.get("milestones", [])) - set(existing_session.get("milestones", []))
            for milestone in new_milestones:
                add_milestone(msg.session_id, milestone, len(out.get("transcript", [])))
                
        except Exception as storage_error:
            struct_logger.error(
                "session.update_failed",
                session_id=msg.session_id,
                error=str(storage_error),
                exc_info=True
            )
            # Continue processing but log the issue

        # Log bot turn
        log_turn(msg.session_id, "bot", out["last_bot"])

        # Prepare attachments
        attachments = []
        if (
            isinstance(out.get("milestones", []), list)
            and "M2_logs" in out["milestones"]
        ):
            attachments.append(SCEN["artifacts"]["har"])

        struct_logger.info(
            "message.reply.success",
            session_id=msg.session_id,
            graph_duration_ms=round(graph_duration * 1000, 2),
            milestones=out.get("milestones", []),
            attachments_count=len(attachments),
            response_length=len(out["last_bot"]),
        )

        return {
            "message": out["last_bot"],
            "attachments": attachments,
            "milestones": out["milestones"],
        }

    except HTTPException:
        raise
    except Exception as e:
        struct_logger.error(
            "message.reply.error",
            session_id=msg.session_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to process message")


@app.post("/finish/{session_id}")
async def finish(session_id: str):
    """End a session and calculate final metrics.

    Args:
        session_id (str): The session identifier.

    Returns:
        dict: Success confirmation.
    """
    struct_logger.info("session.finish.request", session_id=session_id)

    try:
        # Retrieve the session's start time from storage
        session = get_session(session_id)
        if session is None:
            struct_logger.error("session.not_found", session_id=session_id)
            raise HTTPException(status_code=404, detail="Session not found")
        
        started_at = session.get("started_at")
        if not started_at:
            struct_logger.error("session.missing_start_time", session_id=session_id)
            raise HTTPException(status_code=400, detail="Session start time not found")
        
        # Handle different timestamp formats
        if isinstance(started_at, str):
            try:
                started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            except ValueError:
                # Fallback for other string formats
                started_at = datetime.now(timezone.utc)
        elif isinstance(started_at, (int, float)):
            started_at = datetime.fromtimestamp(started_at, tz=timezone.utc)
        elif isinstance(started_at, datetime):
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
        else:
            started_at = datetime.now(timezone.utc)
        
        now = datetime.now(timezone.utc)
        elapsed = (now - started_at).total_seconds()
        end_session(session_id, elapsed)

        struct_logger.info(
            "session.finish.success", session_id=session_id, elapsed_time=elapsed
        )

        return {"ok": True}

    except HTTPException:
        raise
    except Exception as e:
        struct_logger.error(
            "session.finish.error", session_id=session_id, error=str(e), exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to end session")


@app.post("/score")
async def score(payload: Dict[str, Any]):
    """Calculate scenario score based on milestones and penalties.

    Args:
        payload (Dict[str, Any]): Milestones list and penalty float.

    Returns:
        dict: Score breakdown, total score, and pass/fail status.
    """
    struct_logger.info(
        "scoring.request",
        milestones=payload.get("milestones", []),
        penalty=payload.get("penalty", 0.0),
    )

    try:
        ms = set(payload.get("milestones", []))
        rub = SCEN["rubric"]

        # Compute base score
        scoring_start = time.time()
        res = compute_score(ms, rub)
        scoring_duration = time.time() - scoring_start

        # Apply communication penalty
        comm_pen = payload.get("penalty", 0.0)
        original_comm_score = res["by_dimension"]["communication"]
        res["by_dimension"]["communication"] = max(0.0, original_comm_score - comm_pen)

        # Calculate weighted total
        res["total"] = round(
            res["by_dimension"]["discovery"] * rub["weights"]["discovery"]
            + res["by_dimension"]["evidence"] * rub["weights"]["evidence"]
            + res["by_dimension"]["reasoning"] * rub["weights"]["reasoning"]
            + res["by_dimension"]["communication"] * rub["weights"]["communication"],
            3,
        )

        res["pass"] = res["total"] >= rub["pass_threshold"]

        struct_logger.info(
            "scoring.complete",
            milestones_achieved=list(ms),
            penalty_applied=comm_pen,
            original_communication_score=original_comm_score,
            final_communication_score=res["by_dimension"]["communication"],
            total_score=res["total"],
            passed=res["pass"],
            duration_ms=round(scoring_duration * 1000, 2),
        )

        return res

    except Exception as e:
        struct_logger.error(
            "scoring.error", payload=payload, error=str(e), exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to calculate score")


# New debug endpoints for development/testing
@app.get("/debug/session/{session_id}")
async def debug_session(session_id: str):
    """Get detailed session information for debugging."""
    if not os.getenv("DEBUG", "false").lower() == "true":
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        milestones = get_session_milestones(session_id)
        transcript = get_session_transcript(session_id)
        
        return {
            "session": session,
            "milestones": milestones,
            "transcript": transcript,
        }
    except Exception as e:
        struct_logger.error("debug.session.error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve debug info")


# Health check endpoint
@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "scenario_loaded": SCEN is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("DEBUG", "false").lower() == "true",
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        },
    )