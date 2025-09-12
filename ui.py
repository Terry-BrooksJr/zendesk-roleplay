import os

import gradio as gr
import requests
from loguru import logger

BASE = os.environ.get("BASE_URL", "http://localhost:8000")


def start_session(candidate_label):
    """Starts a new scenario session for the given candidate label.

    This function sends a request to the backend to initialize a session and returns the session ID and initial assistant message.

    Args:
        candidate_label (str): The label identifying the candidate.

    Returns:
        tuple: A tuple containing the session ID (str) and a list with the initial assistant message (list of dict).
    """
    logger.info(f"Starting session for candidate_label={candidate_label}")
    r = requests.post(
        f"{BASE}/start", json={"candidate_label": candidate_label or "anon"}
    )
    r.raise_for_status()
    jd = r.json()
    return jd["session_id"], [{"role": "assistant", "content": jd["message"]}]


def chat_fn(user_msg, history, session_id):
    """Handles a chat message from the user and returns the assistant's response.

    This function sends the user's message to the backend, supports streaming responses, and manages error handling and session state.

    Args:
        user_msg: The latest user message, either as a dict or string.
        history: The conversation history.
        session_id: The current session ID.

    Returns:
        dict: A dictionary containing the assistant's reply and any attachments.
    """
    # Gradio ChatInterface(type="messages") passes the latest user message as a dict
    # like {"role": "user", "content": "..."}. Support both dict and plain str.
    if isinstance(user_msg, dict):
        text = user_msg.get("content", "")
    else:
        text = str(user_msg)

    # If session not started, yield an immediate assistant message
    if not session_id:
        return {
            "role": "assistant",
            "content": "Click **Start Scenario** first to initialize a session, then try your message again.",
        }

    bot = None

    # --- Attempt true streaming from the backend (HTTP chunked or SSE-like) ---
    try:
        with requests.post(
            f"{BASE}/reply",
            json={"session_id": session_id, "text": text},
            stream=True,
            timeout=300,
        ) as r:
            # If the server does not stream, requests will still allow iter_lines over one chunk.
            r.raise_for_status()

            assembled = ""
            any_stream = False
            for raw_line in r.iter_lines(decode_unicode=True):
                if raw_line is None or raw_line == "":
                    continue
                any_stream = True
                # Common SSE format: lines start with 'data:'
                line = raw_line
                if line.startswith("data:"):
                    line = line[5:].lstrip()
                # We expect the backend to send incremental tokens as plain text lines,
                # or JSON with a {"delta": "..."} or {"message": "..."}
                try:
                    import json as _json

                    obj = _json.loads(line)
                    delta = (
                        obj.get("delta")
                        or obj.get("message")
                        or obj.get("content")
                        or ""
                    )
                except Exception:
                    logger.debug(f"Non-JSON stream line: {line}")
                    delta = line

                if not isinstance(delta, str):
                    delta = str(delta)

                if delta:
                    assembled += delta

            if any_stream:
                return {"role": "assistant", "content": assembled}

            # If we got here, the server responded but didn't actually stream line-by-line.
            # Fall through to non-streaming handling below using the full body.
            try:
                jd = r.json()
                bot = jd.get("message", "")
                atts = jd.get("attachments") or []
                for a in atts:
                    bot += f"\n\n[Attachment available: {a}]"
            except Exception:
                bot = r.text

    except requests.HTTPError:
        # Surface server error content if available
        try:
            err_text = r.text  # type: ignore[name-defined]
        except Exception:
            err_text = "500 from /reply"
        return {"role": "assistant", "content": f"⚠️ Backend error: {err_text}"}
    except Exception:
        # Networking or parsing failure; fall back to one-shot request next
        bot = None

    # --- Fallback: one-shot request, then client-side token streaming ---
    if not bot:
        try:
            rr = requests.post(
                f"{BASE}/reply", json={"session_id": session_id, "text": text}
            )
            rr.raise_for_status()
            jd = rr.json()
            bot = jd.get("message", "")
            atts = jd.get("attachments") or []
            for a in atts:
                bot += f"\n\n[Attachment available: {a}]"
        except Exception as e:
            return {"role": "assistant", "content": f"⚠️ Request failed: {e}"}

    return {"role": "assistant", "content": bot}


with gr.Blocks() as demo:
    gr.Markdown("### Zendesk-Style Roleplay (Deterministic, LangGraph)")
    session_state = gr.State("")
    name = gr.Textbox(
        label="Candidate Label (not stored as PII)", placeholder="e.g., Candidate-17A"
    )
    start_btn = gr.Button("Start Scenario")
    chat = gr.ChatInterface(
        fn=lambda m, h: chat_fn(m, h, session_state.value),
        type="messages",
    )

    def on_start(label):
        """Initializes a new session and updates the chat interface with the initial message.

        This function starts a new scenario session using the provided label and updates the session state and chat history.

        Args:
            label: The candidate label used to start the session.

        Returns:
            gr.update: An update object for the Gradio chat interface with the initial assistant message.
        """
        chat.editable = True
        sid, hist = start_session(label)
        session_state.value = sid
        # hist is already a list of {role, content} dicts for messages mode
        return gr.update(value=hist)

    start_btn.click(on_start, inputs=name, outputs=chat.chatbot)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_api=False, share=False)
