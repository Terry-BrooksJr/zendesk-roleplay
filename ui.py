import os

import gradio as gr
import requests
from gradio_modal import Modal
from loguru import logger


def make_download_file(session_id: str, fmt: str):
    if not session_id:
        return None
    #     url = f"{BASE}/transcript?session_id={session_id}&fmt={fmt}"
    #     r = requests.get(url, timeout=30)
    #     r.raise_for_status()
    #     suffix = ".jsonl" if fmt == "jsonl" else f".{fmt}"
    #     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    #     tmp.write(r.content)
    #     tmp.flush()
    #     tmp.close()
    #     return tmp.name

    # with gr.Row():
    #     fmt_dd = gr.Dropdown(
    #         choices=["jsonl", "json", "txt"],
    #         value="jsonl",
    #         label="Export format",
    #     )
    #     dl_btn = gr.DownloadButton(
    #         label="⬇️ Download Transcript",
    #         value=None,
    #     )

    def on_download(fmt):
        # friendly guard: no session yet
        if not session_state.value:
            return None
        return make_download_file(session_state.value, fmt)

    dl_btn.click(on_download, inputs=fmt_dd, outputs=dl_btn)


# --- Helper to resolve BASE backend URL ---
def resolve_base():
    """Resolve backend BASE URL by probing /health on common candidates.

    Prefers BASE_URL env var if reachable; otherwise falls back to localhost ports.
    """
    candidates = []
    env = os.environ.get("BASE_URL")
    if env:
        candidates.append(env.rstrip("/"))
    # Common local dev ports for FastAPI
    candidates += [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ]

    for base in candidates:
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.ok:
                logger.info(f"Using backend at {base}")
                return base
        except Exception:
            continue

    # Final fallback to default
    return candidates[0] if candidates else "http://localhost:8000"


BASE = resolve_base()


def start_session(candidate_label):
    logger.info(f"Starting session for candidate_label={candidate_label}")
    try:
        r = requests.post(
            f"{BASE}/start",
            json={"candidate_label": candidate_label or "anon"},
            timeout=30,
        )
        r.raise_for_status()
        jd = r.json()
        return jd["session_id"], [{"role": "assistant", "content": jd["message"]}]
    except requests.exceptions.ConnectionError as e:
        # Provide a clear hint about ports/base URL mismatch
        hint = (
            "Can't reach the backend. Make sure the FastAPI server is running.\n"
            f"Tried BASE={BASE}. If your server is on a different port, set BASE_URL accordingly,\n"
            "e.g. export BASE_URL=http://localhost:8000 (matches app.py default) or http://localhost:9000.\n"
            "Also confirm /health returns 200."
        )
        logger.error(f"Connection to backend failed: {e}")
        return "", [{"role": "assistant", "content": f"⚠️ {hint}"}]
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        return "", [{"role": "assistant", "content": f"⚠️ Failed to start session: {e}"}]


def chat_fn(user_msg, history, session_id):  # sourcery skip: low-code-quality
    """Handles a chat message from the user and returns the assistant's response.

    This function sends the user's message to the backend, supports streaming responses, and manages error handling and session state.

    Args:
        user_msg: The latest user message, either as a dict or string.
        history: The conversation history.
        session_id: The current session ID.

    Returns:
        dict: A dictionary containing the assistant's reply and any attachments.
    """

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
                f"{BASE}/reply",
                json={"session_id": session_id, "text": text},
                timeout=30,
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

    review_btn = gr.Button("Review Instructions")
    chat = gr.ChatInterface(
        fn=lambda m, h: chat_fn(m, h, session_state.value),
        type="messages",
    )
    gr.Markdown(
        """
    [Open Lab Environment](https://www.example.com/company-policies)
        """
    )
    with Modal(visible=True) as intro_modal:
        gr.Markdown(
            """
        ### Customer Support Roleplay Scenario
        
        You are a customer support agent for a fictional e-commerce company. Your task is to assist customers with their inquiries, resolve issues, and provide excellent service.
        ## Instructions

        1. Enter a candidate label (for your reference; not stored as PII) provided to you in the email.
        2. You will be role-playing a customer support agent responding to customer inquiries.
        3. The assistant will simulate customer messages based on the scenario.
        4. You can type your responses in the chat box and press Enter to send.
        5. The assistant will reply based on the scenario context.
        6.Prior to starting, ensure you have read the scenario description, review any the suggested pre-reading materials provided to you below.
        
        You will be provided with a lab environment to test any technical solutions you propose during the roleplay.
        
        Please note that the assistant's responses are generated based on the scenario and may not always be perfect. Use your judgment to guide the conversation.

        **Note:** If you encounter any issues, please contact HR recruiter or respond to the assessment email."""
        )
        scenario_desc_btn = gr.Button("Review Scenario Description")
    with Modal(visible=False) as scenario_modal:
        gr.Markdown(
            """
        ### Scenario Description
            
        You are a Learnosity Application Support Engineer. Your task is to assist customers from all levels of technical skill levels implement Learnosity APIs and enhance their digital assessment capabilities by serving as a trusted technical advisor and providing excellent service.
        
        **Key Responsibilities:**
        - Respond to customer inquiries via email and chat in a timely and professional manner.
        - Troubleshoot and resolve technical issues related to Learnosity APIs.
        - Provide guidance on best practices for using Learnosity products.
        - Collaborate with internal teams to escalate and resolve complex issues.
        - Document customer interactions and solutions in the support ticketing system.

        **The Customer:**
        The customer is a junior developer at a mid-sized educational technology company that uses Learnosity APIs to deliver online assessments. They have a moderate level of technical expertise but are not deeply familiar with all aspects of Learnosity's offerings. They are seeking assistance with integrating new features and resolving some technical challenges they are facing.
        
        **Assumptions:**
        - The customer may not be familiar with all technical terms; use clear and simple language.
        - The customer values prompt and accurate responses.
        - The customer may have follow-up questions or require additional clarification.
        """
        )
        relevant_btn = gr.Button("Review Pre-Reading Materials")
    with Modal(visible=False) as prereading_modal:
        gr.Markdown(
            """
        ### Suggested Pre-Reading Materials
        To help you prepare for the customer support roleplay scenario, here are some suggested pre-reading materials:
        1. [Getting Started With the Author API](https://help.learnosity.com/hc/en-us/articles/360000754958-Getting-Started-With-the-Author-API)
        2. [Release Cadence and Version Lifecycle](https://help.learnosity.com/hc/en-us/articles/360001268538-Release-Cadence-and-Version-Lifecycle)
        3. [Learnosity Releases Overview](https://help.learnosity.com/hc/en-us/articles/360000758837-Learnosity-Releases-Overview)
        4. [Author API Developer Documentation](https://help.learnosity.com/hc/en-us/articles/16458061166365-author-api-Author-API)
        

        Please review these materials to familiarize yourself with best practices in customer support. This will help you perform better during the roleplay scenario.
        """
        )
        name = gr.Textbox(
            label="Candidate Label (not stored as PII)",
            placeholder="Provided in email to you",
        )
        start_btn = gr.Button("Start Scenario")

    def on_start(label):
        """Initializes a new session and updates the chat interface with the initial message.

        This function starts a new scenario session using the provided label and updates the session state and chat history.

        Args:
            label: The candidate label used to start the session.

        Returns:
            gr.update: An update object for the Gradio chat interface with the initial assistant message.
        """
        sid, hist = start_session(label)
        session_state.value = sid
        # hist is already a list of {role, content} dicts for messages mode
        return gr.update(value=hist), Modal(visible=False)

    review_btn.click(
        lambda: (Modal(visible=True), Modal(visible=False), Modal(visible=False)),
        None,
        [intro_modal, scenario_modal, prereading_modal],
    )
    scenario_desc_btn.click(
        lambda: (Modal(visible=False), Modal(visible=True), Modal(visible=False)),
        None,
        [intro_modal, scenario_modal, prereading_modal],
    )
    relevant_btn.click(
        lambda: (Modal(visible=False), Modal(visible=False), Modal(visible=True)),
        None,
        [intro_modal, scenario_modal, prereading_modal],
    )
    start_btn.click(on_start, inputs=name, outputs=[chat.chatbot, prereading_modal])

if __name__ == "__main__":
    demo.launch(server_port=7860, show_api=False, share=False)
