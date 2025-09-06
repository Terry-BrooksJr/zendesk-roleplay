import gradio as gr
import requests, os
BASE = os.environ.get("BASE_URL","http://localhost:8000")

def start_session(candidate_label):
    r = requests.post(f"{BASE}/start", json={"candidate_label": candidate_label or "anon"})
    r.raise_for_status()
    jd = r.json()
    return jd["session_id"], [(None, jd["message"])]

def chat_fn(user_msg, history, session_id):
    r = requests.post(f"{BASE}/reply", json={"session_id": session_id, "text": user_msg})
    r.raise_for_status()
    jd = r.json()
    bot = jd["message"]
    if jd.get("attachments"):
        for a in jd["attachments"]:
            bot += f"\n\n[Attachment available: {a}]"
    history = history + [(user_msg, bot)]
    return history, session_id

with gr.Blocks() as demo:
    gr.Markdown("### Zendesk-Style Roleplay (Deterministic, LangGraph)")
    session_state = gr.State("")
    name = gr.Textbox(label="Candidate Label (not stored as PII)", placeholder="e.g., Candidate-17A")
    start_btn = gr.Button("Start Scenario")
    chat = gr.ChatInterface(
        fn=lambda m, h: chat_fn(m, h, session_state.value),
        type="messages"
    )
    def on_start(label):
        sid, hist = start_session(label)
        session_state.value = sid
        return gr.update(value=hist)
    start_btn.click(on_start, inputs=name, outputs=chat.chatbot)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_api=False)