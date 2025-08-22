# gradio_chat_save.py
import gradio as gr
import time
import os
import json
from datetime import datetime
from uuid import uuid4
from typing import List, Dict, Optional

from openai import OpenAI

SESSIONS_DIR = "./sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

DISPLAY_TEMPLATE = '''[model: {}, generated in {:.3f}s]\n{}'''
API_KEY = os.environ["OPENAI_API_KEY"]

REASONING_MODELS = [
    "o4-mini",
    "o3",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano"
]

OPENAI_MODELS = [
    "o4-mini",
    "o3",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano"
]

def create_chat_completion(
    model: str,
    base_url: str,
    messages: List[dict],
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=API_KEY
    )
    extra_args = {}
    if reasoning_effort:
        extra_args["reasoning_effort"] = reasoning_effort
    if verbosity:
        extra_args["verbosity"] = verbosity
    
    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        **extra_args
    )
    result = completion.choices[0].message.content
    return result

def chat(
    user_text: str,
    history: List[Dict],
    developer_message: str,
    model: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
):
    # Build messages
    messages = []
    if "gpt-oss" in model and reasoning_effort:
        messages.append({"role": "system", "content": f"Reasoning: {reasoning_effort}"})
    
    if developer_message:
        messages.append({"role": "developer", "content": developer_message})
        
    for x in history:
        messages.append({"role": x['role'], "content": x['text']})
    
    messages.append({"role": "user", "content": user_text})
    
    # Configure Params
    extra_params = {}
    if model in REASONING_MODELS:
        extra_params["reasoning_effort"] = reasoning_effort
    if model in OPENAI_MODELS:
        extra_params["verbosity"] = verbosity
    
    result = create_chat_completion(
        model,
        base_url,
        messages,
        **extra_params
    )
    return result

def send_message(
    user_text: str,
    history: List[Dict],
    developer_message: str,
    model: str,
    base_url: str,
    reasoning_effort: str,
    verbosity: str
):
    """
    Called when the user clicks Send. Returns:
      - updated chat_display for gr.Chatbot (list of (user, assistant) tuples)
      - updated state (history list of dicts)
      - cleared message box (empty string)
    """
    if not user_text:
        return gr.update(), history, ""

    # append user turn
    user_entry = {
        "role": "user",
        "text": user_text,
        "time": datetime.utcnow().isoformat() + "Z"
    }
    history.append(user_entry)

    # call model and measure generation time
    start = time.time()
    assistant_text = chat(
        user_text,
        history,
        developer_message,
        model,
        base_url,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity
    )
    gen_time = time.time() - start

    assistant_entry = {
        "role": "assistant",
        "text": assistant_text,
        "time": datetime.utcnow().isoformat() + "Z",
        "gen_time_s": round(gen_time, 6),
        "model": model
    }
    history.append(assistant_entry)

    # build display pairs for gr.Chatbot: (user_text, assistant_text_with_gen_time)
    display = []
    # iterate history and pair sequential user->assistant when present
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            user_msg = history[i]["text"]
            assistant_msg = ""
            if i + 1 < len(history) and history[i+1]["role"] == "assistant":
                a = history[i+1]
                assistant_msg = DISPLAY_TEMPLATE.format(a['model'], a['gen_time_s'], a['text'])
                # assistant_msg = f"**(gen: {a['gen_time_s']:.3f}s, model: {a['model']})**\n\n{a['text']}"
                i += 2
            else:
                i += 1
            display.append((user_msg, assistant_msg))
        else:
            # stray assistant message without paired user (append with empty user)
            display.append(("", f"**(gen: {history[i]['gen_time_s']:.3f}s)**\n\n{history[i]['text']}"))
            i += 1

    return display, history, ""

def clear_session(history: List[Dict]):
    return [], [], "History already empty."


def save_session(history: List[Dict]) -> str:
    """
    Saves the session history to a JSON file under ./sessions and returns the filepath (or error).
    The saved JSON contains the list of turns (dicts) including model and gen_time for assistant turns.
    """
    if not history:
        return "Nothing to save (empty session)."

    filename = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + str(uuid4())[:8] + ".json"
    filepath = os.path.join(SESSIONS_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"saved_at": datetime.utcnow().isoformat() + "Z", "history": history}, f, indent=2, ensure_ascii=False)
        return f"Saved session → {filepath}"
    except Exception as e:
        return f"Failed to save session: {e}"


with gr.Blocks(title="Chat") as demo:
    gr.Markdown("## Chat\nChoose a model (or type a custom model id), chat, and click **Save session** to write JSON to `./sessions/`.")
    with gr.Row():
        with gr.Column(scale=1):
            # Model Params
            gr.Markdown("### Model parameters")
            model = gr.Textbox(label="Model ID", placeholder="e.g. gpt-5", value="gpt-5-nano")
            base_url = gr.Textbox(label="Base URL (optional)", placeholder="e.g. https://api.openai.com/v1", value="https://api.openai.com/v1")
        
            reasoning_effort = gr.Dropdown(label="Reasoning effort", choices=["", "minimal", "low", "medium", "high"], value="")
            verbosity = gr.Dropdown(label="Verbosity", choices=["", "low", "medium", "high"], value="")

            gr.Markdown("### Developer Message")
            developer_message = gr.Textbox(
                lines=6,
                label="Developer message",
                placeholder="Set developer instructions.",
                value=""
            )
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", resizable=True)
            txt = gr.Textbox(placeholder="Type a message and press Send", show_label=False)
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear session")
            save_btn = gr.Button("Save session")
            
            status = gr.Textbox(label="Status", interactive=False)

    # state holds the raw history (list of dicts)
    state = gr.State([])

    # wire up actions
    send_btn.click(
        fn=send_message,
        inputs=[
            txt,
            state,
            developer_message,
            model,
            base_url,
            reasoning_effort,
            verbosity,
        ],
        outputs=[chatbot, state, txt]
    )
    # allow pressing Enter to send (submit)
    txt.submit(
        fn=send_message,
        inputs=[
            txt,
            state,
            developer_message,
            model,
            base_url,
            reasoning_effort,
            verbosity,
        ],
        outputs=[chatbot, state, txt]
    )
    clear_btn.click(fn=clear_session, inputs=[state], outputs=[chatbot, state, status])
    save_btn.click(fn=save_session, inputs=[state], outputs=[status])

if __name__ == "__main__":
    demo.launch()