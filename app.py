"""
app.py
------
Gradio web UI for the Commit Bug Classifier.

Run:
    python app.py

Then open http://localhost:7860 in your browser.
Requires the model to be trained first (commit_bug_classifier.pth must exist),
or set DEMO_MODE=1 to run with random weights for UI testing.
"""

import os
import torch
import tiktoken
import gradio as gr

from model   import GPTModel, CommitBugClassifier
from dataset import (INTENT_LABELS, SEVERITY_LABELS,
                     NUM_INTENT_CLASSES, NUM_SEVERITY_CLASSES)
from inference import classify_commit

# ---------------------------------------------------------------------------
# Config  (must match training)
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "vocab_size":     50257,
    "context_length": 1024,
    "drop_rate":      0.0,
    "qkv_bias":       True,
    "emb_dim":        768,
    "n_layers":       12,
    "n_heads":        12,
}

CHECKPOINT   = "commit_bug_classifier.pth"
DEMO_MODE    = os.environ.get("DEMO_MODE", "0") == "1"
MAX_SEQ_LEN  = 128   # same as training; adjust if you changed it

SEVERITY_COLORS = {
    "Critical": "#ff4444",
    "High":     "#ff8c00",
    "Medium":   "#f5c518",
    "Low":      "#4caf50",
}

EXAMPLE_COMMITS = [
    "fix memory leak in redis connection pool after long-running jobs",
    "patch SQL injection vulnerability in user authentication endpoint",
    "resolve race condition in async job queue causing duplicate processing",
    "fix 403 on OAuth token refresh for mobile clients using refresh tokens",
    "update CI/CD pipeline — missing env variable causing staging deploy crash",
    "fix frontend routing bug where back button skips login page",
    "resolve database deadlock on concurrent order inserts",
    "add missing null check causing NPE in payment processing service",
]

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

def load_model():
    model = CommitBugClassifier(
        GPTModel(BASE_CONFIG),
        num_intent_classes=NUM_INTENT_CLASSES,
        num_severity_classes=NUM_SEVERITY_CLASSES,
    )
    if not DEMO_MODE and os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        print(f"Loaded weights from {CHECKPOINT}")
    else:
        print("WARNING: running with random weights (demo mode or checkpoint not found).")
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------------------------------------------------------------------
# Prediction function called by Gradio
# ---------------------------------------------------------------------------

def predict(commit_text: str):
    if not commit_text.strip():
        return (
            "—", 0,
            gr.update(value="—", elem_classes=[]),
            *[0] * NUM_INTENT_CLASSES,
            *[0] * NUM_SEVERITY_CLASSES,
        )

    result = classify_commit(
        commit_text, model, tokenizer, device, max_length=MAX_SEQ_LEN
    )

    intent      = result["intent"]
    severity    = result["severity"]
    intent_conf = int(result["intent_confidence"] * 100)
    sev_conf    = int(result["severity_confidence"] * 100)

    # All intent probabilities for bar chart (re-run softmax for full dist)
    input_ids = tokenizer.encode(commit_text)[:MAX_SEQ_LEN]
    input_ids += [50256] * (MAX_SEQ_LEN - len(input_ids))
    tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        il, sl = model(tensor)
    intent_probs   = torch.softmax(il,  dim=-1).squeeze().tolist()
    severity_probs = torch.softmax(sl, dim=-1).squeeze().tolist()

    return (
        f"{intent}  ({intent_conf}%)",
        intent_conf,
        f"{severity}  ({sev_conf}%)",
        sev_conf,
        # full distributions for bar charts
        {INTENT_LABELS[i]:   round(intent_probs[i], 4)   for i in range(NUM_INTENT_CLASSES)},
        {SEVERITY_LABELS[i]: round(severity_probs[i], 4) for i in range(NUM_SEVERITY_CLASSES)},
    )

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
## Commit Bug Classifier

Enter a commit message and the model will predict the **bug category** and **severity**.
"""

with gr.Blocks(
    title="Commit Bug Classifier",
    theme=gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css="""
    .card { border-radius: 12px; padding: 1.5rem; background: #1e1e2e; }
    .severity-critical { color: #ff4444 !important; font-weight: 700; }
    .severity-high     { color: #ff8c00 !important; font-weight: 700; }
    .severity-medium   { color: #f5c518 !important; font-weight: 700; }
    .severity-low      { color: #4caf50 !important; font-weight: 700; }
    """,
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            commit_input = gr.Textbox(
                label="Commit Message",
                placeholder="e.g. fix memory leak in redis connection pool after long-running jobs",
                lines=4,
                max_lines=8,
            )
            with gr.Row():
                submit_btn = gr.Button("Classify", variant="primary", size="lg")
                clear_btn  = gr.Button("Clear", size="lg")

            gr.Examples(
                examples=[[e] for e in EXAMPLE_COMMITS],
                inputs=commit_input,
                label="Example commit messages",
            )

        with gr.Column(scale=1):
            intent_out   = gr.Textbox(label="Bug Category",    interactive=False, lines=1)
            intent_conf  = gr.Slider(label="Confidence", minimum=0, maximum=100,
                                     interactive=False, value=0)
            severity_out = gr.Textbox(label="Severity",        interactive=False, lines=1)
            sev_conf     = gr.Slider(label="Confidence", minimum=0, maximum=100,
                                     interactive=False, value=0)

    with gr.Row():
        intent_chart   = gr.Label(label="Intent distribution (all 16 classes)",   num_top_classes=16)
        severity_chart = gr.Label(label="Severity distribution (all 4 classes)",   num_top_classes=4)

    # Wire up buttons
    outputs = [intent_out, intent_conf, severity_out, sev_conf, intent_chart, severity_chart]
    submit_btn.click(fn=predict, inputs=commit_input, outputs=outputs)
    commit_input.submit(fn=predict, inputs=commit_input, outputs=outputs)
    clear_btn.click(fn=lambda: ("", None), outputs=[commit_input, intent_out])

   

if __name__ == "__main__":
    demo.launch(share=False)
