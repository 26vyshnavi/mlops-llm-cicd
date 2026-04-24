"""
app.py — Gradio web interface for the fine-tuned summarization model.

WHY GRADIO?
  Gradio generates a complete web UI from a Python function in ~5 lines.
  It also integrates natively with HuggingFace Spaces (zero extra config),
  making deployment trivially simple. For production APIs, swap this for
  FastAPI — but for demos and internal tools, Gradio is perfect.

LAZY LOADING PATTERN:
  We don't load the model at import time. Instead, get_pipeline() loads
  it on the first actual request. This means:
    1. The app starts fast (important for CI health checks)
    2. Tests can import this module without triggering a 300MB download
    3. Docker can start, serve a /health endpoint, then load the model
       in the background if needed

HOW HUGGING FACE SPACES WORKS:
  When you push this repo to a HF Space:
    1. HF detects 'sdk: docker' in README.md and builds the Dockerfile
    2. The container is started with port 7860 exposed
    3. Gradio listens on 0.0.0.0:7860 — HF proxies public traffic there
    4. The Space URL (https://huggingface.co/spaces/user/space) is live
"""

import os
import gradio as gr

from src.config import InferenceConfig
from src.inference import SummarizationPipeline

# ── Lazy model loading ─────────────────────────────────────────────────────────
# _pipeline is the singleton. It's populated on the first call to get_pipeline().
_pipeline = None  # type: SummarizationPipeline | None


def get_pipeline() -> SummarizationPipeline:
    """Return the shared SummarizationPipeline, loading it if necessary."""
    global _pipeline
    if _pipeline is None:
        config = InferenceConfig(
            # MODEL_PATH env var is set in the HF Space settings or Dockerfile.
            # Falls back to local ./model_output, then to base distilgpt2.
            model_path=os.getenv("MODEL_PATH", "./model_output"),
        )
        _pipeline = SummarizationPipeline(config=config)
    return _pipeline


# ── Core function (what Gradio wraps) ─────────────────────────────────────────
def summarize(text: str, max_tokens: int) -> str:
    """
    Summarize the input text.

    This is the function Gradio calls when the user clicks 'Submit'.
    It must accept the same types as the Gradio input components.

    Args:
        text:       Raw input text from the Textbox component.
        max_tokens: Number of new tokens to generate (from the Slider).

    Returns:
        Generated summary string.
    """
    if not text or not text.strip():
        return "⚠️  Please provide some text to summarize."

    return get_pipeline().summarize(text, max_new_tokens=max_tokens)


# ── Gradio UI definition ───────────────────────────────────────────────────────
# gr.Interface is a high-level Gradio class for function → UI mapping.
# For more control (multiple tabs, state, streaming), use gr.Blocks instead.
demo = gr.Interface(
    fn=summarize,

    inputs=[
        gr.Textbox(
            lines=10,
            label="Input Article",
            placeholder="Paste a news article or any long text here...",
        ),
        gr.Slider(
            minimum=30,
            maximum=200,
            value=80,
            step=10,
            label="Max Summary Length (tokens)",
        ),
    ],

    outputs=gr.Textbox(
        label="Generated Summary",
        show_copy_button=True,
    ),

    title="📰 DistilGPT-2 Summarizer",
    description=(
        "Fine-tuned **DistilGPT-2** on the XSum dataset for single-sentence "
        "news article summarization. Built with a full MLOps CI/CD pipeline "
        "(GitHub Actions → Docker → Hugging Face Spaces)."
    ),

    # Pre-filled examples users can click to try immediately
    examples=[
        [
            (
                "Scientists at MIT have developed a new type of battery that can "
                "charge to 80% in just five minutes and lasts for over 1,000 charge "
                "cycles. The battery uses a solid-state electrolyte made from a ceramic "
                "material, replacing the liquid electrolytes used in conventional "
                "lithium-ion batteries. This eliminates the risk of fire and allows "
                "for much faster ion movement. The team believes the technology could "
                "be commercially available within three years."
            ),
            80,
        ],
        [
            (
                "The European Central Bank raised interest rates by 25 basis points "
                "on Thursday, bringing the key deposit rate to 4.0%, the highest level "
                "in the eurozone's history. ECB President Christine Lagarde said the "
                "decision was driven by persistently high core inflation, which remained "
                "at 5.3% in August despite a series of rate hikes over the past year. "
                "Markets had widely expected the move, though some economists argued "
                "the ECB was risking a deeper recession in Germany."
            ),
            80,
        ],
    ],

    # Gradio theme options: "default", "soft", "monochrome", "glass", "base"
    theme=gr.themes.Soft(),
    allow_flagging="never",   # Disable the 'Flag' button (not needed for demo)
)


if __name__ == "__main__":
    # When running locally:           python app/app.py
    # When running in Docker/Spaces:  same command, but server_name="0.0.0.0"
    #                                 so the container exposes the port correctly.
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces (required in Docker)
        server_port=int(os.getenv("PORT", 7860)),  # HF Spaces uses 7860
    )
