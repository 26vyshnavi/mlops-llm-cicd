"""
app.py — Gradio web interface for the fine-tuned summarization model.
"""

# FIX I001: stdlib imports first, then third-party, then local
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr

from src.config import InferenceConfig
from src.inference import SummarizationPipeline

# Lazy singleton — loaded on first request, not at import time.
# This keeps tests fast and lets Docker start before the model loads.
_pipeline = None  # type: SummarizationPipeline | None


def get_pipeline() -> SummarizationPipeline:
    """Return the shared pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        config = InferenceConfig(
            model_path=os.getenv("MODEL_PATH", "./model_output"),
        )
        _pipeline = SummarizationPipeline(config=config)
    return _pipeline


def summarize(text: str, max_tokens: int) -> str:
    """Called by Gradio when the user clicks Submit."""
    if not text or not text.strip():
        return "⚠️  Please provide some text to summarize."
    return get_pipeline().summarize(text, max_new_tokens=max_tokens)


demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(
            lines=10,
            label="Input Article",
            placeholder="Paste a news article or any long text here...",
        ),
        gr.Slider(minimum=30, maximum=200, value=80, step=10, label="Max Summary Length (tokens)"),
    ],
    outputs=gr.Textbox(label="Generated Summary"),
    title="📰 DistilGPT-2 Summarizer",
    description=(
        "Fine-tuned **DistilGPT-2** on XSum for single-sentence news summarization. "
        "Built with a full MLOps CI/CD pipeline (GitHub Actions → Docker → HF Spaces)."
    ),
    examples=[
        [
            (
                "Scientists at MIT have developed a new battery that charges to 80% in five "
                "minutes and lasts over 1,000 cycles. The solid-state electrolyte eliminates "
                "fire risk and allows faster ion movement. The team expects commercial "
                "availability within three years."
            ),
            80,
        ],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )
