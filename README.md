# MLOps CI/CD Pipeline for LLMs

> Fine-tuned **DistilGPT-2** for text summarization, with a full MLOps pipeline:
> GitHub Actions CI/CD → Docker → Hugging Face Spaces

## Project Structure

```
mlops-llm-cicd/
├── .github/workflows/
│   ├── ci.yml          # Lint → Test → Docker Build  (every push)
│   └── cd.yml          # Fine-tune → HF Hub → Deploy (push to main only)
├── src/
│   ├── config.py       # Dataclass configs (hyperparams, inference settings)
│   ├── data.py         # XSum dataset loading & tokenization
│   ├── train.py        # Fine-tuning with HuggingFace Trainer
│   ├── inference.py    # SummarizationPipeline class
│   └── evaluate.py     # ROUGE metric computation
├── app/
│   └── app.py          # Gradio web interface
├── tests/              # Unit tests (all model calls mocked)
├── Dockerfile          # Multi-stage Docker build
├── requirements.txt    # Production deps
└── pyproject.toml      # Build config + ruff + pytest settings
```

## How It Works

### Model

**DistilGPT-2** is a distilled GPT-2 (~82M parameters). It's a *causal language model* — it predicts the next token given all previous ones. We frame summarization as a completion task:

```
Article: {document}
TL;DR: {summary}<|endoftext|>
```

The model learns to complete "TL;DR:" with a summary during fine-tuning. At inference, we feed the article + "TL;DR:" and let it generate.

### Dataset

**XSum** (BBC, ~200k examples). Each example is a news article paired with a 1-sentence summary. We use a 500–1000 sample subset in CI for speed.

### CI Pipeline (every push)

```
push → Lint (ruff) → Unit Tests (pytest + mocks) → Docker Build
```

### CD Pipeline (push to main)

```
merge to main → Fine-tune model → Push to HF Hub → Deploy to HF Spaces
```

## Setup

### Local development

```bash
# Clone and install
https://github.com/26vyshnavi/mlops-llm-cicd
cd mlops-llm-cicd
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest

# Fine-tune (small subset)
python -m src.train

# Run the app locally
python app/app.py
# → Open http://localhost:7860
```

### Docker

```bash
# Build
docker build -t llm-summarizer .

# Run (falls back to base distilgpt2 if no fine-tuned model)
docker run -p 7860:7860 llm-summarizer

# Run with a fine-tuned model from HF Hub
docker run -p 7860:7860 -e MODEL_PATH=your-user/distilgpt2-summarizer llm-summarizer
```

### GitHub Secrets (required for CD)

Add these at **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Your HF API token (write access) — get it at huggingface.co/settings/tokens |
| `HF_USERNAME` | Your HuggingFace username |
| `HF_SPACE_NAME` | Name for your Space (e.g. `distilgpt2-summarizer`) |

## Key Concepts Demonstrated

- **Prompt engineering for decoder-only models** — framing summarization as completion
- **HuggingFace Trainer API** — full training loop with evaluation, checkpointing, LR scheduling
- **Mocked unit tests** — fast, isolated tests that never download a real model
- **Multi-stage Docker builds** — smaller images by separating build and runtime deps
- **GitHub Actions concurrency controls** — cancel stale CI runs, serialize CD
- **HF Hub as a model registry** — push/pull model weights instead of baking them into Docker
- **Lazy model loading** — app starts fast; model loads on first request
- **ROUGE evaluation** — automatic quality measurement in the CD pipeline

## ROUGE Scores

After fine-tuning on 500 XSum samples for 1 epoch (CI demo config):

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.1798 |
| ROUGE-2 | 0.0282 |
| ROUGE-L | 0.1306 |

Full fine-tuning (200k samples, 3 epochs, GPU) would push ROUGE-2 to ~0.12–0.16.
