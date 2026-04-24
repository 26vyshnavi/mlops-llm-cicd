"""
train.py — Fine-tuning DistilGPT-2 on XSum for summarization.

WHAT IS FINE-TUNING?
  A pre-trained model like DistilGPT-2 already "knows" English — it has learned
  grammar, facts, and writing style from a massive internet corpus. Fine-tuning
  means we continue training on a much smaller, task-specific dataset (XSum) so
  the model specialises in generating summaries.

  We're NOT training from scratch. We take the existing weights and nudge them
  with a small learning rate (2e-5 vs ~1e-3 for from-scratch training) so we
  don't "forget" what the model already knows (catastrophic forgetting).

THE TRAINER API:
  HuggingFace's Trainer handles the entire training loop:
    - Forward pass, loss computation, backward pass, optimizer step
    - Learning rate scheduling
    - Gradient accumulation
    - Evaluation every N steps/epochs
    - Model checkpointing
    - Logging to TensorBoard / WandB / etc.

  This lets us focus on the ML decisions, not boilerplate.
"""

import os
from typing import Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import TrainingConfig
from src.data import load_and_preprocess


def train(config: Optional[TrainingConfig] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Fine-tune DistilGPT-2 on XSum summarization data.

    Steps:
      1. Load + preprocess the dataset (from data.py)
      2. Load the pre-trained model
      3. Configure the DataCollator
      4. Set up TrainingArguments
      5. Run Trainer.train()
      6. Save model + tokenizer to disk
      7. Optionally push to HuggingFace Hub

    Args:
        config: TrainingConfig dataclass. Uses defaults if None.

    Returns:
        Tuple of (fine-tuned model, tokenizer)
    """
    if config is None:
        config = TrainingConfig()

    print("=" * 60)
    print(f"Model : {config.model_name}")
    print(f"Data  : {config.dataset_name} ({config.dataset_split_size} samples)")
    print(f"Epochs: {config.num_epochs} | LR: {config.learning_rate} | BS: {config.batch_size}")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    dataset, tokenizer = load_and_preprocess(config)

    # ── Step 2: Load model ─────────────────────────────────────────────────────
    # AutoModelForCausalLM automatically picks the right architecture for
    # "distilgpt2" (GPT2LMHeadModel). The weights are downloaded from HF Hub
    # and cached locally on first run.
    print(f"\nLoading model '{config.model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Resize token embeddings if we added tokens (we didn't here, but good practice)
    model.resize_token_embeddings(len(tokenizer))

    # ── Step 3: Data collator ──────────────────────────────────────────────────
    # DataCollatorForLanguageModeling does two things:
    #   1. Pads batches to the same length within each batch
    #   2. For CLM (mlm=False): shifts labels left by 1 position so position i
    #      predicts token i+1 (the standard causal LM objective)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # mlm=False → Causal LM (GPT-style), not Masked LM (BERT-style)
    )

    # ── Step 4: Training arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.output_dir,

        # Training schedule
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,

        # Learning rate warmup: linearly increase LR for the first 5% of steps
        # then follow a cosine decay. Helps avoid large gradient updates early on.
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        # Evaluation + checkpointing
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",        # Save checkpoint after each epoch
        load_best_model_at_end=True,  # Reload the best checkpoint when done
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=50,
        report_to="none",   # Disable WandB/TensorBoard in CI (set to "wandb" locally)

        # Performance
        # fp16=True,        # Uncomment if using a CUDA GPU (speeds up 2x)
        dataloader_num_workers=0,  # 0 is safer in containerised environments
    )

    # ── Step 5: Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Step 6: Save locally ──────────────────────────────────────────────────
    # Save BOTH model weights and tokenizer. Inference code will load from here.
    print(f"\nSaving fine-tuned model to '{config.output_dir}'...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # ── Step 7: Push to HuggingFace Hub (optional) ────────────────────────────
    # The HF_TOKEN env var must be set (configured as a GitHub secret in CI).
    # Push creates/updates a public model repo at hub.com/{hub_model_id}.
    if config.hub_model_id and os.getenv("HF_TOKEN"):
        print(f"\nPushing model to HF Hub: '{config.hub_model_id}'...")
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        trainer.push_to_hub(config.hub_model_id)
        tokenizer.push_to_hub(config.hub_model_id)
        print(f"Model available at: https://huggingface.co/{config.hub_model_id}")
    else:
        print("\n[Info] Skipping HF Hub push (HF_TOKEN not set or hub_model_id not configured).")

    print("\nTraining complete!")
    return model, tokenizer


if __name__ == "__main__":
    # Run: python -m src.train
    # Override defaults by passing a custom config, e.g.:
    #   config = TrainingConfig(num_epochs=1, dataset_split_size=500)
    #   train(config)
    train()
