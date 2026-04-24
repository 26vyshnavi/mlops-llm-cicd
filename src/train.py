"""
train.py — Fine-tuning DistilGPT-2 on XSum for summarization.
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
    """Fine-tune DistilGPT-2 on XSum summarization data."""
    if config is None:
        config = TrainingConfig()

    print("=" * 60)
    print(f"Model : {config.model_name}")
    print(f"Data  : {config.dataset_name} ({config.dataset_split_size} samples)")
    print(f"Epochs: {config.num_epochs} | LR: {config.learning_rate} | BS: {config.batch_size}")
    print("=" * 60)

    dataset, tokenizer = load_and_preprocess(config)

    print(f"\nLoading model '{config.model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # mlm=False → Causal LM objective (GPT-style), not Masked LM (BERT-style)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )

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

    print(f"\nSaving fine-tuned model to '{config.output_dir}'...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if config.hub_model_id and os.getenv("HF_TOKEN"):
        print(f"\nPushing model to HF Hub: '{config.hub_model_id}'...")
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])
        trainer.push_to_hub(config.hub_model_id)
        tokenizer.push_to_hub(config.hub_model_id)
        print(f"Model available at: https://huggingface.co/{config.hub_model_id}")

    print("\nTraining complete!")
    return model, tokenizer


if __name__ == "__main__":
    train()
