"""
test_config.py — Tests for configuration dataclasses.

These tests have NO external dependencies — they run instantly and
verify that our config system behaves as expected.
"""

import pytest
from src.config import TrainingConfig, InferenceConfig


class TestTrainingConfig:

    def test_default_values(self):
        """Verify that default hyperparameters are sensible."""
        config = TrainingConfig()
        assert config.model_name == "distilgpt2"
        assert config.dataset_name == "xsum"
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.eval_split == 0.1
        assert config.hub_model_id is None  # Hub push is opt-in

    def test_custom_override(self):
        """Dataclass fields should accept custom values."""
        config = TrainingConfig(batch_size=16, num_epochs=1, dataset_split_size=500)
        assert config.batch_size == 16
        assert config.num_epochs == 1
        assert config.dataset_split_size == 500

    def test_hub_model_id_can_be_set(self):
        """Ensure hub_model_id can be configured for HF Hub push."""
        config = TrainingConfig(hub_model_id="my-user/distilgpt2-summarizer")
        assert config.hub_model_id == "my-user/distilgpt2-summarizer"

    def test_output_dir_default(self):
        """Output directory should default to a local path."""
        config = TrainingConfig()
        assert config.output_dir == "./model_output"

    def test_max_length_is_positive(self):
        """Sequence length must be a positive integer."""
        config = TrainingConfig(max_length=512)
        assert config.max_length > 0

    def test_learning_rate_is_small(self):
        """Fine-tuning LR should be much smaller than from-scratch LR."""
        config = TrainingConfig()
        # Typical from-scratch LR is 1e-3; fine-tuning should be < 1e-4 baseline
        assert config.learning_rate < 1e-3

    def test_eval_split_is_fraction(self):
        """eval_split should be between 0 and 1."""
        config = TrainingConfig()
        assert 0 < config.eval_split < 1


class TestInferenceConfig:

    def test_default_values(self):
        config = InferenceConfig()
        assert config.model_path == "./model_output"
        assert config.max_new_tokens == 80
        assert 0 < config.temperature <= 1.0
        assert 0 < config.top_p <= 1.0

    def test_temperature_range(self):
        """Temperature should be a positive float."""
        config = InferenceConfig(temperature=0.5)
        assert config.temperature == 0.5

    def test_custom_model_path(self):
        config = InferenceConfig(model_path="my-user/distilgpt2-summarizer")
        assert config.model_path == "my-user/distilgpt2-summarizer"
