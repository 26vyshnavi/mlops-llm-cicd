"""
test_config.py — Tests for configuration dataclasses.
No external dependencies — runs instantly.

FIX F401: pytest not imported — fixtures are injected by conftest automatically.
FIX I001: imports sorted alphabetically.
"""

from src.config import InferenceConfig, TrainingConfig


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.model_name == "distilgpt2"
        assert config.dataset_name == "xsum"
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.eval_split == 0.1
        assert config.hub_model_id is None

    def test_custom_override(self):
        config = TrainingConfig(batch_size=16, num_epochs=1, dataset_split_size=500)
        assert config.batch_size == 16
        assert config.num_epochs == 1
        assert config.dataset_split_size == 500

    def test_hub_model_id_can_be_set(self):
        config = TrainingConfig(hub_model_id="my-user/distilgpt2-summarizer")
        assert config.hub_model_id == "my-user/distilgpt2-summarizer"

    def test_output_dir_default(self):
        config = TrainingConfig()
        assert config.output_dir == "./model_output"

    def test_max_length_is_positive(self):
        config = TrainingConfig(max_length=512)
        assert config.max_length > 0

    def test_learning_rate_is_small(self):
        config = TrainingConfig()
        assert config.learning_rate < 1e-3

    def test_eval_split_is_fraction(self):
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
        config = InferenceConfig(temperature=0.5)
        assert config.temperature == 0.5

    def test_custom_model_path(self):
        config = InferenceConfig(model_path="my-user/distilgpt2-summarizer")
        assert config.model_path == "my-user/distilgpt2-summarizer"
