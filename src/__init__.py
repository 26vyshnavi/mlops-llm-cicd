# src package
# Note: heavy imports (transformers, torch) are NOT imported here.
# Import them directly in the modules that need them to keep the package
# lightweight at collection time (faster pytest startup, smaller imports).
from src.config import TrainingConfig, InferenceConfig

__all__ = ["TrainingConfig", "InferenceConfig"]
