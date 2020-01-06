from gpt2web.models.manager import ModelManager
from gpt2web.models.registry import ModelRegistry


class Config:
    models_path: str = None
    model_registry: ModelRegistry = None
    model_manager: ModelManager = None


