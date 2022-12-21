from typing import Dict, Any

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.trainers import VanillaTrainer


class ModelsParams(SerializableModel):
    policy: Dict[str, Any] = {}
    value: Dict[str, Any] = {}


class TrainerParams(SerializableModel):
    type: str = VanillaTrainer.category
    params: Dict[str, Dict[str, Any]] = {}


class Params(SerializableModel):
    models: ModelsParams = ModelsParams()
    trainer: TrainerParams = TrainerParams()
    generator: Dict[str, Any] = {}
