from dataclasses import dataclass
from typing import Dict, Any

from kilroy_module_pytorch_py_sdk import Generator
from kilroy_module_pytorch_py_sdk.models.registry import ModelsRegistry
from kilroy_module_pytorch_py_sdk.trainers import Trainer


@dataclass
class TrainerState:
    trainer: Trainer
    params: Dict[str, Dict[str, Any]]


@dataclass
class State:
    models: ModelsRegistry
    trainer: TrainerState
    generator: Generator
