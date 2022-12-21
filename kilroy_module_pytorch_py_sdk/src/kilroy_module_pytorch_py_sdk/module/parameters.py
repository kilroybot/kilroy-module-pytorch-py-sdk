from typing import Dict, Any, Type

from kilroy_module_pytorch_py_sdk.generator import Generator
from kilroy_module_pytorch_py_sdk.module.state import State
from kilroy_module_pytorch_py_sdk.trainers import Trainer
from kilroy_module_pytorch_py_sdk.trainers import VanillaTrainer
from kilroy_server_py_utils import (
    NestedParameter,
    CategorizableBasedParameter,
    classproperty,
)


class TrainerParameter(CategorizableBasedParameter[State, Trainer]):
    @classmethod
    async def _get_categorizable(cls, state: State) -> Trainer:
        return state.trainer.trainer

    @classmethod
    async def _set_categorizable(cls, state: State, value: Trainer) -> None:
        state.trainer.trainer = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return {
            "models": state.models,
            **state.trainer.params.get(category, {}),
        }

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[Trainer]:
        return VanillaTrainer


class GeneratorParameter(NestedParameter[State, Generator]):
    pass
