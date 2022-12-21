from typing import Dict, Any, Optional, MutableSequence, Type

from kilroy_module_pytorch_py_sdk.optimizers import Optimizer, AdamOptimizer
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.schedulers import Scheduler
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.methods import (
    Method,
    MiniBatchGradientDescentMethod,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.state import (
    State,
)
from kilroy_server_py_utils import (
    CategorizableBasedParameter,
    CategorizableBasedOptionalParameter,
    MultipleCategorizableBasedOptionalParameter,
    classproperty,
)


class PolicyOptimizerParameter(CategorizableBasedParameter[State, Optimizer]):
    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        policy = await state.models.policy.loader.get()
        return {
            "parameters": policy.model.parameters(),
            **state.models.policy.optimizer.params.get(category, {}),
        }

    @classmethod
    async def _get_categorizable(cls, state: State) -> Optimizer:
        return state.models.policy.optimizer.optimizer

    @classmethod
    async def _set_categorizable(cls, state: State, value: Optimizer) -> None:
        state.models.policy.optimizer.optimizer = value
        if state.models.policy.scheduler.scheduler is not None:
            optimizer = await value.get()
            await state.models.policy.scheduler.scheduler.change_optimizer(
                optimizer
            )

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[Optimizer]:
        return AdamOptimizer


class PolicySchedulerParameter(
    CategorizableBasedOptionalParameter[State, Scheduler]
):
    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return {
            "optimizer": await state.models.policy.optimizer.optimizer.get(),
            **state.models.policy.scheduler.params.get(category, {}),
        }

    @classmethod
    async def _get_categorizable(cls, state: State) -> Optional[Scheduler]:
        return state.models.policy.scheduler.scheduler

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: Optional[Scheduler]
    ) -> None:
        state.models.policy.scheduler.scheduler = value


class ValueOptimizerParameter(CategorizableBasedParameter[State, Optimizer]):
    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        value = await state.models.value.loader.get()
        return {
            "parameters": value.model.parameters(),
            **state.models.value.optimizer.params.get(category, {}),
        }

    @classmethod
    async def _get_categorizable(cls, state: State) -> Optimizer:
        return state.models.value.optimizer.optimizer

    @classmethod
    async def _set_categorizable(cls, state: State, value: Optimizer) -> None:
        state.models.value.optimizer.optimizer = value
        if state.models.value.scheduler.scheduler is not None:
            optimizer = await value.get()
            await state.models.value.scheduler.scheduler.change_optimizer(
                optimizer
            )

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[Optimizer]:
        return AdamOptimizer


class ValueSchedulerParameter(
    CategorizableBasedOptionalParameter[State, Scheduler]
):
    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return {
            "optimizer": await state.models.value.optimizer.optimizer.get(),
            **state.models.value.scheduler.params.get(category, {}),
        }

    @classmethod
    async def _get_categorizable(cls, state: State) -> Optional[Scheduler]:
        return state.models.value.scheduler.scheduler

    @classmethod
    async def _set_categorizable(
        cls, state: State, value: Optional[Scheduler]
    ) -> None:
        state.models.value.scheduler.scheduler = value


class MethodParameter(CategorizableBasedParameter[State, Method]):
    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.method.params.get(category, {})

    @classmethod
    async def _get_categorizable(cls, state: State) -> Method:
        return state.method.method

    @classmethod
    async def _set_categorizable(cls, state: State, value: Method) -> None:
        state.method.method = value

    # noinspection PyMethodParameters
    @classproperty
    def default_categorizable(cls) -> Type[Method]:
        return MiniBatchGradientDescentMethod


class RegularizationsParameter(
    MultipleCategorizableBasedOptionalParameter[State, PolicyRegularization]
):
    @classmethod
    async def _get_categorizables(
        cls, state: State
    ) -> Optional[MutableSequence[PolicyRegularization]]:
        return state.regularizations.regularizations

    @classmethod
    async def _set_categorizables(
        cls,
        state: State,
        value: Optional[MutableSequence[PolicyRegularization]],
    ) -> None:
        state.regularizations.regularizations = value

    @classmethod
    async def _get_params(cls, state: State, category: str) -> Dict[str, Any]:
        return state.regularizations.params.get(category, {})
