from abc import ABC
from functools import partial
from pathlib import Path
from typing import (
    Collection,
    AsyncIterable,
    Set,
    Type,
    Tuple,
    Dict,
    Any,
    Optional,
)

from torch import Tensor

from kilroy_module_pytorch_py_sdk.gae import GeneralizedAdvantageEstimator
from kilroy_module_pytorch_py_sdk.models.abc import SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelLoader
from kilroy_module_pytorch_py_sdk.models.registry import ModelsRegistry
from kilroy_module_pytorch_py_sdk.optimizers import Optimizer
from kilroy_module_pytorch_py_sdk.regularizations.policy import (
    PolicyRegularization,
)
from kilroy_module_pytorch_py_sdk.schedulers import Scheduler
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.controls import (
    TrainingControls,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.methods import (
    Method,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.parameters import (
    PolicyOptimizerParameter,
    PolicySchedulerParameter,
    MethodParameter,
    ValueOptimizerParameter,
    ValueSchedulerParameter,
    RegularizationsParameter,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.params import (
    OptimizerParams,
    ModelParams,
    ModelsParams,
    Params,
    SchedulerParams,
    MethodParams,
    RegularizationsParams,
)
from kilroy_module_pytorch_py_sdk.trainers.ac.supervised.state import (
    State,
    ModelsState,
    OptimizerState,
    SchedulerState,
    ModelState,
    MethodState,
    RegularizationsState,
)
from kilroy_module_server_py_sdk import Metric, Metrizable
from kilroy_server_py_utils import (
    Configurable,
    classproperty,
    Parameter,
    Configuration,
    Savable,
)


class SupervisedTrainerBase(Metrizable, Configurable[State], ABC):
    def __init__(
        self,
        config: Configuration[State],
        models: ModelsRegistry,
        **kwargs,
    ) -> None:
        self._models = models
        super().__init__(config, **kwargs)

    @classmethod
    async def _build_optimizer(
        cls, loader: ModelLoader, params: OptimizerParams
    ) -> Optimizer:
        category = params.type
        loaded = await loader.get()
        return await cls._build_generic(
            Optimizer,
            category=category,
            parameters=loaded.model.parameters(),
            **params.params.get(category, {}),
        )

    @classmethod
    async def _build_optimizer_state(
        cls, loader: ModelLoader, params: OptimizerParams
    ) -> OptimizerState:
        return OptimizerState(
            optimizer=await cls._build_optimizer(loader, params),
            params=params.params,
        )

    @classmethod
    async def _build_scheduler(
        cls, optimizer: Optimizer, params: SchedulerParams
    ) -> Optional[Scheduler]:
        category = params.type
        if category is None:
            return None
        return await cls._build_generic(
            Scheduler,
            category=category,
            optimizer=await optimizer.get(),
            **params.params.get(category, {}),
        )

    @classmethod
    async def _build_scheduler_state(
        cls, optimizer: Optimizer, params: SchedulerParams
    ) -> SchedulerState:
        return SchedulerState(
            scheduler=await cls._build_scheduler(optimizer, params),
            params=params.params,
        )

    @classmethod
    async def _build_model_state(
        cls, loader: ModelLoader[SequentialModel], params: ModelParams
    ) -> ModelState:
        optimizer = await cls._build_optimizer_state(loader, params.optimizer)
        return ModelState(
            loader=loader,
            optimizer=optimizer,
            scheduler=await cls._build_scheduler_state(
                optimizer.optimizer, params.scheduler
            ),
            step=0,
            epoch=0,
        )

    @classmethod
    async def _build_models_state(
        cls, models: ModelsRegistry, params: ModelsParams
    ) -> ModelsState:
        return ModelsState(
            policy=await cls._build_model_state(models.policy, params.policy),
            value=await cls._build_model_state(models.value, params.value),
            baseline=models.baseline,
        )

    @classmethod
    async def _build_method(cls, params: MethodParams) -> Method:
        return await cls._build_generic(
            Method,
            category=params.type,
            **params.params.get(params.type, {}),
        )

    @classmethod
    async def _build_method_state(cls, params: MethodParams) -> MethodState:
        return MethodState(
            method=await cls._build_method(params),
            params=params.params,
        )

    @classmethod
    async def _build_regularization(
        cls, category: str, params: Dict[str, Any]
    ) -> PolicyRegularization:
        return await cls._build_generic(
            PolicyRegularization, category=category, **params
        )

    @classmethod
    async def _build_regularizations_state(
        cls, params: RegularizationsParams
    ) -> RegularizationsState:
        return RegularizationsState(
            regularizations=[
                await cls._build_regularization(
                    category, params.params.get(category, {})
                )
                for category in params.types
            ]
            if params.types is not None
            else None,
            params=params.params,
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            models=await self._build_models_state(self._models, params.models),
            method=await self._build_method_state(params.method),
            regularizations=await self._build_regularizations_state(
                params.regularizations
            ),
        )

    @classmethod
    async def _save_optimizer_state(
        cls, state: OptimizerState, directory: Path
    ) -> None:
        if isinstance(state.optimizer, Savable):
            await state.optimizer.save(directory / "optimizer")

        state_dict = {"type": state.optimizer.category, "params": state.params}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_scheduler_state(
        cls, state: SchedulerState, directory: Path
    ) -> None:
        if isinstance(state.scheduler, Savable):
            await state.scheduler.save(directory / "scheduler")

        state_dict = {
            "type": state.scheduler.category if state.scheduler else None,
            "params": state.params,
        }
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_model_state(
        cls, state: ModelState, directory: Path
    ) -> None:
        await cls._save_optimizer_state(
            state.optimizer, directory / "optimizer"
        )
        await cls._save_scheduler_state(
            state.scheduler, directory / "scheduler"
        )

        state_dict = {"step": state.step, "epoch": state.epoch}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_models_state(
        cls, state: ModelsState, directory: Path
    ) -> None:
        await cls._save_model_state(state.policy, directory / "policy")
        await cls._save_model_state(state.value, directory / "value")

    @classmethod
    async def _save_method_state(
        cls, state: MethodState, directory: Path
    ) -> None:
        if isinstance(state.method, Savable):
            await state.method.save(directory / "method")

        state_dict = {"type": state.method.category, "params": state.params}
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_regularizations_state(
        cls, state: RegularizationsState, directory: Path
    ) -> None:
        for regularization in state.regularizations or []:
            category = regularization.category
            if isinstance(regularization, Savable):
                await regularization.save(directory / category)

        state_dict = {
            "types": [
                regularization.category
                for regularization in state.regularizations
            ]
            if state.regularizations is not None
            else None,
            "params": state.params,
        }
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_models_state(state.models, directory / "models")
        await cls._save_method_state(state.method, directory / "method")
        await cls._save_regularizations_state(
            state.regularizations, directory / "regularizations"
        )

    @classmethod
    async def _load_optimizer(
        cls,
        directory: Path,
        loader: ModelLoader,
        state_dict: Dict[str, Any],
        params: OptimizerParams,
    ) -> Optimizer:
        policy = await loader.get()
        category = state_dict.get("type", params.type)
        return await cls._load_generic(
            directory,
            Optimizer,
            category=category,
            parameters=policy.model.parameters(),
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_optimizer, loader, params),
        )

    @classmethod
    async def _load_optimizer_state(
        cls, directory: Path, loader: ModelLoader, params: OptimizerParams
    ) -> OptimizerState:
        state_dict = await cls._load_state_dict(directory)
        return OptimizerState(
            optimizer=await cls._load_optimizer(
                directory / "optimizer", loader, state_dict, params
            ),
            params=state_dict.get("params", params.params),
        )

    @classmethod
    async def _load_scheduler(
        cls,
        directory: Path,
        optimizer: Optimizer,
        state_dict: Dict[str, Any],
        params: SchedulerParams,
    ) -> Optional[Scheduler]:
        category = state_dict.get("type", params.type)
        if category is None:
            return None
        return await cls._load_generic(
            directory,
            Scheduler,
            category=category,
            optimizer=await optimizer.get(),
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_scheduler, optimizer, params),
        )

    @classmethod
    async def _load_scheduler_state(
        cls, directory: Path, optimizer: Optimizer, params: SchedulerParams
    ) -> SchedulerState:
        state_dict = await cls._load_state_dict(directory)
        return SchedulerState(
            scheduler=await cls._load_scheduler(
                directory / "scheduler", optimizer, state_dict, params
            ),
            params=state_dict.get("params", params.params),
        )

    @classmethod
    async def _load_model_state(
        cls,
        directory: Path,
        loader: ModelLoader[SequentialModel],
        params: ModelParams,
    ) -> ModelState:
        state_dict = await cls._load_state_dict(directory)
        optimizer = await cls._load_optimizer_state(
            directory / "optimizer", loader, params.optimizer
        )
        return ModelState(
            loader=loader,
            optimizer=optimizer,
            scheduler=await cls._load_scheduler_state(
                directory / "scheduler", optimizer.optimizer, params.scheduler
            ),
            step=state_dict.get("step", 0),
            epoch=state_dict.get("epoch", 0),
        )

    @classmethod
    async def _load_models_state(
        cls, directory: Path, models: ModelsRegistry, params: ModelsParams
    ) -> ModelsState:
        return ModelsState(
            policy=await cls._load_model_state(
                directory / "policy", models.policy, params.policy
            ),
            value=await cls._load_model_state(
                directory / "value", models.value, params.value
            ),
            baseline=models.baseline,
        )

    @classmethod
    async def _load_method(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: MethodParams,
    ) -> Method:
        category = state_dict.get("type", params.type)
        return await cls._load_generic(
            directory,
            Method,
            category=category,
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_method, params),
        )

    @classmethod
    async def _load_method_state(
        cls, directory: Path, params: MethodParams
    ) -> MethodState:
        state_dict = await cls._load_state_dict(directory)
        return MethodState(
            method=await cls._load_method(
                directory / "method", state_dict, params
            ),
            params=state_dict.get("params", params.params),
        )

    @classmethod
    async def _load_regularization(
        cls,
        directory: Path,
        category: str,
        params: Dict[str, Any],
    ) -> PolicyRegularization:
        return await cls._load_generic(
            directory,
            PolicyRegularization,
            category=category,
            **params,
            default=partial(cls._build_regularization, category, params),
        )

    @classmethod
    async def _load_regularizations_state(
        cls, directory: Path, params: RegularizationsParams
    ) -> RegularizationsState:
        state_dict = await cls._load_state_dict(directory)
        types = state_dict.get("types", params.types)
        return RegularizationsState(
            regularizations=[
                await cls._load_regularization(
                    directory / category,
                    category,
                    state_dict.get("params", params.params).get(category, {}),
                )
                for category in types
            ]
            if types is not None
            else None,
            params=state_dict.get("params", params.params),
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        return State(
            models=await self._load_models_state(
                directory / "models", self._models, params.models
            ),
            method=await self._load_method_state(
                directory / "method", params.method
            ),
            regularizations=await self._load_regularizations_state(
                directory / "regularizations", params.regularizations
            ),
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if isinstance(state.method.method, Configurable):
                await state.method.method.cleanup()
            for regularization in state.regularizations.regularizations or []:
                if isinstance(regularization, Configurable):
                    await regularization.cleanup()
            policy = state.models.policy
            if isinstance(policy.optimizer.optimizer, Configurable):
                await policy.optimizer.optimizer.cleanup()
            if isinstance(policy.scheduler.scheduler, Configurable):
                await policy.scheduler.scheduler.cleanup()
            value = state.models.value
            if isinstance(value.optimizer.optimizer, Configurable):
                await value.optimizer.optimizer.cleanup()
            if isinstance(value.scheduler.scheduler, Configurable):
                await value.scheduler.scheduler.cleanup()
        await super().cleanup()


class SupervisedTrainer(SupervisedTrainerBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            PolicyOptimizerParameter,
            PolicySchedulerParameter,
            ValueOptimizerParameter,
            ValueSchedulerParameter,
            MethodParameter,
            RegularizationsParameter,
        }

    async def get_metrics(self) -> Collection[Metric]:
        metrics = []

        async with self.state.read_lock() as state:
            metrics.extend(await state.method.method.get_metrics())
            for regularization in state.regularizations.regularizations or []:
                metrics.extend(await regularization.get_metrics())

        return metrics

    async def fit(
        self,
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
        gae: GeneralizedAdvantageEstimator,
    ) -> None:
        async with self.state.read_lock() as state:
            policy = await state.models.policy.loader.get()
            value = await state.models.value.loader.get()
            baseline = await state.models.baseline.get()
            method = state.method.method
            regularizations = state.regularizations.regularizations

        async def __step_policy() -> int:
            async with self.state.write_lock() as state:
                await self._step_policy(state)
                return state.models.policy.step

        async def __increment_policy_epoch() -> int:
            async with self.state.write_lock() as state:
                state.models.policy.epoch += 1
                return state.models.policy.epoch

        async def __step_value() -> int:
            async with self.state.write_lock() as state:
                await self._step_value(state)
                return state.models.value.step

        async def __increment_value_epoch() -> int:
            async with self.state.write_lock() as state:
                state.models.value.epoch += 1
                return state.models.value.epoch

        policy_controls = TrainingControls(
            model=policy,
            step=__step_policy,
            increment_epoch=__increment_policy_epoch,
        )
        value_controls = TrainingControls(
            model=value,
            step=__step_value,
            increment_epoch=__increment_value_epoch,
        )

        await method.fit(
            policy_controls,
            value_controls,
            baseline,
            data,
            gae,
            regularizations or [],
        )

    @staticmethod
    async def _step_policy(state: State) -> None:
        await state.models.policy.optimizer.optimizer.step()
        if state.models.policy.scheduler.scheduler is not None:
            await state.models.policy.scheduler.scheduler.step()
        state.models.policy.step += 1

    @staticmethod
    async def _step_value(state: State) -> None:
        await state.models.value.optimizer.optimizer.step()
        if state.models.value.scheduler.scheduler is not None:
            await state.models.value.scheduler.scheduler.step()
        state.models.value.step += 1
