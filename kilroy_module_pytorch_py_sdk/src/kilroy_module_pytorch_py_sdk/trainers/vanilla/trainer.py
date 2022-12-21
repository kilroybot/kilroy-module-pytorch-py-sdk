from abc import ABC
from functools import partial
from pathlib import Path
from typing import (
    AsyncIterable,
    Tuple,
    Collection,
    Dict,
    Any,
    Set,
    Type,
    Optional,
)

import torch
from torch import Tensor

from kilroy_module_pytorch_py_sdk.models.registry import ModelsRegistry
from kilroy_module_pytorch_py_sdk.scalers.reward import RewardScaler
from kilroy_module_pytorch_py_sdk.trainers.trainer import Trainer
from kilroy_module_pytorch_py_sdk.trainers.vanilla.parameters import (
    SupervisedParameter,
    ReinforcedParameter,
    ScalerParameter,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.params import (
    Params,
    ScalerParams,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.reinforced.trainer import (
    ReinforcedTrainer,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.state import (
    State,
    ScalerState,
)
from kilroy_module_pytorch_py_sdk.trainers.vanilla.supervised.trainer import (
    SupervisedTrainer,
)
from kilroy_module_server_py_sdk import Metric
from kilroy_server_py_utils import (
    Configurable,
    Savable,
    Parameter,
    classproperty,
)


class VanillaTrainerBase(Trainer, Configurable[State], ABC):
    @classmethod
    async def _build_supervised_trainer(
        cls, models: ModelsRegistry, params: Dict[str, Any]
    ) -> SupervisedTrainer:
        return await cls._build_generic(
            SupervisedTrainer, models=models, **params
        )

    @classmethod
    async def _build_reinforced_trainer(
        cls, models: ModelsRegistry, params: Dict[str, Any]
    ) -> ReinforcedTrainer:
        return await cls._build_generic(
            ReinforcedTrainer, models=models, **params
        )

    @classmethod
    async def _build_scaler(
        cls, params: ScalerParams
    ) -> Optional[RewardScaler]:
        if params.type is None:
            return None
        return await cls._build_generic(
            RewardScaler,
            category=params.type,
            **params.params.get(params.type, {}),
        )

    @classmethod
    async def _build_scaler_state(cls, params: ScalerParams) -> ScalerState:
        return ScalerState(
            scaler=await cls._build_scaler(params),
            params=params.params,
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        return State(
            models=self._models,
            supervised=await self._build_supervised_trainer(
                self._models, params.supervised
            ),
            reinforced=await self._build_reinforced_trainer(
                self._models, params.reinforced
            ),
            scaler=await self._build_scaler_state(params.scaler),
        )

    @staticmethod
    async def _save_supervised_trainer(
        trainer: SupervisedTrainer, directory: Path
    ) -> None:
        if isinstance(trainer, Savable):
            await trainer.save(directory)

    @staticmethod
    async def _save_reinforced_trainer(
        trainer: ReinforcedTrainer, directory: Path
    ) -> None:
        if isinstance(trainer, Savable):
            await trainer.save(directory)

    @classmethod
    async def _save_scaler_state(
        cls, state: ScalerState, directory: Path
    ) -> None:
        if isinstance(state.scaler, Savable):
            await state.scaler.save(directory / "scaler")

        state_dict = {
            "type": state.scaler.category if state.scaler else None,
            "params": state.params,
        }
        await cls._save_state_dict(state_dict, directory)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_supervised_trainer(
            state.supervised, directory / "supervised"
        )
        await cls._save_reinforced_trainer(
            state.reinforced, directory / "reinforced"
        )
        await cls._save_scaler_state(state.scaler, directory / "scaler")

    @classmethod
    async def _load_supervised_trainer(
        cls, directory: Path, models: ModelsRegistry, params: Dict[str, Any]
    ) -> SupervisedTrainer:
        return await cls._load_generic(
            directory,
            SupervisedTrainer,
            models=models,
            **params,
            default=partial(cls._build_supervised_trainer, models, params),
        )

    @classmethod
    async def _load_reinforced_trainer(
        cls, directory: Path, models: ModelsRegistry, params: Dict[str, Any]
    ) -> ReinforcedTrainer:
        return await cls._load_generic(
            directory,
            ReinforcedTrainer,
            models=models,
            **params,
            default=partial(cls._build_reinforced_trainer, models, params),
        )

    @classmethod
    async def _load_scaler(
        cls, state_dict: Dict[str, Any], params: ScalerParams, directory: Path
    ) -> Optional[RewardScaler]:
        category = state_dict.get("type", params.type)
        if category is None:
            return None
        return await cls._load_generic(
            directory / "scaler",
            RewardScaler,
            category=category,
            **state_dict.get("params", params.params).get(category, {}),
            default=partial(cls._build_scaler, params),
        )

    @classmethod
    async def _load_scaler_state(
        cls,
        directory: Path,
        params: ScalerParams,
    ) -> ScalerState:
        state_dict = await cls._load_state_dict(directory)
        return ScalerState(
            scaler=await cls._load_scaler(
                state_dict, params, directory / "scaler"
            ),
            params=state_dict.get("params", params.params),
        )

    async def _load_saved_state(self, directory: Path) -> State:
        params = Params(**self._kwargs)
        return State(
            models=self._models,
            supervised=await self._load_supervised_trainer(
                directory / "supervised", self._models, params.supervised
            ),
            reinforced=await self._load_reinforced_trainer(
                directory / "reinforced", self._models, params.reinforced
            ),
            scaler=await self._load_scaler_state(
                directory / "scaler", params.scaler
            ),
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        async with self.state.write_lock() as state:
            await state.supervised.cleanup()
            await state.reinforced.cleanup()
            if isinstance(state.scaler.scaler, Configurable):
                await state.scaler.scaler.cleanup()


class VanillaTrainer(VanillaTrainerBase):
    # noinspection PyMethodParameters
    @classproperty
    def parameters(cls) -> Set[Type[Parameter]]:
        return {
            SupervisedParameter,
            ReinforcedParameter,
            ScalerParameter,
        }

    async def load_saved(self, directory: Path) -> None:
        await self._models.policy.acquire()
        await self._models.baseline.acquire()
        return await super().load_saved(directory)

    async def init(self) -> None:
        await self._models.policy.acquire()
        await self._models.baseline.acquire()
        await super().init()

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._models.policy.release()
        await self._models.baseline.release()

    async def get_metrics(self) -> Collection[Metric]:
        async with self.state.read_lock() as state:
            sm = list(await state.supervised.get_metrics())
            rm = list(await state.reinforced.get_metrics())
            return sm + rm

    async def _transform_data(
        self,
        data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]],
    ) -> AsyncIterable[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        async for context, response, score in data:
            async with self.state.read_lock() as state:
                scaler = state.scaler.scaler
            if scaler is not None:
                reward = await scaler.scale(score.item())
                reward = torch.tensor(reward).float()
            else:
                reward = score

            yield context, response, score, reward

    async def fit_supervised(
        self, data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        async with self.state.read_lock() as state:
            supervised = state.supervised

        await supervised.fit(self._transform_data(data))

    async def fit_reinforced(
        self, data: AsyncIterable[Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        async with self.state.read_lock() as state:
            reinforced = state.reinforced

        await reinforced.fit(self._transform_data(data))
