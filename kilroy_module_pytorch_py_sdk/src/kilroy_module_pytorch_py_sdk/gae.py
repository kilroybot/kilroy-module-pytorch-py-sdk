from typing import Dict, Any, Callable, Awaitable

import torch
from pydantic import Field
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_pytorch_py_sdk.utils import unpack_to_list, pack_list
from kilroy_module_server_py_sdk import SerializableState
from kilroy_server_py_utils import Configurable, Parameter, classproperty


class State(SerializableState):
    gamma: float = 0.99
    lambda_: float = Field(0.95, alias="lambda")


class GeneralizedAdvantageEstimator(Configurable[State]):
    class GammaParameter(Parameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.99,
                "title": cls.pretty_name,
            }

    class LambdaParameter(Parameter[State, float]):
        # noinspection PyMethodParameters
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.95,
                "title": cls.pretty_name,
            }

        @classmethod
        async def _get(cls, state: State) -> float:
            return state.lambda_

        @classmethod
        async def _set(
            cls, state: State, value: float
        ) -> Callable[[], Awaitable]:
            original_value = state.lambda_

            async def undo():
                state.lambda_ = original_value

            state.lambda_ = value
            return undo

    async def calculate(
        self, rewards: PackedSequence, values: PackedSequence
    ) -> PackedSequence:
        async with self.state.read_lock() as state:
            gamma = state.gamma
            lambda_ = state.lambda_

        batch_rewards = unpack_to_list(rewards)
        batch_values = unpack_to_list(values)

        batch_advantages = []

        for rewards, values in zip(batch_rewards, batch_values):
            advantages = []
            advantage = 0

            for i in reversed(range(len(rewards))):
                reward = rewards[i]
                value = values[i]
                next_value = (
                    torch.zeros(1) if i == len(rewards) - 1 else values[i + 1]
                )

                delta = reward + gamma * next_value - value
                advantage = delta + gamma * lambda_ * advantage
                advantages.append(advantage)

            advantages = torch.stack(advantages[::-1])
            batch_advantages.append(advantages)

        return pack_list(batch_advantages)
