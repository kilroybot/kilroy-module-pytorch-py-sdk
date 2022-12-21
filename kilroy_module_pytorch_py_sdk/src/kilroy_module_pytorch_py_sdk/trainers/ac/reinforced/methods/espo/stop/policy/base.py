from abc import ABC, abstractmethod

from torch.nn.utils.rnn import PackedSequence

from kilroy_server_py_utils import Categorizable, classproperty, normalize


class PolicyStopCondition(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("PolicyStopCondition"))

    @abstractmethod
    async def should_stop(
        self,
        old_logprobs: PackedSequence,
        old_gathered_logprobs: PackedSequence,
        new_logprobs: PackedSequence,
        new_gathered_logprobs: PackedSequence,
        iteration: int,
    ) -> bool:
        pass
