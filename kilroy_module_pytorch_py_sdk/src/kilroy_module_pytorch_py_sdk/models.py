from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence


class LanguageModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: PackedSequence) -> PackedSequence:
        pass


class RewardModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: PackedSequence) -> Tensor:
        pass
