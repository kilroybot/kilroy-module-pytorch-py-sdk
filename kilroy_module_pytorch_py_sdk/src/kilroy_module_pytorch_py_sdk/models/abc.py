from abc import ABC, abstractmethod

from torch import nn
from torch.nn.utils.rnn import PackedSequence


class SequentialModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: PackedSequence) -> PackedSequence:
        pass
