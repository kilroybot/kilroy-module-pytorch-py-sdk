from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, indices: List[int]) -> str:
        pass
