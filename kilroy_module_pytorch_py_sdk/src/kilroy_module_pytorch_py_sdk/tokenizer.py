from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, indices: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def start_token(self) -> int:
        pass

    @property
    @abstractmethod
    def end_token(self) -> int:
        pass
