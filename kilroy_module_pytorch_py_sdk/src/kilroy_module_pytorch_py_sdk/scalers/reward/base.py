from abc import ABC, abstractmethod

from kilroy_module_server_py_sdk import Categorizable, classproperty, normalize


class RewardScaler(Categorizable, ABC):
    # noinspection PyMethodParameters
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("RewardScaler"))

    @abstractmethod
    async def scale(self, reward: float) -> float:
        pass
