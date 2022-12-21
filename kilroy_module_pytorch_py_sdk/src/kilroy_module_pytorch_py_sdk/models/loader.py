from abc import ABC, abstractmethod
from asyncio import Lock
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Type

from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer

ModelType = TypeVar("ModelType")
LoaderType = TypeVar("LoaderType", bound="Loader")


@dataclass
class ModelInfo(Generic[ModelType]):
    model: ModelType
    tokenizer: Tokenizer
    batch_size: int
    lock: Lock


class ModelLoader(Generic[ModelType], ABC):
    _info: Optional[ModelInfo[ModelType]]
    _references: int
    _lock: Lock

    def __init__(self, lock: Lock) -> None:
        super().__init__()
        self._info = None
        self._references = 0
        self._lock = lock

    @classmethod
    async def create(cls: Type[LoaderType], **kwargs) -> LoaderType:
        return cls(lock=Lock(), **kwargs)

    @abstractmethod
    async def _load(self) -> ModelInfo[ModelType]:
        pass

    @abstractmethod
    async def _save(self, info: ModelInfo[ModelType]) -> None:
        pass

    @abstractmethod
    async def _reset(self) -> ModelInfo[ModelType]:
        pass

    async def acquire(self) -> ModelInfo[ModelType]:
        async with self._lock:
            if self._info is None:
                self._info = await self._load()
            self._references += 1
            return self._info

    async def release(self) -> None:
        async with self._lock:
            if self._references == 0:
                return
            self._references -= 1
            if self._references == 0:
                await self._save(self._info)
                self._info = None

    async def get(self) -> ModelInfo[ModelType]:
        async with self._lock:
            if self._info is None:
                raise RuntimeError("You need to acquire the model first.")
            return self._info

    async def save(self) -> None:
        async with self._lock:
            if self._info is not None:
                await self._save(self._info)

    async def reset(self) -> None:
        async with self._lock:
            if self._info is not None:
                await self._save(self._info)
            self._info = await self._reset()

    async def __aenter__(self) -> ModelInfo[ModelType]:
        return await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
