import shutil
import sys
from abc import ABC, abstractmethod
from io import BytesIO
from itertools import chain
from threading import Lock
from typing import (
    Dict,
    Generic,
    Iterator,
    MutableMapping,
    Optional,
    Protocol,
    TypeVar,
)

import torch
from diskcache import Cache, Disk, UNKNOWN
from torch import Tensor

K = TypeVar("K")
V = TypeVar("V")


class Sizer(Protocol[V]):
    def __call__(self, value: V) -> int:
        ...


class CacheLike(MutableMapping[K, V], Generic[K, V], ABC):
    @property
    @abstractmethod
    def directory(self) -> str:
        pass

    def __enter__(self) -> "CacheLike":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class TensorDisk(Disk):
    # TODO: add compression
    # see DiskCache docs, because there is an example there

    def get(self, key, raw):
        data = super().get(key, raw)
        with BytesIO(data) as f:
            return torch.load(f)

    def store(self, value, read, key=UNKNOWN):
        if not read:
            with BytesIO() as f:
                torch.save(value, f)
                value = f.getvalue()
        return super().store(value, read)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            with BytesIO(data) as f:
                data = torch.load(f)
        return data


class TensorCache(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, disk=TensorDisk, **kwargs)


class ProgressiveCache(CacheLike[K, V], Generic[K, V]):
    _backend: Cache
    _buffer_size: int
    _sizer: Sizer[V]
    _current_size: int
    _memory_cache: Dict[K, V]
    _lock = Lock

    def __init__(
        self,
        backend: Optional[Cache] = None,
        buffer_size: int = 0,
        sizer: Sizer[V] = sys.getsizeof,
    ) -> None:
        super().__init__()
        self._backend = backend if backend is not None else Cache()
        self._buffer_size = buffer_size
        self._sizer = sizer
        self._current_size = 0
        self._memory_cache = {}
        self._lock = Lock()

    @property
    def directory(self) -> str:
        return self._backend.directory

    def flush(self) -> None:
        to_transfer = {}
        with self._lock:
            for key in self._memory_cache:
                value = self._memory_cache.pop(key)
                to_transfer[key] = value
                self._current_size -= self._sizer(value)
        for key, value in to_transfer.items():
            self._backend[key] = value

    def __setitem__(self, key: K, value: V) -> None:
        size = self._sizer(value)
        with self._lock:
            if self._current_size + size <= self._buffer_size:
                self._memory_cache[key] = value
                self._current_size += size
                return
        self._backend[key] = value

    def __delitem__(self, key: K) -> None:
        with self._lock:
            if key in self._memory_cache:
                size = self._sizer(self._memory_cache[key])
                self._memory_cache.__delitem__(key)
                self._current_size -= size
                return
        self._backend.__delitem__(key)

    def __getitem__(self, key: K) -> V:
        value = self._memory_cache.get(key, None)
        if value is not None:
            return value
        return self._backend[key]

    def __len__(self) -> int:
        return len(self._memory_cache) + len(self._backend)

    def __iter__(self) -> Iterator[K]:
        return chain(iter(self._memory_cache), iter(self._backend))

    def __enter__(self):
        return self._backend.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return self._backend.__exit__(exc_type, exc_val, exc_tb)


class ProgressiveTensorCache(ProgressiveCache[K, Tensor], Generic[K]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, sizer=self._get_tensor_size, **kwargs)

    @staticmethod
    def _get_tensor_size(tensor: Tensor) -> int:
        # TODO: better way to size tensors
        # this one reports too much because tensors can share storage
        return sys.getsizeof(tensor.storage())


class SelfCleaningCache(CacheLike[K, V], Generic[K, V]):
    def __init__(self, backend: CacheLike[K, V]) -> None:
        self._backend = backend

    @property
    def directory(self) -> str:
        return self._backend.directory

    def __setitem__(self, key: K, value: V) -> None:
        return self._backend.__setitem__(key, value)

    def __delitem__(self, key: K) -> None:
        return self._backend.__delitem__(key)

    def __getitem__(self, key: K) -> V:
        return self._backend.__getitem__(key)

    def __len__(self) -> int:
        return len(self._backend)

    def __iter__(self) -> Iterator[K]:
        return iter(self._backend)

    def __enter__(self):
        return self._backend.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        value = self._backend.__exit__(exc_type, exc_val, exc_tb)
        shutil.rmtree(self._backend.directory, ignore_errors=True)
        return value
