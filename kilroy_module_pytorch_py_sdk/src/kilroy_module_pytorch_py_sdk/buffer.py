from typing import Generic, MutableMapping, TypeVar
from uuid import uuid4

from torch import Tensor, nn
from torch.autograd.graph import saved_tensors_hooks

from kilroy_module_pytorch_py_sdk.utils import SelfCleaningKey

StoreType = TypeVar("StoreType", bound=MutableMapping)
ModelType = TypeVar("ModelType", bound=nn.Module)


class TensorBuffer(Generic[StoreType]):
    _store: StoreType

    def __init__(self, store: StoreType) -> None:
        self._store = store
        self._hooks_ctx = saved_tensors_hooks(self.pack_hook, self.unpack_hook)

    @property
    def store(self) -> StoreType:
        return self._store

    def pack_hook(self, x: Tensor) -> SelfCleaningKey:
        # TODO: remove SelfCleaningKey and just return UUID int
        # on unpack_hook add this int to some set
        # and make flush() method to delete all keys from this set
        # this method should be called after backwards() is called
        key = SelfCleaningKey(uuid4().hex, self._store)
        self._store[key.key] = x.clone()
        return key

    def unpack_hook(self, x: SelfCleaningKey) -> Tensor:
        return self._store[x.key]

    def __enter__(self) -> "TensorBuffer":
        self._hooks_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return self._hooks_ctx.__exit__(exc_type, exc_value, traceback)
