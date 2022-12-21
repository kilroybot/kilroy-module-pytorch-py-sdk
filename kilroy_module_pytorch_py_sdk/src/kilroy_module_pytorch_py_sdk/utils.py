from asyncio import Lock
from contextlib import contextmanager
from types import TracebackType
from typing import (
    AsyncIterator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    AsyncIterable,
    Generic,
    MutableMapping,
    Type,
)
from uuid import uuid4

import torch
from aiostream.aiter_utils import AsyncIteratorContext
from aiostream.stream import iterate
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from kilroy_module_pytorch_py_sdk.models.abc import SequentialModel
from kilroy_server_py_utils.utils import batchify, background

T = TypeVar("T")


def slice_sequences(sequences: Iterable[Tensor], s: slice) -> List[Tensor]:
    return [sequence[s] for sequence in sequences]


def truncate_first_element(
    sequences: Iterable[Tensor],
) -> List[Tensor]:
    return slice_sequences(sequences, slice(1, None))


def truncate_last_element(
    sequences: Iterable[Tensor],
) -> List[Tensor]:
    return slice_sequences(sequences, slice(-1))


def pad(x: Iterable[Tensor], pad_value: float = 0) -> Tuple[Tensor, List[int]]:
    x = list(x)
    return (
        pad_sequence(x, batch_first=True, padding_value=pad_value),
        [len(s) for s in x],
    )


def unpad(x: Tensor, lengths: Iterable[int]) -> List[Tensor]:
    return [s[:length] for s, length in zip(x, lengths)]


def pack_padded(
    x: Tensor, lengths: Optional[Iterable[int]] = None
) -> PackedSequence:
    lengths = (
        lengths if lengths is not None else torch.tensor([x.shape[1]] * len(x))
    )
    return pack_padded_sequence(
        x, lengths, batch_first=True, enforce_sorted=False
    )


def pack_list(x: Iterable[Tensor]) -> PackedSequence:
    return pack_padded(*pad(x))


def unpack_to_padded(
    x: PackedSequence, pad_value: float = 0
) -> Tuple[Tensor, Tensor]:
    return pad_packed_sequence(x, batch_first=True, padding_value=pad_value)


def unpack_to_list(x: PackedSequence) -> List[Tensor]:
    return unpad(*unpack_to_padded(x))


def squash_packed(x, fn):
    return PackedSequence(
        fn(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices
    )


@contextmanager
def freeze(model: nn.Module) -> AsyncIterator[nn.Module]:
    original_state = {}

    for name, param in model.named_parameters():
        original_state[name] = param.requires_grad
        param.requires_grad = False

    try:
        yield model
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = original_state[name]


async def batched_forward(
    model: SequentialModel,
    input: PackedSequence,
    batch_size: Optional[int] = None,
) -> PackedSequence:
    sequences = unpack_to_list(input)
    batches = [
        [x async for x in batch]
        async for batch in batchify(sequences, batch_size)
    ]
    inputs = [pack_list(batch) for batch in batches]
    outputs = [await background(model, input) for input in inputs]
    return pack_list([x for batch in outputs for x in unpack_to_list(batch)])


async def gather_logprobs(
    logprobs: PackedSequence,
    sequences: List[Tuple[Tensor, Tensor]],
) -> PackedSequence:
    logprobs = unpack_to_list(logprobs)
    logprobs = [
        logprobs[(len(context) - 1) : -1].gather(1, response)
        for logprobs, (context, response) in zip(logprobs, sequences)
    ]
    return pack_list(logprobs)


class CachingAsyncIterable(AsyncIterable[T], Generic[T]):
    _ctx: AsyncIteratorContext[T]
    _iterator: AsyncIterator[T]
    _cache: MutableMapping[str, T]
    _prefix: str
    _watermark: int
    _lock: Lock

    def __init__(
        self,
        iterable: AsyncIterable[T],
        cache: Optional[MutableMapping[str, T]] = None,
        prefix: Optional[str] = None,
    ):
        self._ctx = iterate(iterable).stream()
        self._cache = cache if cache is not None else {}
        self._prefix = prefix if prefix is not None else uuid4().hex
        self._watermark = 0

    def _make_key(self, i: int) -> str:
        return f"{self._prefix}-{i}"

    async def _get_at(self, i: int) -> T:
        key = self._make_key(i)
        async with self._lock:
            if key in self._cache:
                return self._cache[key]

            for _ in range(i - self._watermark + 1):
                data = await self._iterator.__anext__()
                key = self._make_key(self._watermark)
                self._cache[key] = data
                self._watermark += 1

            return self._cache[self._make_key(self._watermark - 1)]

    async def __aiter__(self) -> AsyncIterator[T]:
        i = 0
        while True:
            try:
                yield await self._get_at(i)
            except StopAsyncIteration:
                return
            i += 1

    async def __aenter__(self) -> "CachingAsyncIterable[T]":
        self._lock = Lock()
        await self._ctx.__aenter__()
        self._iterator = self._ctx.__aiter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        for i in range(self._watermark):
            key = self._make_key(i)
            del self._cache[key]
        self._watermark = 0
        await self._ctx.__aexit__(exc_type, exc, traceback)
        return None
