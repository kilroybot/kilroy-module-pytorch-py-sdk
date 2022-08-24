import json
from typing import Any, Dict, Optional

import torch
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    TextData,
    TextOnlyPost,
    background,
    classproperty,
)
from torch import Tensor

from kilroy_module_pytorch_py_sdk.tokenizer import Tokenizer


class State(SerializableState):
    max_characters: Optional[int] = None


class Codec(Configurable[State]):
    class MaxCharactersParameter(Parameter[State, Optional[int]]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": ["integer", "null"], "minimum": 0}

    async def encode(
        self, tokenizer: Tokenizer, sequence: Tensor
    ) -> Dict[str, Any]:
        indices = sequence.flatten().tolist()

        text = await background(tokenizer.decode, indices)

        async with self.state.read_lock() as state:
            text = text[: state.max_characters]

        post = TextOnlyPost(text=TextData(content=text))
        return json.loads(post.json())

    async def decode(
        self, tokenizer: Tokenizer, post: Dict[str, Any]
    ) -> Tensor:
        post = TextOnlyPost.parse_obj(post)
        text = post.text.content

        async with self.state.read_lock() as state:
            text = text[: state.max_characters]

        indices = await background(tokenizer.encode, text)
        return torch.tensor(indices).view(-1, 1)
