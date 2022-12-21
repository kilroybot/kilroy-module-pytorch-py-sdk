from dataclasses import dataclass
from typing import List, Tuple

from torch import Tensor


@dataclass
class State:
    data: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
    max_size: int
