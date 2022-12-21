from dataclasses import dataclass
from typing import List, Pattern


@dataclass
class State:
    contexts: List[str]
    regex: Pattern[str]
    max_length: int
