from typing import List

from kilroy_module_py_shared import SerializableModel


class Params(SerializableModel):
    contexts: List[str] = []
    regex: str = r"^(^(?!.*\s+[a-zA-Z0-9_']*$).+$)|(^(?!.*[\.\?!]+).+$)$"
    max_length: int = 16
