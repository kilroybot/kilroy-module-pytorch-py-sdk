from dataclasses import dataclass

from kilroy_module_pytorch_py_sdk import ModelLoader, SequentialModel


@dataclass
class ModelsRegistry:
    policy: ModelLoader[SequentialModel]
    value: ModelLoader[SequentialModel]
    baseline: ModelLoader[SequentialModel]
