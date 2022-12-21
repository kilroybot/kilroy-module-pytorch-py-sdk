from typing import Dict, Any

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.scalers.reward import WindowRewardScaler


class ScalerParams(SerializableModel):
    type: str = WindowRewardScaler.category
    params: Dict[str, Dict[str, Any]] = {}


class Params(SerializableModel):
    supervised: Dict[str, Any] = {}
    reinforced: Dict[str, Any] = {}
    scaler: ScalerParams = ScalerParams()
