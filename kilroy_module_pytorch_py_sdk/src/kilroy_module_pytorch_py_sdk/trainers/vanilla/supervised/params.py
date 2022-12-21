from typing import Dict, Any, Optional, List

from kilroy_module_py_shared import SerializableModel
from kilroy_module_pytorch_py_sdk.optimizers import AdamOptimizer
from kilroy_module_pytorch_py_sdk.trainers.vanilla.supervised.methods import (
    MiniBatchGradientDescentMethod,
)


class OptimizerParams(SerializableModel):
    type: str = AdamOptimizer.category
    params: Dict[str, Dict[str, Any]] = {}


class SchedulerParams(SerializableModel):
    type: Optional[str] = None
    params: Dict[str, Dict[str, Any]] = {}


class ModelParams(SerializableModel):
    optimizer: OptimizerParams = OptimizerParams()
    scheduler: SchedulerParams = SchedulerParams()


class ModelsParams(SerializableModel):
    policy: ModelParams = ModelParams()


class MethodParams(SerializableModel):
    type: str = MiniBatchGradientDescentMethod.category
    params: Dict[str, Dict[str, Any]] = {}


class RegularizationsParams(SerializableModel):
    types: Optional[List[str]] = None
    params: Dict[str, Dict[str, Any]] = {
        "departure": {
            "weight": 2.0,
            "metrics": {
                "directLoss": {
                    "name": "supervisedDepartureBatchLoss",
                    "label": "Supervised Departure Batch Loss",
                    "xAxisKey": "batch",
                    "xAxisLabel": "Batch",
                },
                "aggregatedLoss": {
                    "name": "supervisedDepartureEpochLoss",
                    "label": "Supervised Departure Epoch Loss",
                    "xAxisKey": "epoch",
                    "xAxisLabel": "Epoch",
                },
            },
        },
        "entropy": {
            "weight": 0.5,
            "metrics": {
                "directLoss": {
                    "name": "supervisedEntropyBatchLoss",
                    "label": "Supervised Entropy Batch Loss",
                    "xAxisKey": "batch",
                    "xAxisLabel": "Batch",
                },
                "aggregatedLoss": {
                    "name": "supervisedEntropyEpochLoss",
                    "label": "Supervised Entropy Epoch Loss",
                    "xAxisKey": "epoch",
                    "xAxisLabel": "Epoch",
                },
            },
        },
    }


class Params(SerializableModel):
    models: ModelsParams = ModelsParams()
    method: MethodParams = MethodParams()
    regularizations: RegularizationsParams = RegularizationsParams()
