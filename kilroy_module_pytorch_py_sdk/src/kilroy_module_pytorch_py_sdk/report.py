from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor

from kilroy_module_server_py_sdk import StandardMetric


@dataclass
class Metrics:
    direct: Optional[StandardMetric] = None
    aggregated: Optional[StandardMetric] = None


@dataclass
class MetricReport:
    values: Tensor
    metrics: Metrics = field(default_factory=Metrics)
