from typing import Tuple, Dict, Any, Optional, List

from kilroy_module_server_py_sdk import StandardMetric
from kilroy_server_py_utils import Observable


class LineMetric(StandardMetric):
    def __init__(
        self,
        observable: Observable[Tuple[int, Dict[str, Any]]],
        name: str,
        label: str,
        x_axis_key: str,
        x_axis_label: str,
        y_axis_key: str,
        y_axis_label: str,
        tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            observable,
            name,
            label,
            "line",
            x_axis_key,
            x_axis_label,
            y_axis_key,
            y_axis_label,
            tags,
        )


class LossMetric(LineMetric):
    def __init__(
        self,
        observable: Observable[Tuple[int, Dict[str, Any]]],
        name: str,
        label: str,
        x_axis_key: str,
        x_axis_label: str,
        y_axis_key: str = "loss",
        y_axis_label: str = "Loss",
        tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            observable,
            name,
            label,
            x_axis_key,
            x_axis_label,
            y_axis_key,
            y_axis_label,
            tags,
        )


class ScoreMetric(LineMetric):
    def __init__(
        self,
        observable: Observable[Tuple[int, Dict[str, Any]]],
        name: str,
        label: str,
        x_axis_key: str,
        x_axis_label: str,
        y_axis_key: str = "score",
        y_axis_label: str = "Score",
        tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            observable,
            name,
            label,
            x_axis_key,
            x_axis_label,
            y_axis_key,
            y_axis_label,
            tags,
        )
