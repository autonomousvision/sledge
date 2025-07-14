# TODO: v0.2 add doctrings
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class GraphConfig:

    frame: Tuple[float, float]
    pose_interval: float


@dataclass
class Graph(AbstractModelFeature):
    """ """

    lines: FeatureDataType  # (lines, points, 3)
    line_adjacency: FeatureDataType  # (lines, lines)
    mask: FeatureDataType  # (lines, points)

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""

    @cached_property
    def is_mask(self) -> bool:
        """Inherited, see superclass."""
        return len(self.lines) > 0 and self.lines.shape[-1] == 3 and len(self.lines) == len(self.mask)

    def to_device(self, device: torch.device) -> Graph:
        """Implemented. See interface."""
        validate_type(self.lines, torch.Tensor)
        validate_type(self.line_adjacency, torch.Tensor)
        validate_type(self.mask, torch.Tensor)
        return Graph(
            lines=self.lines.to(device=device),
            line_adjacency=self.line_adjacency.to(device=device),
            mask=self.mask.to(device=device),
        )

    def to_feature_tensor(self) -> Graph:
        """Inherited, see superclass."""
        return Graph(
            lines=to_tensor(self.lines),
            line_adjacency=to_tensor(self.line_adjacency),
            mask=to_tensor(self.mask),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Graph:
        """Implemented. See interface."""
        return Graph(lines=data["lines"], line_adjacency=data["line_adjacency"], mask=data["mask"])

    def unpack(self) -> List[Graph]:
        """Implemented. See interface."""
        return [
            Graph(lines, line_adjacency, mask)
            for lines, line_adjacency, mask in zip(self.lines, self.line_adjacency, self.mask)
        ]

    @classmethod
    def collate(cls, batch: List[Graph]) -> Graph:
        batch_size = len(batch)
        max_lines = max([graph.lines.shape[0] for graph in batch])
        max_points = max([graph.lines.shape[1] for graph in batch])

        collate_lines = np.zeros((batch_size, max_lines, max_points, 3), dtype=np.float32)
        collate_line_adjacency = np.zeros((batch_size, max_lines, max_lines), dtype=np.float32)
        collate_mask = np.zeros((batch_size, max_lines, max_points), dtype=bool)

        for batch_idx, graph in enumerate(batch):
            num_lines = graph.lines.shape[0]
            num_points = graph.lines.shape[1]

            collate_lines[batch_idx, :num_lines, :num_points] = graph.lines
            collate_line_adjacency[batch_idx, :num_lines, :num_lines] = graph.line_adjacency

            for line_idx, line_mask in enumerate(graph.mask):
                collate_mask[batch_idx, line_idx, :num_points] = line_mask

        return Graph(collate_lines, collate_line_adjacency, collate_mask)
