from dataclasses import dataclass, field
from typing import List, Literal, Optional, Type

import torch
from kornia.core import Device
from lightglue import SuperPoint
from torch import nn, Tensor
from typing_extensions import assert_never

from base_config import InstantiateConfig

DetectorType = Literal["SuperPoint"]


@dataclass
class ImagePyramidGridDetectorConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: ImagePyramidGridDetector)
    """Target class to instantiate."""
    num_pyramid_levels: int = 4
    """Number of pyramid levels."""
    detector: DetectorType = "SuperPoint"
    """Which type of detector to use"""
    max_num_keypoints: int = 256
    """Maximum number features to detect."""
    grid_selection_cell_H: int = 30
    """Grid selection cell height"""
    grid_selection_cell_W: int = 30
    """Grid selection cell width"""
    grid_selection_threshold: float = 1e-6
    """Grid selection threshold"""


class ImagePyramidGridDetector(nn.Module):
    r"""Module for differentiable local feature detection on image pyramid"""

    config: ImagePyramidGridDetectorConfig
    detector: SuperPoint

    def __init__(
            self,
            config: ImagePyramidGridDetectorConfig,
            device: Device,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self._setup_detector()

    def forward(
            self,
            image_pyramid: List[Tensor],
            mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """Detect keypoints on image pyramid"""
        if self.config.detector == "SuperPoint":
            features = self.detector({"image": image_pyramid[0]})
            keypoints = features["keypoints"]
            descriptors = features["descriptors"]
            scores = features["keypoint_scores"]
            image_pyramid_keypoints = self._grid_selection(
                image_pyramid,
                keypoints,
                scores,
            )
            return image_pyramid_keypoints
        else:
            assert_never(self.config.detector)

    def _setup_detector(self):
        if self.config.detector == "SuperPoint":
            self.detector = SuperPoint(
                max_num_keypoints=self.config.max_num_keypoints,
            ).eval().to(self.device)
        else:
            raise ValueError(f"Unknown detector type {self.config.detector}")

    def _grid_selection(
            self,
            image_pyramid: List[Tensor],
            original_image_keypoints: Tensor,
            original_image_responses: Tensor,
    ) -> List[Tensor]:
        image_pyramid_keypoints = []
        orig_image = image_pyramid[0]
        assert orig_image.dim() == 4
        one, c, h, w = orig_image.shape
        assert one == 1
        cell_h = self.config.grid_selection_cell_H
        cell_w = self.config.grid_selection_cell_W

        def grid_selection_single_level(level: int) -> Tensor:
            im_scale_factor = pow(2, level)
            cell_scale_factor = pow(1.414, level)
            im_h_scaled = int(h // im_scale_factor)
            im_w_scaled = int(w // im_scale_factor)
            cell_h_scaled = int(cell_h // cell_scale_factor)
            cell_w_scaled = int(cell_w // cell_scale_factor)
            num_cells_h = int(im_h_scaled // cell_h_scaled) + 1
            num_cells_w = int(im_w_scaled // cell_w_scaled) + 1

            selected_features = torch.zeros((num_cells_h, num_cells_w, 3), device=self.device)
            for keypoint, response in zip(
                    original_image_keypoints.squeeze(), original_image_responses.squeeze()
            ):
                x = keypoint[0] / im_scale_factor
                y = keypoint[1] / im_scale_factor
                cell_idx_H = int(y // cell_h_scaled)
                cell_idx_W = int(x // cell_w_scaled)
                if selected_features[cell_idx_H, cell_idx_W, 2] < response:
                    selected_features[cell_idx_H, cell_idx_W, 0] = x
                    selected_features[cell_idx_H, cell_idx_W, 1] = y
                    selected_features[cell_idx_H, cell_idx_W, 2] = response

            selected_keypoints = selected_features[
                                     (selected_features[..., 2] > self.config.grid_selection_threshold)
                                 ][..., :2]
            return torch.unique(torch.floor(selected_keypoints).int(), dim=0)

        for lvl in range(len(image_pyramid)):
            image_pyramid_keypoints.append(grid_selection_single_level(lvl))

        return image_pyramid_keypoints
