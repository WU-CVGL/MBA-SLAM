from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type

import kornia.geometry as KG
from kornia.core import Device
from torch import nn, Tensor

from base_config import InstantiateConfig

from detectors import ImagePyramidGridDetectorConfig


@dataclass
class FrameConfig(InstantiateConfig):
    """Configuration for frame instantiation."""
    _target: Type = field(default_factory=lambda: Frame)
    """Target class to instantiate."""
    detector: ImagePyramidGridDetectorConfig = field(
        default_factory=lambda: ImagePyramidGridDetectorConfig(
            num_pyramid_levels=4,
            max_num_keypoints=256,
        )
    )
    """Detector configuration"""
    patch_size: int = 9
    """Size of patches"""


class Frame:
    """Image Frame"""
    config: FrameConfig
    index: int
    image: Tensor
    image_pyramid: List[Tensor]
    image_pyramid_keypoints: List[Tensor]
    depth: Optional[Tensor]
    capture_time: float
    exposure_time: Optional[float]
    detector: nn.Module
    device: Device

    def __init__(
            self,
            config: FrameConfig,
            device: Device = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        self.detector = config.detector.setup(device=device)
        self.num_pyramid_levels = self.config.detector.num_pyramid_levels

    def update(
            self,
            index: int,
            image: Tensor,
            capture_time: float,
            exposure_time: Optional[float] = None,
            depth: Optional[Tensor] = None,
    ) -> None:
        self.index = index
        self.image = image
        self.capture_time = capture_time
        self.exposure_time = exposure_time
        self.depth = depth

        self.image_pyramid = KG.transform.build_pyramid(
            self.image[None, ...],
            max_level=self.num_pyramid_levels,
        )

    def detect(self):
        assert self.exposure_time is None, "Should only detect features on sharp frames"
        self.image_pyramid_keypoints = self.detector(self.image_pyramid)
