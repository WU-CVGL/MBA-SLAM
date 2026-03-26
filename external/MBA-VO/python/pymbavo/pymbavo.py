import numpy as np
import torch
from torch import Tensor
from typing import Dict
from jaxtyping import Float

from pymbavo_lib import BATracker, Factory, Frame, ImageFP32, ImageU8, ImagePyramidU8, PinholeCamera, Vector2d


class MBAVO:
    def __init__(self, configs: Dict) -> None:
        self.patch_size = int(configs['patch_size'])
        self.half_patch_size = self.patch_size // 2
        self.pattern = self.generate_patch_pattern(self.half_patch_size)
        self.tracker = BATracker(configs, self.pattern.flatten().numpy().astype(np.int32))
        self.factory = Factory()

    @staticmethod
    def generate_patch_pattern(half_patch_size: int) -> Float[Tensor, "patch_size patch_size 2"]:
        x = torch.arange(-half_patch_size, half_patch_size + 1)
        y = torch.arange(-half_patch_size, half_patch_size + 1)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        return torch.dstack([grid_x, grid_y])
