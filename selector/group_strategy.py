import cv2
import numpy as np
from copy import deepcopy

from registry import Registries

from .base_strategy import BaseStrategy


@Registries.strategy.register("group")
class GroupStrategy(BaseStrategy):
    def __init__(self, score: object, **kwargs):
        super().__init__(score, **kwargs)

    def get_datas(self,
                  video_path: str,
                  group_size: int = 3,
                  transforms: object = None,
                  **kwargs):
        self.group_size = group_size
        self.video_cap = cv2.VideoCapture(video_path, 0)
        self.datas = self.score(video_cap=self.video_cap,
                                group_size=group_size,
                                transforms=transforms)
        return deepcopy(self.datas)
