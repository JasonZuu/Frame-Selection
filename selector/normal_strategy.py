import cv2
import numpy as np
from copy import deepcopy

from registry import Registries

from .base_strategy import BaseStrategy


@Registries.strategy.register("normal")
class NormalStrategy(BaseStrategy):
    def __init__(self, score: object, **kwargs):
        super().__init__(score, **kwargs)

    def get_datas(self,
                  video_path: str,
                  transforms: object = None,
                  **kwargs):
        self.group_size = 1
        self.video_cap = cv2.VideoCapture(video_path, 0)
        self.datas = self.score(video_cap=self.video_cap,
                                transforms=transforms)
        return deepcopy(self.datas)  
