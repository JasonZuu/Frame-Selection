import cv2
import numpy as np

from registry import Registries

from .base_score import BaseScore


@Registries.score.register("uniform")
class UniformScore(BaseScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score_frame(self,
                    video_cap: cv2.VideoCapture,
                    group_size=2,
                    transforms=None,
                    **kwargs) -> list:
        assert group_size > 1
        datas = []
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in range(frame_count):
            if idx%group_size == 0:
                data = {"index": idx,
                        "score": 1}
            else:
                data = {"index": idx,
                        "score": 0}
            datas.append(data)
        datas = self._sort_score(datas)
        datas[-1]["score"] = 0 # 防止在k=1时，正则化出现错误
        return datas
