import cv2
import numpy as np

from registry import Registries

from .base_score import BaseScore


@Registries.score.register("random")
class RandomScore(BaseScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(1027)

    def score_frame(self,
                    video_cap: cv2.VideoCapture,
                    group_size=None,
                    transforms=None,
                    **kwargs) -> list:
        assert group_size is None ,f"random score can not be called by group strategy"
        datas = []
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scores = np.random.rand(frame_count)
        for idx in range(frame_count):
            data = {"index": idx,
                    "score": scores[idx]}
            datas.append(data)
        datas = self._sort_score(datas)
        return datas
