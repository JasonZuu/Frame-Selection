import cv2
import numpy as np


from registry import Registries

from .base_scorer import BaseScorer


@Registries.scorer.register("random")
class RandomScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score_frame(self,
                    group_size: int = 1,
                    resize_shape: tuple = (64, 64),
                    sort: bool = True,
                    unitized: bool = True) -> list:
        assert self.scores is not None and self.scores == [], "please call reset first"

        frame_count = len(self.frames)
        scores = np.random.rand(frame_count)
        for idx in range(frame_count):
            data = {"index": idx,
                    "score": scores[idx]}
            self.scores.append(data)

        if sort:
            self.scores = self._sort_score(self.scores)
        if unitized:
            self.scores = self._unitize(self.scores)
        return self.scores
