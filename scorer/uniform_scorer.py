import cv2
import numpy as np

from registry import Registries

from .base_scorer import BaseScorer


@Registries.scorer.register("uniform")
class UniformScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score_frame(self,
                    group_size: int = 1,
                    resize_shape: tuple = (64, 64),
                    sort: bool = True,
                    unitized: bool = True) -> list:
        assert self.scores is not None and self.scores == [], "please call reset first"
        assert group_size > 1, f"the group size for {self.__class__.__name__} should be greater than 1"

        frame_count = len(self.frames)
        for idx in range(frame_count):
            if idx % group_size == 0:
                data = {"index": idx,
                        "score": 1}
            else:
                data = {"index": idx,
                        "score": 0}
            self.scores.append(data)

        if sort:
            self.scores = self._sort_score(self.scores)
        if unitized:
            self.scores = self._unitize(self.scores)
        return self.scores
