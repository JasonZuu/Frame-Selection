import cv2
import numpy as np
from skimage.metrics import structural_similarity

from registry import Registries

from .base_scorer import BaseScorer


@Registries.scorer.register("ssim")
class SSIMScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_for_one_group(self, frame_group: list) -> list:
        ref_img = frame_group[0]
        scores = []
        for i_frame in range(len(frame_group)):
            score = 1.0 - structural_similarity(ref_img, frame_group[i_frame])
            scores.append(score)
        return scores

    def score_frame(self,
                    group_size: int = 1,
                    resize_shape: tuple = (64, 64),
                    sort: bool = True,
                    unitized: bool = True) -> list:
        assert self.scores is not None and self.scores == [], "please call reset first"
        assert group_size > 1, f"the group size for {self.__class__.__name__} should be greater than 1"

        scores = []
        frame_count = len(self.frames)
        for i_frame in range(frame_count):
            frame = self.frames[i_frame]
            frame = cv2.cvtColor(self.frames[i_frame], cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, resize_shape)
            
            if i_frame == 0:
                group = []
            elif i_frame % group_size == 0:
                ssim_scores = self._score_for_one_group(group)
                scores.extend(ssim_scores)
                group = []
            group.append(frame)
        self.scores = [{"index": i_frame,
                  "score": scores[i_frame]} for i_frame in range(len(scores))]
        
        if sort:
            self.scores = self._sort_score(self.scores)
        if unitized:
            self.scores = self._unitize(self.scores)
        return self.scores
