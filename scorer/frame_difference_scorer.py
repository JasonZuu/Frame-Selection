from subprocess import call
import cv2
import numpy as np

from registry import Registries

from .base_scorer import BaseScorer


@Registries.scorer.register("fd")
class FrameDifferenceScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score(self, resize_shape) -> list:
        frame_count = len(self.frames)
        datas = []

        for idx in range(frame_count):
            if idx == 0:
                data = {"index": idx,
                        "score": 255.0}
            else:
                frame_last = cv2.resize(self.frames[idx-1], resize_shape)
                frame = cv2.resize(self.frames[idx], resize_shape)
                score = np.abs(frame_last-frame).mean()
                data = {"index": idx,
                        "score": score}
            datas.append(data)

        return datas

    def _group_score(self, group_size, resize_shape) -> list:
        frame_count = len(self.frames)
        scores = []

        for i_frame in range(frame_count):
            if i_frame == 0:
                idx_group = []
            elif i_frame % group_size == 0:
                group_scores = self._group_score_function(idx_group, resize_shape)
                scores.extend(group_scores)
                idx_group = []
            idx_group.append(i_frame)

        datas = [{"index": idx,
                  "score": scores[idx]} for idx in range(len(scores))]
        return datas

    def _group_score_function(self, idx_group: list, resize_shape:list) -> list:
        ref_frame = cv2.resize(self.frames[idx_group[0]], resize_shape)
        scores = [0.0]
        for i_frame in range(1, len(idx_group)):
            scored_frame = cv2.resize(
                self.frames[idx_group[i_frame]], resize_shape)
            score = np.abs(ref_frame-scored_frame).mean()
            scores.append(score)
        return scores

    def score_frame(self,
                    group_size: int = 1,
                    resize_shape: tuple = (64, 64),
                    sort: bool = True,
                    unitized: bool = True) -> list:
        assert self.scores is not None and self.scores == [], "please call reset first"

        if group_size == 1:
            self.scores = self._score(resize_shape)
        elif group_size > 1:
            self.scores = self._group_score(group_size, resize_shape)

        if sort:
            self.scores = self._sort_score(self.scores)
        if unitized:
            self.scores = self._unitize(self.scores)
        return self.scores
