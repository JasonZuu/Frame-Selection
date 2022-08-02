from subprocess import call
import cv2
import numpy as np

from registry import Registries

from .base_scorer import BaseScorer


@Registries.scorer.register("fd")
class FrameDifferenceScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score(self) -> list:
        frame_count = len(self.frames)
        datas = []

        for idx in range(frame_count):
            if idx == 0:
                data = {"index": idx,
                        "score": 255.0}
            else:
                frame_last = cv2.resize(self.frames[idx-1], self.resize_shape)
                frame = cv2.resize(self.frames[idx], self.resize_shape)
                score = np.abs(frame_last-frame).mean()
                data = {"index": idx,
                        "score": score}
            datas.append(data)

        return datas

    def _group_score(self) -> list:
        frame_count = len(self.frames)
        scores = []

        for i_frame in range(frame_count):
            if i_frame == 0:
                idx_group = []
            elif i_frame % self.group_size == 0:
                group_scores = self._group_score_function(idx_group)
                scores.extend(group_scores)
                idx_group = []
            idx_group.append(i_frame)

        datas = [{"index": idx,
                  "score": scores[idx]} for idx in range(len(scores))]
        return datas

    def _group_score_function(self, idx_group: list) -> list:
        ref_frame = cv2.resize(self.frames[idx_group[0]], self.resize_shape)
        scores = [0.0]
        for i_frame in range(1, len(idx_group)):
            scored_frame = cv2.resize(
                self.frames[idx_group[i_frame]], self.resize_shape)
            score = np.abs(ref_frame-scored_frame).mean()
            scores.append(score)
        return scores

    def score_frame(self, sort: bool = True, unitized: bool = True) -> list:
        assert self.scores is not None and self.scores == [], "please call reset first"

        if self.group_size == 1:
            self.scores = self._score()
        elif self.group_size > 1:
            self.scores = self._group_score()

        if sort:
            self.scores = self._sort_score(self.scores)
        if unitized:
            self.scores = self._unitize(self.scores)
        return self.scores
