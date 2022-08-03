from abc import abstractmethod
from copy import deepcopy
import functools
from typing import Callable, Optional, List
from unittest import TestCase


class BaseScorer:
    def __init__(self, **kwargs):
        super().__init__()
        self.scores = None

    def reset(self,
              frames: list,
              video_infos: dict):
        self.frames = frames
        self.video_infos = video_infos
        self.scores = []

    @abstractmethod
    def score_frame(self,
                    group_size: int = 1,
                    resize_shape: tuple = (64, 64),
                    sort: bool = True,
                    unitized: bool = True) -> list:
        """
        Return the raw score of each frame
        Scale of each score may vary due to the difference between score function

        Return format:
            [{"index":index1, "score":score1}, {"index":index2, "score":score2}, ...]
            This list is sorted from the max score to the min score
        """
        pass

    def _sort_score(self, datas):
        def cmp(data, data_):
            score = data["score"]
            score_ = data_["score"]
            if score > score_:
                return 1
            elif score == score_:
                return 0
            else:
                return -1
        sorted_datas = sorted(
            datas, key=functools.cmp_to_key(cmp), reverse=True)
        return sorted_datas

    def _unitize(self, datas: list):
        # unitized scores of datas and return
        unitized_datas = []
        scores = [datas[i]["score"] for i in range(len(datas))]
        max_score = max(scores)
        min_score = min(scores)
        for data in datas:
            score = (data["score"] - min_score)/(max_score - min_score)
            unitized_datas.append({"index": data["index"],
                                   "score": score})
        return unitized_datas


class TestScorer(TestCase):
    def __init__(self, methodName: str, scorer: BaseScorer, cap: Callable) -> None:
        super().__init__(methodName)
        self.cap = cap
        self.scorer = scorer
        self.tested_name = "Scorer"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        test_video_path = "my_unittest/test.mp4"
        self.cap.reset(test_video_path)
        frames, infos = self.cap.extract_frame()
        self.scorer.reset(frames, infos)

    def test_score_frame(self):
        print(
            f"--------------------{self.tested_name} test score_frame--------------------")
        scores = self.scorer.score_frame(
            group_size=24, resize_shape=(64, 64), sort=True, unitized=True)
        print(scores[:10])
