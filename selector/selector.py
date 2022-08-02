import cv2
import numpy as np
from typing import Optional, Callable, Union
from unittest import TestCase


class Selector:
    def __init__(self, **kwargs):
        super().__init__()
        self.frames = None
        self.scores = None

    def reset(self,
              frames: list,
              scores: list):
        assert len(frames) == len(scores)
        self.frames = frames
        self.scores = scores

    def select_frames(self,
                      select_num: Union[int, float]= 0.1,
                      reverse: bool = False) -> list:
        assert self.frames is not None, "please call reset first"

        wanted_frames = []

        if isinstance(select_num, int):
            select_count = select_num
        elif isinstance(select_num, float) and 0.0 < select_num <= 1.0:
            select_count = int(select_num*len(self.frames))
        else:
            raise TypeError("select_num should be a float(0.0<frac<=1.0) or an int")

        if reverse is False:
            wanted_scores = self.scores[:select_count]
        else:
            wanted_scores = self.scores[-select_count:]

        # get wanted frames
        wanted_idxs = [score["index"] for score in wanted_scores]
        for idx in wanted_idxs:
            wanted_frame = self.frames[idx]
            wanted_frames.append(wanted_frame)

        return wanted_frames


class TestSelector(TestCase):
    def __init__(self, methodName: str, selector:Selector, scorer:Callable, cap:Callable) -> None:
        super().__init__(methodName)
        self.cap = cap
        self.scorer = scorer
        self.selector = selector
        self.tested_name = "Selector"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        test_video_path = "my_unittest/test.mp4"
        self.cap.reset()
        frames, infos = self.cap.extract_frame(test_video_path)
        self.scorer.reset(frames, infos, group_size=24, resize_shape=(64,64))
        scores = self.scorer.score_frame(sort=True, unitized=True)
        self.selector.reset(frames, scores)

    def test_select_frame(self):
        print(
            f"--------------------{self.tested_name} test select_frame--------------------")
        frames = self.selector.select_frames(select_num=10, reverse=False)
        self.assertEqual(len(frames), 10)