from abc import abstractmethod
from typing import Callable
from unittest import TestCase


class BaseCritor:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self, selected_frames: list):
        pass

    @abstractmethod
    def evaluate_selection(self):
        pass


class TestCritor(TestCase):
    def __init__(self, methodName: str, critor: BaseCritor, selector: Callable, scorer: Callable, cap: Callable) -> None:
        super().__init__(methodName)
        self.cap = cap
        self.scorer = scorer
        self.selector = selector
        self.critor = critor
        self.tested_name = "Critor"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        test_video_path = "my_unittest/test.mp4"
        self.cap.reset(test_video_path)
        frames, infos = self.cap.extract_frame()
        self.scorer.reset(frames, infos)
        scores = self.scorer.score_frame(
            group_size=24, resize_shape=(64, 64), sort=True, unitized=True)
        self.selector.reset(frames, scores)
        frames = self.selector.select_frames(select_num=10, reverse=False)
        self.critor.reset(frames)

    def test_evaluate_selection(self):
        print(
            f"--------------------{self.tested_name} test evaluate_selection--------------------")
        score = self.critor.evaluate_selection()
        print(f"score:{score}")
