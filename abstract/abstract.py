from tokenize import group
from typing import Optional, Tuple, Union
from unittest import TestCase

from registry import Registries
from selector import Selector
from capturer import Capturer


class FrameSelectionAbstract:
    def __init__(self) -> None:
        Registries.import_all_modules()
        self.cap = Capturer()
        self.selector = Selector()
        self.all_frames = None
        self.infos = None
        self.scores = None
        self.selected_frames = None
        self.selection_score = None

    def reset(self,
              video_path: str,
              scorer_label: str = "uniform",
              critor_label: str = "intuitive"):
        assert scorer_label in Registries.scorer, f"scorer {scorer_label} does not exist"
        assert critor_label in Registries.critor, f"critor {critor_label} does not exist"

        self.cap.reset(video_path)
        self.scorer = Registries.scorer[scorer_label]()
        self.critor = Registries.critor[critor_label]()
        self.all_frames, self.infos = self.cap.extract_frame()

    def score(self,
              group_size: int = 1,
              resize_shape: tuple = (64, 64),
              sort: bool = True,
              unitized: bool = True) -> list:
        self.scorer.reset(frames=self.all_frames, video_infos=self.infos)
        self.scores = self.scorer.score_frame(
            group_size, resize_shape, sort, unitized)

        return self.scores

    def select(self,
               select_num: Union[int, float] = 0.1,
               reverse: bool = False) -> Tuple["selection", "selection_score"]:
        self.selector.reset(self.all_frames, self.scores)
        self.selected_frames = self.selector.select_frame(select_num, reverse)
        self.critor.reset(self.selected_frames)
        self.selection_score = self.critor.evaluate_selection()

        return self.selected_frames, self.selection_score


class TestAbstract(TestCase):
    def __init__(self, methodName: str, abstract: FrameSelectionAbstract) -> None:
        super().__init__(methodName)
        self.abstract = abstract
        self.tested_name = "Abstract"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        test_video_path = "my_unittest/test.mp4"
        scorer_label = "uniform"
        critor_label = "intuitive"
        self.abstract.reset(video_path=test_video_path,
                            scorer_label=scorer_label, critor_label=critor_label)

    def test_score(self):
        print(
            f"--------------------{self.tested_name} test score--------------------")
        score = self.abstract.score(group_size=24, resize_shape=(64,64), sort=True, unitized=True)
        print(score[:10])
    
    def test_select(self):
        print(f"--------------------{self.tested_name} test select--------------------")
        selection, score = self.abstract.select(select_num=10, reverse=False)
        self.assertEqual(len(selection), 10)
        print(f"score:{score}")
