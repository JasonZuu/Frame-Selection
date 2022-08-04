from typing import Optional, Tuple, Union
from unittest import TestCase
import os

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
              video_path: str = None,
              scorer_key: str = "ssim",
              critor_key: str = "intuitive",
              help_mode: bool = False):
        if help_mode:
            help_info = {"scorer_keys": list(Registries.scorer.keys()),
                         "critor_keys": list(Registries.critor.keys())
                         }
            return help_info
        assert video_path is not None, "if you do not enter with help mode, you should give video_path"
        assert os.path.exists(video_path), "the video do not exist"
        assert scorer_key in Registries.scorer, f"scorer {scorer_key} does not exist"
        assert critor_key in Registries.critor, f"critor {critor_key} does not exist"

        self.cap.reset(video_path)
        self.scorer = Registries.scorer[scorer_key]()
        self.critor = Registries.critor[critor_key]()
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

    def export(self,
               export_dir: str) -> list:
        export_dir = os.path.join(export_dir, self.infos["video_name"])
        export_paths = self.selector.export_frame(export_dir=export_dir)
        return export_paths


class TestAbstract(TestCase):
    def __init__(self, methodName: str, abstract: FrameSelectionAbstract) -> None:
        super().__init__(methodName)
        self.abstract = abstract
        self.tested_name = "Abstract"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        help_infos = self.abstract.reset(help_mode=True)
        print(help_infos)
        test_video_path = "my_unittest/test.mp4"
        scorer_key = "uniform"
        critor_key = "intuitive"
        self.abstract.reset(video_path=test_video_path,
                            scorer_key=scorer_key,
                            critor_key=critor_key,
                            help_mode=False)

    def test_score(self):
        print(
            f"--------------------{self.tested_name} test score--------------------")
        score = self.abstract.score(
            group_size=24, resize_shape=(64, 64), sort=True, unitized=True)
        print(score[:10])

    def test_select(self):
        print(
            f"--------------------{self.tested_name} test select--------------------")
        selection, score = self.abstract.select(select_num=10, reverse=False)
        self.assertEqual(len(selection), 10)
        print(f"score:{score}")

    def test_export(self):
        print(
            f"--------------------{self.tested_name} test export--------------------")
        export_paths = self.abstract.export("data")
        print(export_paths[:5])
