import cv2
import numpy as np
from typing import Optional, Callable, Union
from unittest import TestCase
from pathlib import Path
import os


class Selector:
    def __init__(self, **kwargs):
        super().__init__()
        self.frames = None
        self.scores = None
        self.selected_frames:dict = None # {idx: frame}

    def reset(self,
              frames: list,
              scores: list):
        self.frames = frames
        self.scores = scores

    def select_frame(self,
                     select_num: Union[int, float] = 0.1,
                     reverse: bool = False) -> list:
        assert self.frames is not None, "please call reset first"

        wanted_frames = []
        select_num = int(select_num) if select_num >= 1 else select_num

        if isinstance(select_num, int):
            select_count = select_num
        elif isinstance(select_num, float) and 0.0 < select_num <= 1.0:
            select_count = int(select_num*len(self.frames))
        else:
            raise TypeError(
                "select_num should be a float(0.0<frac<=1.0) or an int")

        if reverse is False:
            wanted_scores = self.scores[:select_count]
        else:
            wanted_scores = self.scores[-select_count:]

        # get wanted frames
        wanted_idxs = [score["index"] for score in wanted_scores]
        for idx in wanted_idxs:
            wanted_frame = self.frames[idx]
            wanted_frames.append(wanted_frame)

        self.selected_frames = {f"{idx}": self.frames[idx] for idx in wanted_idxs}

        return wanted_frames
    
    def export_frame(self, export_dir:str) -> list:
        assert self.selected_frames is not None, "please select frame first"

        export_paths = []

        for frame_idx, frame in self.selected_frames.items():
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            export_path = os.path.join(export_dir, f"{frame_idx}.jpg")
            cv2.imwrite(export_path, frame)
            export_paths.append(export_path)
        
        return export_paths
        
class TestSelector(TestCase):
    def __init__(self, methodName: str, selector: Selector, scorer: Callable, cap: Callable) -> None:
        super().__init__(methodName)
        self.cap = cap
        self.scorer = scorer
        self.selector = selector
        self.tested_name = "Selector"

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

    def test_select_frame(self):
        print(
            f"--------------------{self.tested_name} test select_frame--------------------")
        frames = self.selector.select_frame(select_num=10, reverse=False)
        self.assertEqual(len(frames), 10)
    
    def test_export_frame(self):
        print(
            f"--------------------{self.tested_name} test export_frame--------------------")
        export_paths = self.selector.export_frame(export_dir="data")
        print(export_paths[:5])
