from abc import abstractmethod
import cv2
import numpy as np
from typing import Tuple
from unittest import TestCase


class Capturer:
    def __init__(self, *args, **kwargs):
        self.frames = None
        self.infos = None

    def reset(self):
        self.frames = []
        self.infos = {}

    def extract_frame(self,
                      video_path: str) -> Tuple["frames", "info"]:
        assert self.frames is not None and self.frames == [], "please call reset first"
        cap = cv2.VideoCapture(video_path)
        # get infos
        self.infos["fps"] = cap.get(cv2.CAP_PROP_FPS)
        self.infos["frame_shape"] = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                     cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # (width, height)
        # get frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        return self.frames, self.infos


class TestCapturer(TestCase):
    def __init__(self, methodName: str, cap:Capturer) -> None:
        super().__init__(methodName)
        self.cap = cap
        self.tested_name = "Capturer"

    def test_reset(self):
        print(
            f"--------------------{self.tested_name} test reset--------------------")
        self.cap.reset()

    def test_extract_frame(self):
        print(
            f"--------------------{self.tested_name} test extract_frame--------------------")
        test_video_path = "my_unittest/test.mp4"
        frames, infos = self.cap.extract_frame(test_video_path)
        print(f"frame length: {len(frames)}")
        print(infos)