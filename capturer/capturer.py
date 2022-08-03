from abc import abstractmethod
import cv2
import numpy as np
from typing import Tuple
from unittest import TestCase


class Capturer:
    def __init__(self, *args, **kwargs):
        self.video_path = None

    def reset(self, video_path):
        self.video_path = video_path
        self.frames = []
        self.infos = {}

    def extract_frame(self) -> Tuple["frames", "info"]:
        assert self.video_path is not None and self.frames == [], "please call reset first"
        cap = cv2.VideoCapture(self.video_path)
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
        test_video_path = "my_unittest/test.mp4"
        self.cap.reset(test_video_path)

    def test_extract_frame(self):
        print(
            f"--------------------{self.tested_name} test extract_frame--------------------")
        frames, infos = self.cap.extract_frame()
        print(f"frame length: {len(frames)}")
        print(infos)