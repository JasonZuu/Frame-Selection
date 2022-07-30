import shlex
import subprocess
import cv2
import numpy as np
from copy import deepcopy

from registry import Registries

from .base_strategy import BaseStrategy


@Registries.strategy.register("encode_based")
class EncodeBasedStrategy(BaseStrategy):
    def __init__(self, score: object, **kwargs):
        super().__init__(score, **kwargs)

    def _get_frame_types(self, video_fn):
        command = f'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1 {video_fn}'
        args = shlex.split(command)
        out = subprocess.check_output(args).decode()
        frame_types = out.replace('pict_type=', '').split()
        return zip(range(len(frame_types)), frame_types)

    def get_datas(self,
                  video_path,
                  frame_type="I",
                  **kwargs):
        self.video_cap = cv2.VideoCapture(video_path, 0)
        frame_types = self._get_frame_types(video_path)
        self.wanted_frame_indexs = [x[0] for x in frame_types if x[1] == frame_type]

    def select_frames(self,
                      frac: float = None,
                      reverse: bool = False) -> list:
        wanted_frames = []
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 把帧指针归0

        success, frame = self.video_cap.read()
        idx = 0
        while success:
            if idx in self.wanted_frame_indexs:
                wanted_frames.append(frame)
            success, frame = self.video_cap.read()
            idx += 1

        return wanted_frames
