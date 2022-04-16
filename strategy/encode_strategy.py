import shlex
import subprocess
import cv2
import numpy as np
from copy import deepcopy

from registry import Registries

from .base_strategy import BaseStrategy


@Registries.strategy.register("encode")
class EncodeStrategy(BaseStrategy):
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
        frame_types = self._get_frame_types(video_path)
        wanted_frame_index = [x[0] for x in frame_types if x[1] == frame_type]
        self.datas = [{"index": i_frame,
                       "score": 1 if i_frame in wanted_frame_index else 0}
                      for i_frame in range(len(frame_types))]
