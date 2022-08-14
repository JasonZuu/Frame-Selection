import cv2
import numpy as np

from registry import Registries
from utils import conv_numpy

from .base_scorer import BaseScorer


@Registries.scorer.register("smd2")
class SMD2Scorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score_frame(self,
                    group_size: int = 1,
                    resize_shape: tuple = (64, 64),
                    sort: bool = True,
                    unitized: bool = True) -> list:
        assert self.scores is not None and self.scores == [], "please call reset first"

        frame_count = int(len(self.frames))
        for i_frame in range(frame_count):
            frame = cv2.cvtColor(self.frames[i_frame], cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, resize_shape)

            score = smd2_conv_numpy(frame)
            data = {"index": i_frame,
                    "score": score}
            self.scores.append(data)

        if sort:
            self.scores = self._sort_score(self.scores)
        if unitized:
            self.scores = self._unitize(self.scores)
        return self.scores


"""
    calculate the smd according to its definition
    very slow
"""


def smd2_defination(img, **kwargs):
    f = np.matrix(img) / 255.0  # 返回矩阵
    x, y = f.shape
    smd2 = 0
    for i in range(x - 1):
        for j in range(y - 1):
            smd2 += np.abs(f[i, j]-f[i+1, j])*np.abs(f[i, j]-f[i, j+1])
    return smd2


"""
    Calculate smd2 with convolution
    quicker than the defination one more than 10 times
"""

def smd2_conv_numpy(gray):
    kernel_x = np.array([[1.0, -1.0],  # shape: (2, 2)
                            [0.0, 0.0]])
    kernel_y = np.array([[1.0, 0.0],
                        [-1.0, 0.0]])
                        
    gray_x = conv_numpy(gray, kernel_x, stride=1)
    gray_y = conv_numpy(gray, kernel_y, stride=1)

    result = np.sum(gray_x * gray_y)
    return result
