import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from registry import Registries

from .base_scorer import BaseScorer


@Registries.scorer.register("smd2")
class SMD2Scorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.smd2_conv = SMD2_Conv2d()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cpu()
        self.smd2_conv = self.smd2_conv.to(self.device)
        self.transform = transforms.ToTensor()

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
            frame = self.transform(frame)
            frame = frame.to(self.device)

            score = self.smd2_conv(frame)
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


class SMD2_Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = np.array([[[[1.0, -1.0],  # shape: (1, 1, 2, 2)
                            [0.0, 0.0]]]])
        kernel_y = np.array([[[[1.0, 0.0],
                            [-1.0, 0.0]]]])
        self.kernel_x = nn.Parameter(torch.tensor(kernel_x, dtype=torch.float))
        self.kernel_y = nn.Parameter(torch.tensor(kernel_y, dtype=torch.float))

    def forward(self, x):
        x = x.view(-1, *x.shape)
        out_x = F.conv2d(input=x, weight=self.kernel_x)
        out_y = F.conv2d(input=x, weight=self.kernel_y)
        out_x = torch.abs(out_x)
        out_y = torch.abs(out_y)
        # out = torch.mul(out_x, out_y)
        out = out_x * out_y
        return torch.sum(out).detach().cpu()

# for an image


@torch.no_grad()
def smd2_conv(gray, CNN: nn.Module, device: torch.device):
    gray = cv2.resize(gray, SET_IMG_SHAPE, interpolation=cv2.INTER_CUBIC)
    gray = gray / 255.0
    gray = torch.tensor(gray, dtype=torch.float32)
    gray = gray.to(device)
    smd2 = CNN(gray)
    return smd2


def smd2(img, **kwargs):
    score = smd2_conv(img, **kwargs)
    return float(score)
