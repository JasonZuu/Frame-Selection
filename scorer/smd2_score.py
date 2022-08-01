import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registries
from utils import norm

from .base_scorer import BaseScore


@Registries.score.register("smd2")
class SMD2Score(BaseScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.smd2_conv = SMD2_Conv2d()
    
    def score_frame(self,
                    video_cap: cv2.VideoCapture,
                    transforms=None,
                    **kwargs) -> list:
        assert self.group_size > 1, f"the group size for {self.__class__.__name__} should be greater than 1"
        scores = []
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i_frame in range(frame_count):
            success, frame = video_cap.read()
            if transforms is not None:
                frame = transforms(frame)
            frame = torch.Tensor(frame/255.0)
            score = self.smd2_conv(frame)
            scores.append(score)
        datas = [{"index":i_frame,
                  "score":scores[i_frame]} for i_frame in range(len(scores))]
        datas = self._sort_score(datas)
        return datas


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
        kernel_x = np.array([[[[1.0, -1.0], # shape: (1, 1, 2, 2)
                            [0.0, 0.0]]]])
        kernel_y = np.array([[[[1.0, 0.0],
                            [-1.0, 0.0]]]])
        self.kernel_x = nn.Parameter(torch.tensor(kernel_x, dtype=torch.float))
        self.kernel_y = nn.Parameter(torch.tensor(kernel_y, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros((1)))

    def forward(self, x):
        x = x.view(-1, 1, *x.shape)
        out_x = F.conv2d(x, self.kernel_x, self.bias)
        out_y = F.conv2d(x, self.kernel_x, self.bias)
        out_x = torch.abs(out_x)
        out_y = torch.abs(out_y)
        # out = torch.mul(out_x, out_y)
        out = out_x * out_y
        return torch.sum(out).detach().cpu()

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
