from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from registry import Registries
from utils import PullFaceTool

from .base_eval import BaseEval


@Registries.evaluation.register("mae")
class MAEEval(BaseEval):
    def eval_score(self, frames: list, label: int, trans=None) -> float:
        assert label in (0, 1), "label must be either 0 or 1"
        score = 0
        for frame in frames:
            faces = self.face_tool.pull_faces_tensor(frame)
            labels = torch.Tensor([label for i in range(faces.shape[0])])
            labels = labels.int().to(self.device)
            faces = faces.to(self.device)
            out = self.model(faces)
            out = F.softmax(out, dim=1)
            logits = out.max(1).values
            mae_score = F.l1_loss(logits, labels, reduction="mean")
            score += float(mae_score)
        score = score / len(frames)
        return float(score)
