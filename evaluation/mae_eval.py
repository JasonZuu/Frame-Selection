import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from registry import Registries

from .base_eval import BaseEval


@Registries.evaluation.register("mae")
class MAEEval(BaseEval):
    def __init__(self, model:nn.modules):
        super().__init__()
        self.model = model
        self.trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((224, 224))
        ])

    def eval_score(self, frames: list, labels:torch.Tensor, trans=None) -> float:
        if trans is not None:
            self.trans = trans
        data = self.trans(frames)
        out = self.model(data)
        mae_score = F.l1_loss(out, labels, reduction="mean")
        return float(mae_score)
