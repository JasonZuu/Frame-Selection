import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from registry import Registries

from .base_eval import BaseEval


@Registries.evaluation.register("precision")
class PrecisionEval(BaseEval):
    @torch.no_grad()
    def eval_score(self, frames: list, label:int, trans=None) -> float:
        assert label in (0, 1), "label must be either 0 or 1"
        score = 0
        for frame in frames:
            faces = self.face_tool.pull_faces_tensor(frame)
            if faces is None:
                continue
            labels = torch.Tensor([label for i in range(faces.shape[0])])
            faces = faces.to(self.device)
            out = self.model(faces)
            out = out.detach().cpu()
            prec_score = (out.argmax(1) == labels).sum()/len(labels)
            score += prec_score
        if score != 0:
            score = score.detach().cpu() / len(frames)
        return 1.0 if score > 0.5 else 0.0
