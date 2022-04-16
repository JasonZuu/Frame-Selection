import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import numpy as np

from registry import Registries

from .base_eval import BaseEval


@Registries.evaluation.register("auc")
class AUCEval(BaseEval):
    @torch.no_grad()
    def eval_score(self, frames: list, label:int, trans=None) -> float:
        assert label in (0, 1), "label must be either 0 or 1"
        score = 0
        for i_frame in range(len(frames)):
            frame = frames[i_frame]
            faces = self.face_tool.pull_faces_tensor(frame)
            if faces is None:
                continue
            faces = faces.to(self.device)
            out = self.model(faces)
            if i_frame == 0:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=0)
        logits = outs.detach().cpu().numpy()
        labels = np.array([label for i in range(logits.shape[0])])
        fpr, tpr, _ = roc_curve(labels, logits[:, 1])
        auc_score = auc(fpr, tpr)
        score = 0 if auc_score is None else auc_score
        return float(score)
