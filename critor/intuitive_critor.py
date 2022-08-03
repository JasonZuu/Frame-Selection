from skimage.metrics import structural_similarity
from skimage.measure import shannon_entropy
import numpy as np
import cv2

from registry import Registries

from .base_critor import BaseCritor


@Registries.critor.register("intuitive")
class IntuitiveCritor(BaseCritor):
    def __init__(self) -> None:
        super().__init__()
        self.selected_frames = None

    def reset(self, selected_frames: list):
        self.selected_frames = selected_frames

    def evaluate_selection(self):
        assert self.selected_frames is not None, "please call reset first"
        entropies = []
        ssim_scores = []
        for i in range(len(self.selected_frames)):
            frame = self.selected_frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (64,64))

            entropy = shannon_entropy(frame)
            entropies.append(entropy)

            for j in range(i, len(self.selected_frames)):
                compared_frame = self.selected_frames[j]
                compared_frame = cv2.cvtColor(compared_frame, cv2.COLOR_BGR2GRAY)
                compared_frame = cv2.resize(compared_frame, (64,64))

                ssim_score = structural_similarity(frame, compared_frame)
                ssim_scores.append(ssim_score)
        eval_score = np.mean(entropies) / np.mean(ssim_scores)
        return eval_score
