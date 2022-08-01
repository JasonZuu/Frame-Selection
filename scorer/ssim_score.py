import cv2
import numpy as np
from skimage.metrics import structural_similarity

from registry import Registries

from .base_scorer import BaseScore


@Registries.score.register("ssim")
class SSIMScore(BaseScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_for_one_group(self, frame_group:list) -> list:
        ref_img = frame_group[0]
        scores = []
        for i_frame in range(len(frame_group)):
            score = 1.0 - structural_similarity(ref_img, frame_group[i_frame])
            scores.append(score)
        return scores

    def score_frame(self,
                    video_cap: cv2.VideoCapture,
                    group_size: int=2,
                    transforms=None,
                    **kwargs) -> list:
        assert group_size > 1
        scores = []
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i_frame in range(frame_count):
            success, frame = video_cap.read()
            if transforms is not None:
                frame = transforms(frame)
            if i_frame == 0:
                group = []
            elif i_frame%group_size == 0:
                ssim_scores = self._score_for_one_group(group)
                scores.extend(ssim_scores)
                group = []
            group.append(frame)
        datas = [{"index":i_frame,
                  "score":scores[i_frame]} for i_frame in range(len(scores))]
        datas = self._sort_score(datas)
        return datas
