from subprocess import call
import cv2
import numpy as np

from registry import Registries

from .base_score import BaseScore


def _fd_score_for_one_group(frame_group: list) -> list:
    ref_img = frame_group[0]
    scores = []
    for i_frame in range(len(frame_group)):
        score = np.abs(ref_img-frame_group[i_frame]).mean()
        scores.append(score)
    return scores


@Registries.score.register("fd")
class FrameDifferenceScore(BaseScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score(self,
               video_cap: cv2.VideoCapture,
               transforms=None) -> list:
        scores = []
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i_frame in range(frame_count):
            success, frame = video_cap.read()
            if transforms is not None:
                frame = transforms(frame)
            if i_frame == 0:
                scores.append(255.0)
            else:
                fd = np.abs(_frame-frame).mean()
                scores.append(fd)
            _frame = frame
        # datas
        datas = [{"index": idx,
                  "score": scores[idx]} for idx in range(len(scores))]
        datas = self._sort_score(datas)
        return datas

    def _group_score(self,
                     video_cap: cv2.VideoCapture,
                     group_size: int = 2,
                     transforms=None) -> list:
        scores = []
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i_frame in range(frame_count):
            success, frame = video_cap.read()
            if transforms is not None:
                frame = transforms(frame)
            if i_frame == 0:
                group = []
            elif i_frame % group_size == 0:
                fd_scores = _fd_score_for_one_group(group)
                scores.extend(fd_scores)
                group = []
            group.append(frame)
        # datas
        datas = [{"index": idx,
                  "score": scores[idx]} for idx in range(len(scores))]
        datas = self._sort_score(datas)
        return datas

    def score_frame(self,
                    video_cap: cv2.VideoCapture,
                    group_size=None,
                    transforms=None,
                    **kwargs) -> list:
        if group_size is None:
            return self._score(video_cap=video_cap, transforms=transforms)
        else:
            return self._group_score(video_cap=video_cap, group_size=group_size, transforms=transforms)
