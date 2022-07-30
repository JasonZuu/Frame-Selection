from abc import abstractmethod
import cv2
import numpy as np

class BaseStrategy:
    def __init__(self, score, **kwargs):
        super().__init__()
        self.score = score
        self.video_cap = None
        self.datas = None

    @abstractmethod
    def get_datas(self,
                  video_path, 
                  transforms:object=None):
        pass


    def select_frames(self,
                      frac: float = None,
                      reverse: bool = False) -> list:
        wanted_frames = []
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 把帧指针归0
        if frac is None:
            select_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)/self.group_size)
        elif isinstance(frac, int):
            select_count = frac
        elif isinstance(frac, float) and 0.0 < frac <= 1.0:
            select_count = int(frac*len(self.datas))
        else:
            raise TypeError("frac should be float(0.0<frac<=1.0) or int")

        if reverse is False:
            datas = self.datas[:select_count]
        else:
            datas = self.datas[-select_count:]
        # get wanted frames
        frame_idxs = sorted([data["index"] for data in datas])
        success, frame = self.video_cap.read()
        idx = 0
        i_frame_idxs = 0
        while success:
            if i_frame_idxs < len(frame_idxs) and idx == frame_idxs[i_frame_idxs]:
                i_frame_idxs += 1
                wanted_frames.append(frame)
            success, frame = self.video_cap.read()
            idx += 1

        return wanted_frames

    def __call__(self, *args, **kwargs):
        return self.select_frames(*args, **kwargs)
    