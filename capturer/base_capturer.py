from abc import abstractmethod
import cv2
import numpy as np

class BaseCapturer:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.frames = None
        self.infos = {}

    @abstractmethod
    def extract_frame(self,
                  video_path):
        pass

    