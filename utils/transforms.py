import cv2
import numpy as np

class Transforms:
    def __init__(self) -> None:
        pass

    def transform(self, img:np.ndarray, **kwargs) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        return img
    
    def __call__(self, *args, **kwds):
        return self.transform(*args, **kwds)