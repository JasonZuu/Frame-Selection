import cv2
import numpy as np

class NumpyTransforms:
    def __init__(self, img_shape=[256, 256]) -> None:
        self.img_shape=img_shape

    def transform(self, img:np.ndarray, **kwargs) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.img_shape)
        return img
    
    def __call__(self, *args, **kwds):
        return self.transform(*args, **kwds)