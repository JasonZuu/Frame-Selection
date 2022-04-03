import cv2
import numpy as np
from copy import deepcopy
import torch

from .model import DBFace, detect, intv

@torch.no_grad()
class PullFaceTool:
    def __init__(self, 
                weights_path:str="data/weights/dbface/dbface.pth",
                device:str="cpu"):
        self.device = torch.device(device)
        self.model = DBFace()
        self.model.load(weights_path, device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def pull_faces(self, img)->np.ndarray:
        bboxs = detect(self.model, img, device=self.device, threshold=0.4)
        faces = self._bboxs_to_faces(img, bboxs, size_threshold=10)
        return faces

    def _bboxs_to_faces(self, image, bboxs, size_threshold=10):
        """
        对图片中的人脸进行截取，并返回人脸图像列表
        @Author  :   JasonZuu
        @Time    :   2021/03/31 17:48:28
        :params: image: cv2读取的图片
        :params: bboxs: detect获取的图像中人脸检测结果
        :Params: size_threshold: face.size的最小值，小于该值的face会被过滤
        :return: faces: 包含当前帧人脸的列表
        """
        faces = []
        for bbox in bboxs:
            x, y, r, b = intv(bbox.box)
            cropped = image[y:b, x:r]
            if cropped.size > size_threshold*3:
                faces.append(deepcopy(cropped))
        return faces

if __name__ == "__main__":
    img = cv2.imread("data/FSS/frames/c23/real/000/B/1.jpg")
    tool = PullFaceTool(weights_path="data/weights/dbface/dbface.pth", device="cuda:0")
    faces = tool.pull_faces(img)
    for face in faces:
        cv2.imshow("test", face)
        cv2.waitKey(0)
