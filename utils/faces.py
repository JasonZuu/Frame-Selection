import cv2
import numpy as np
from copy import deepcopy
import torch
from torchvision import transforms

from .model import DBFace, detect, intv


@torch.no_grad()
class PullFaceTool:
    def __init__(self,
                 weights_path: str = "data/weights/dbface/dbface.pth",
                 device: str = "cpu",
                 face_shape=[224, 224],
                 **kwargs):
        self.device = torch.device(device)
        self.model = DBFace()
        self.model.load(weights_path, device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.face_shape = face_shape
        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.face_shape),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def pull_faces_numpy(self, image, size_threshold=10) -> np.ndarray:
        bboxs = detect(self.model, image, device=self.device, threshold=0.4)
        faces = []
        for bbox in bboxs:
            x, y, r, b = intv(bbox.box)
            cropped = image[y:b, x:r]
            if cropped.size > size_threshold*3:
                face = cv2.resize(cropped, self.face_shape)
                faces.append(deepcopy(face))
        return faces

    def pull_faces_tensor(self, image, size_threshold=10) -> torch.Tensor:
        bboxs = detect(self.model, image, device=self.device, threshold=0.4)
        faces = None
        for bbox in bboxs:
            x, y, r, b = intv(bbox.box)
            cropped = image[y:b, x:r]
            # cv2.imshow("test", cropped)
            # cv2.waitKey(0)
            if cropped.size > size_threshold*3:
                face = self.tensor_transforms(cropped)
                face = face.view(-1, *face.shape)
                if faces is None:
                    faces = face
                else:
                    faces = torch.cat((faces, face), dim=0)
            else:
                continue
        return faces
