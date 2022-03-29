import cv2
from registry import Registries
from utils import Transforms

if __name__ == "__main__":
    Registries.import_all_modules()
    video_path = "data/dataset/IF_demo/001.mp4"
    Score = Registries.score["ssim"]()
    trans = Transforms()
    Strategy = Registries.strategy["group"](score=Score)
    datas = Strategy.get_datas(video_path, group_size=40, transforms=trans)
    frames = Strategy()
    for frame in frames:
        cv2.imshow("test",frame)
        cv2.waitKey(0)
