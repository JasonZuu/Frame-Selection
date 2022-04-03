import cv2
from registry import Registries
from utils import Transforms

if __name__ == "__main__":
    Registries.import_all_modules()
    video_path = "data/datasets/IF/raw/002.mp4"
    Score = Registries.score["fd"]()
    trans = Transforms()
    Strategy = Registries.strategy["group"](score=Score)
    datas = Strategy.get_datas(video_path, group_size=10, transforms=trans)
    frames = Strategy(10)
    for frame in frames:
        cv2.imshow("test",frame)
        cv2.waitKey(0)
