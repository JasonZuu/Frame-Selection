import cv2
import argparse
import yaml
import torch
import time

from registry import Registries
from utils import NumpyTransforms, PullFaceTool
from dataloader import MyDataLoader


parser = argparse.ArgumentParser(
    description="Default runner of Frame Selection")
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/demo.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

Registries.import_all_modules()

# videos and labels
data_loader = MyDataLoader(**config["dataloader_params"])

# Set up Strategy
Score = Registries.score[config["strategy_params"]["score_key"]]()
trans = NumpyTransforms(config["strategy_params"]["score_img_shape"])
Strategy = Registries.strategy[config["strategy_params"]["strategy_key"]](
    score=Score)

# Set up Evaluation
face_tool = PullFaceTool(**config["facetool_params"])
model = Registries.model[config["model_params"]["key"]](
    **config["model_params"])
Eval = Registries.evaluation[config["eval_params"]["key"]](
    model, face_tool, **config["eval_params"])

# Running
print("------------------Start Evaluating----------------------")
start_time = time.time()
scores = 0
unused_video_count = 0
data_count = data_loader.data_count()
for i_batch, (video_path, label) in enumerate(data_loader.get_dataloader()):
    video_path = video_path[0]
    label = int(label)
    print(f"Processing {i_batch+1}/{data_count} video", end=":\t")
    Strategy.get_datas(
        video_path, group_size=config["running_params"]["group_size"], transforms=trans)
    frames = Strategy(config["running_params"]["frac"])
    if len(frames) == 0:
        unused_video_count += 1
        continue
    score = Eval(frames, label=label)
    scores += score
    print(scores)
    torch.cuda.empty_cache()
if data_count == unused_video_count:
    print("no valid video")
else:
    scores /= data_count-unused_video_count
    print(scores)
end_time = time.time()
span_time = end_time - start_time
print(f"Lasting for {span_time} second")