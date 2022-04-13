import cv2
import argparse
import yaml
import torch

from registry import Registries
from utils import NumpyTransforms, PullFaceTool


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
video_path = "data/datasets/celeb/fake/id0_id1_0000.mp4"

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
Strategy.get_datas(
    video_path, group_size=config["running_params"]["group_size"], transforms=trans)
frames = Strategy(config["running_params"]["frac"])
score = Eval(frames, label=1)
print(score)
