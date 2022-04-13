import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registries

from man_src import MultiAttnNet


@Registries.model.register("man")
def get_man(num_classes,
            weight_path: str = None,
            device="cpu",
            **kwargs):
    model = MultiAttnNet(num_classes=num_classes)
    device = torch.device(device)
    if weight_path is not None:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model
