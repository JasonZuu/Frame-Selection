from abc import abstractmethod
import torch
import torch.nn as nn

class BaseEval:
    def __init__(self,
                 model: nn.modules,
                 face_tool,
                 device: str = "cpu",
                 **kwargs):
        super().__init__()
        self.model = model
        self.face_tool = face_tool
        self.device = torch.device(device)
    
    @abstractmethod
    def eval_score(self, frames:list, label:int, trans=None) -> float:
        """
        label=0 for real and label=1 for fake
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.eval_score(*args, **kwargs)
