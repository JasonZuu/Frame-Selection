from abc import abstractmethod

class BaseEval:
    def __init__(self):
        pass
    
    @abstractmethod
    def eval_score(self, frames:list) -> float:
        pass

    def __call__(self, *args, **kwargs):
        return self.eval_score(*args, **kwargs)
