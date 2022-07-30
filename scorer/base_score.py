from abc import abstractmethod
from copy import deepcopy
import functools


class BaseScore:
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def score_frame(self, 
                    video_cap,
                    group_size = None,
                    transforms = None,
                    **kwargs)->list:
        """
        Return the raw score of each frame
        Scale of each score may vary due to the difference between score function
        
        Params:
            video_cap: The CV2's VideoCapture object
            transforms: An image's size transformer, which is realized in utils
        Return format:
            [{"index":index1, "score":score1}, {"index":index2, "score":score2}, ...]
            This list is sorted from the max score to the min score
        """
        pass

    def normalized_score_frame(self,
                                video_cap,
                                group_size = None,
                                transforms = None,
                                **kwargs):
        if group_size is not None:
            datas = self.score_frame(video_cap, group_size, transforms, **kwargs)
        else:
            datas = self.score_frame(video_cap, transforms=transforms, **kwargs)
        normalized_datas = self._normalize(datas)
        return normalized_datas

    def _sort_score(self, datas):
        def cmp(data, data_):
            score = data["score"]
            score_ = data_["score"]
            if score > score_:
                return 1
            elif score == score_:
                return 0
            else:
                return -1
        sorted_datas = sorted(datas, key=functools.cmp_to_key(cmp), reverse=True)
        return sorted_datas

    def _normalize(self, datas:list):
        # normalized scores of datas and return
        normalized_datas = []
        max_score = datas[0]["score"]
        min_score = datas[-1]["score"]
        for data in datas:
            score = (data["score"] - min_score)/(max_score - min_score)
            normalized_datas.append({"index": data["index"],
                                     "score": score})
        return normalized_datas

        
    def __call__(self, *args, **kwds):
        return self.score_frame(*args, **kwds)
    