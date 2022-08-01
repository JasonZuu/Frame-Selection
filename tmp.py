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