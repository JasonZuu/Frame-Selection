
def norm(datas:list) -> list:
    max_value = max(datas)
    min_value = min(datas)
    norm_data = [(data-min_value)/(max_value-min_value) for data in datas]
    return norm_data