def factorial(N):
    if N == 0 or N == 1:
        return 1
    else:
        return N*factorial(N-1)


def unitize(datas:list) -> list:
    max_value = max(datas)
    min_value = min(datas)
    unitized_data = [(data-min_value)/(max_value-min_value) for data in datas]
    return unitized_data