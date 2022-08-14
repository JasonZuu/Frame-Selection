import numpy as np
from numba import jit

@jit(nopython=True)
def conv_numpy(inputs, filter, stride:int=1):
    """
    the filter's shape must be N*N
    """
    H, W = inputs.shape
    filter_size = filter.shape[0]
    result = np.zeros((int((H - filter_size)/stride + 1), int((W - filter_size)/stride + 1)))

    #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
    for r in range(0, int((H - filter_size)/stride + 1)):
        for c in range(0,  int((W - filter_size)/stride + 1)):
            # 池化大小的输入区域
            cur_input = inputs[r*stride:r*stride + filter_size,
                        r*stride:r*stride + filter_size]
            #和核进行乘法计算
            cur_output = cur_input * filter
            #再把所有值求和
            conv_sum = np.sum(cur_output)
            #当前点输出值
            result[r, c] = conv_sum

    return result

if __name__ == "__main__":
    inputs = np.ones((7,7))
    filter = np.ones((3,3))
    result = conv_numpy(inputs=inputs, filter=filter, stride=2)
    print(result)
