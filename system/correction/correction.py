# -*- coding: utf-8 -*-
# @Time         : 2024/12/26 16:50
# @Author       : Song Zilong
# @Software     : PyCharm
# @Description  : 标签修正相关
import numpy as np
from scipy.spatial.distance import cdist


def lid_term(x, batch, k=20):
    print("x:{}".format(x))
    eps = 1e-6
    x = np.asarray(x, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    distances = cdist(x, batch)
    print("distances:{}".format(distances))
    sorted_distances = np.sort(distances, axis=1)[:, 1:k + 1]
    print("sorted_distances:{}".format(sorted_distances))

    lids = np.apply_along_axis(lambda v: - k / (np.sum(np.log(v / (v[-1] + eps))) + eps), axis=1, arr=sorted_distances)
    return lids
