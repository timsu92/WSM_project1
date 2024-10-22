import numpy as np
from numpy import dot, isnan
from numpy.linalg import norm

def euclidean_distance(a, b):
    return norm(a - b)

def cosine(a, b):
    ans = dot(a, b) / (norm(a) * norm(b))
    if isnan(ans):
        return 0
    return ans

def normalized(vec: np.ndarray):
    length = norm(vec)
    if length == 0:
        return vec
    return vec / length