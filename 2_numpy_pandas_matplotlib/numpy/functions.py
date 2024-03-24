from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    size = min(len(X), len(X[0]))
    res = 0
    flag = False
    for i in range(size):
        if X[i][i] >= 0:
            res += X[i][i]
            flag = True
    return res if flag else -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    x = sorted(x)
    y = sorted(y)
    return x == y


def max_prod_mod_3(x: List[int]) -> int:
    res = 0
    flag = False
    for i in range(len(x) - 1):
        prod = x[i] * x[i + 1]
        if prod % 3 == 0:
           res = max(res, prod) if flag else prod
           flag = True
    return res if flag else -1
    



def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    height = len(image)
    width = len(image[0])
    num_channels = len(weights)
    res = [[0 for _ in range(width)] for _ in range(height)]
    for h in range(height):
        for w in range(width):
            weighted_sum = 0
            for c in range(num_channels):
                weighted_sum += image[h][w][c] * weights[c]
            res[h][w] = weighted_sum
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    x1 = []
    for element, count in x:
        x1.extend([element] * count)
    y1 = []
    for element, count in y:
        y1.extend([element] * count)
    if len(x1) != len(y1):
        return -1
    res = sum(x * y for x, y in zip(x1, y1))
    return res


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    n = len(X)
    m = len(Y)
    res = []
    for i in range(n):
        row = []
        for j in range(m):
            dot_product = sum(X[i][k] * Y[j][k] for k in range(len(X[i])))
            norm_x = sum(X[i][k] ** 2 for k in range(len(X[i])))
            norm_y = sum(Y[j][k] ** 2 for k in range(len(Y[j])))
            if norm_x == 0 or norm_y == 0:
                row.append(1.0)
            else:
                row.append(dot_product / (norm_x ** 0.5 * norm_y ** 0.5))
        res.append(row)
    return res

