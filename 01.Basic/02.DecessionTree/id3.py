import numpy as np
import math
from sklearn.datasets import load_iris
from pprint import pprint


# The whole entropy of two big circles combined
def calc_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict])  # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])  # right hand side entropy
    s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false  # overall entropy, again weighted average
    return s


def entropy_func(c, n):
    """
    The math formula
    """
    return -(c*1.0/n)*math.log(c*1.0/n, 2)


# get the entropy of one big circle showing above
def entropy_of_one_division(division):
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division==c)
        e = n_c * 1.0 / n * entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
        s += e
    return s, n


def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)


class DecisionTreeClassifier(object):
    def __init__(self, max_depth, features):
        self.depth = 0
        self.max_depth = max_depth
        self.features = features

    def all_same(self, items):
        return all(x == items[0] for x in items)

    def fit(self, x, y, parent_node={}, depth=0):
        if parent_node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val': y[0]}
        elif depth >= self.max_depth:
            return None
        else:
            col, cutoff, entropy = self.find_best_split(x, y)
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]

            parent_node = {'col': self.features[col], 'index_col': col, 'cutoff': cutoff, 'val': np.round(np.mean(y))}
            parent_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth + 1)
            parent_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth + 1)
            self.depth += 1
            self.trees = parent_node
            return parent_node

    def find_best_split(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None

        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.calc_best_split(c, y)
            if entropy == 0:  # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff

        return col, cutoff, min_entropy

    def calc_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = calc_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff


if __name__ == "__main__":
    iris = load_iris()

    x = iris.data
    y = iris.target
    clf = DecisionTreeClassifier(max_depth=7, features=iris.feature_names)
    m = clf.fit(x, y)
    pprint(m)
