import numpy as np

from sklearn.linear_model import LogisticRegression
import pandas as pd


# sys.path.append("../src")

# from MLFunctions import MLRegressions as ml
# from ..src.MLFunctions
import src.MLFunctions.MLRegressions as ml


def test_MSError():
    assert ml.calcMSError([3], [3]) == 0
    assert ml.calcMSError([3], [5]) == 4
    assert ml.calcMSError([5], [3]) == 4
    assert ml.calcMSError([5, 1, 4, 6], [3, 3, 5, 6]) == 2.25
    assert ml.calcMSError([5, 1, 4, 6], [5, 1, 4, 6]) == 0


def test_dataScaling():
    assert (
        ml.scaleData(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
        == [[0], [0.25], [0.5], [0.75], [1]]
    ).all() == True

    assert (
        ml.scaleData(np.array([[1.0, 10], [2.0, 20], [3.0, 30], [4.0, 40], [5.0, 50]]))
        == [[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1, 1]]
    ).all() == True


def test_sigmoid():
    assert ml.sigmoid(0) == 0.5
    assert ml.sigmoid(1) == 0.7310585786300049
    assert ml.sigmoid(-1) == 0.2689414213699951


def test_gradientDescentIdentical():
    updateWeights, updateC = ml.gradientDesecent(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        0.1,
    )
    assert updateWeights == 0
    assert updateC == 0
