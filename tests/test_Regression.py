import numpy as np


# sys.path.append("../src")

# from MLFunctions import MLRegressions as ml
# from ..src.MLFunctions
import src.MLFunctions.MLRegressions as ml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pytest import approx


def test_MSError():
    assert ml.calcMSError([3], [3]) == 0
    assert ml.calcMSError([3], [5]) == 4
    assert ml.calcMSError([5], [3]) == 4
    assert ml.calcMSError([5, 1, 4, 6], [3, 3, 5, 6]) == 2.25
    assert ml.calcMSError([5, 1, 4, 6], [5, 1, 4, 6]) == 0


def test_dataScaling():

    assert ml.scaleData(
        np.array([[1.0, 10], [2.0, 20], [3.0, 30], [4.0, 40], [5.0, 50]])
    ) == approx(np.array([[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1, 1]]))


def test_sigmoid():
    assert ml.sigmoid(0) == approx(0.5)
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


def test_gradientDescentUpdate1():
    updateWeights, updateC = ml.gradientDesecent(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        0.1,
    )
    assert updateWeights == approx(np.array([-0.3]))
    assert updateC == approx(-0.1)


def predictData(X, weigths, c):
    n = len(X)
    dimensions = X.shape[1]

    YPredClass = []

    for i in range(0, n):
        currentValue = 0
        for dim in range(0, dimensions):
            currentValue = currentValue + X[i][dim] * weigths[dim]
        currentValue += c

        if currentValue > 0.5:
            YPredClass.append(1)
        else:
            YPredClass.append(0)

    return YPredClass


def test_CompareToSklear():
    dimensions = 2
    numberOfSamples = 100
    learningRate = 0.1
    iterations = 10000

    X, Y = ml.getLogisticRegressionDemoData(numberOfSamples, dimensions)
    Xscaled = ml.scaleData(X)
    Yscaled = ml.scaleData(Y)

    weigths, c = ml.runLogisticRegression(Xscaled, Yscaled, learningRate, iterations)

    yPredicted = predictData(Xscaled, weigths, c)
    acc_score = accuracy_score(Y, yPredicted, normalize=True)

    reg = LogisticRegression().fit(Xscaled, Yscaled)
    scoreSKLearn = reg.score(Xscaled, Yscaled)

    print(reg.coef_)
    print(reg.intercept_)
    print(weigths)

    assert acc_score == approx(scoreSKLearn, 2.0)


def test_CompareScalerToSklear():
    from sklearn.preprocessing import MinMaxScaler

    dimensions = 2
    numberOfSamples = 100
    learningRate = 0.1
    iterations = 10000

    X, Y = ml.getLogisticRegressionDemoData(numberOfSamples, dimensions)
    XscaledOwn = ml.scaleData(X)

    scaler = MinMaxScaler()
    xSKLearn = scaler.fit_transform(X)

    assert np.allclose(XscaledOwn, xSKLearn)
