import numpy as np
import sklearn

# from sklearn import datasets


def calcMSError(calculatedResults, expectedResults):
    error = 0
    for i in range(len(calculatedResults)):
        error = error + (calculatedResults[i] - expectedResults[i]) ** 2

    error = error / len(calculatedResults)
    return error


def test_answer():
    assert calcMSError(3, 3) == 0


def gradientDesecent(Data, Y, Y_predicted, learningRate):
    n = len(Data)
    dimensions = Data.shape[1]
    # dimensions = 1
    updateWeights = np.zeros(dimensions)
    updateC = 0

    for index in range(0, n):
        for currentDim in range(0, dimensions):
            updateWeights[currentDim] = updateWeights[currentDim] + (
                Data[index][currentDim] * (Y[index] - Y_predicted[index])
            )

    for currentDim in range(0, dimensions):
        updateWeights[currentDim] = (-1 / n) * updateWeights[currentDim]
        updateWeights[currentDim] = updateWeights[currentDim] * learningRate

    for index in range(0, n):
        updateC = updateC + (Y[index] - Y_predicted[index])

    updateC = (-1 / n) * updateC
    updateC = updateC * learningRate

    return updateWeights, updateC


def scaleData(X):
    # Get Min and max values in input array
    if len(X.shape) == 1:
        ##Single Value Array as used in Y values
        min = X[0]
        max = min
        for index in range(0, len(X)):
            if X[index] < min:
                min = X[index]
            if X[index] > max:
                max = X[index]

        for index in range(0, len(X)):
            X[index] = (X[index] - min) / (max - min)
    else:
        dimensions = X.shape[1]

        for currentDim in range(0, dimensions):
            min = X[0][currentDim]
            max = min
            for index in range(0, len(X)):
                if X[index][currentDim] < min:
                    min = X[index][currentDim]
                if X[index][currentDim] > max:
                    max = X[index][currentDim]
            # X_Res = []

            for index in range(0, len(X)):
                X[index][currentDim] = (X[index][currentDim] - min) / (max - min)

    return X


def getLinearRegressionDemoData(nSamples, featureDimensions, noise):
    X, Y, coef = sklearn.datasets.make_regression(
        n_samples=nSamples,
        n_features=featureDimensions,
        n_informative=featureDimensions,
        noise=noise,
        coef=True,
        random_state=0,
    )

    return X, Y


def getLogisticRegressionDemoData(nSamples, featureDimensions):

    X, Y = sklearn.datasets.make_classification(
        n_samples=nSamples,
        n_features=featureDimensions,
        n_informative=featureDimensions,
        n_redundant=0,
        n_clusters_per_class=1,
    )

    return X, Y


def runLinearRegression(X, Y, learningRate, iterations):
    n = len(X)
    dimensions = X.shape[1]
    # dimensions = 1

    weights = np.zeros(dimensions)
    c = 0

    for iter in range(iterations):
        Y_Pred = []
        for i in range(0, n):
            currentPrediction = 0
            for dim in range(0, len(weights)):
                # print(Data[i][dim])
                currentPrediction = currentPrediction + weights[dim] * X[i][dim]
            currentPrediction = currentPrediction + c
            Y_Pred.append(currentPrediction)

        deltaWeights, deltaC = gradientDesecent(X, Y, Y_Pred, learningRate)

        c = c - deltaC
        for dim in range(len(weights)):
            weights[dim] = weights[dim] - deltaWeights[dim]

    return weights, c


def plotData(X, Y):
    from matplotlib import pyplot as plt

    fig = plt.figure()
    if X.shape[1] == 1:
        ax = plt.axes()
        ax.scatter(X, Y, color="yellowgreen", marker=".")
        # fig.show()
        return ax, fig
    elif X.shape[1] == 2:
        ax = fig.add_subplot(111, projection="3d")
        DataX = X[:, 0]
        DataY = X[:, 1]
        ax.scatter3D(DataX, DataY, Y)
        # fig.show()
        return ax, fig


def plotResults(X, Y, weights, c):
    from matplotlib import pyplot as plt

    dimensions = X.shape[1]

    if dimensions > 2:
        print("Cannot plot higher dimensional data")
        return

    ax, fig = plotData(X, Y)

    if dimensions == 1:
        ax.plot(
            [min(X), max(X)], [c + weights[0] * min(X), c + weights[0] * max(X)], "--g"
        )
        # fig.show()
    elif dimensions == 2:
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        xs = np.tile(np.arange(-3, 3), (6, 1))
        ys = np.tile(np.arange(-3, 3), (6, 1)).T
        zs = xs * weights[0] + ys * weights[1] + c
        ax.plot_surface(xs, ys, zs, alpha=0.5)
        # fig.show()


def sigmoid(x):
    exp = np.exp(-x)
    return 1 / (1 + exp)


def runLogisticRegression(X, Y, learningRate, iterations):

    n = len(X)
    dimensions = X.shape[1]

    weights = np.zeros(dimensions)
    c = 0

    for iter in range(iterations):
        Y_Pred = []
        for i in range(0, n):
            currentPrediction = 0
            for dim in range(0, len(weights)):
                currentPrediction = currentPrediction + weights[dim] * X[i][dim]
            currentPrediction = currentPrediction + c
            # Sigmoid function
            currentPrediction = sigmoid(currentPrediction)
            Y_Pred.append(currentPrediction)

        deltaWeights, deltaC = gradientDesecent(X, Y, Y_Pred, learningRate)

        c = c - deltaC
        for dim in range(len(weights)):
            weights[dim] = weights[dim] - deltaWeights[dim]
    return weights, c
