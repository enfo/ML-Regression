import sys

sys.path.append("../src")


from MLFunctions import MLRegressions as ml


def test_answer():
    assert ml.calcMSError([3], [3]) == 0
