from t2f.ranking.skfeature.function.sparse_learning_based.UDFS import udfs
import numpy.random as npr

while True:
    z = npr.randint(10, 100)
    y = npr.randint(10, 100)
    X = npr.random((z, y))

    udfs(X)