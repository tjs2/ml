import numpy as np
import sklearn as sk

from sklearn.neighbors.classification import KNeighborsClassifier


def evaluate(Xtra, ytra, Xtst, ytst, k=1, positive_label=1):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn.fit(Xtra, ytra)

    y_true = ytst
    y_pred = knn.predict(Xtst)

    return evaluate_results(y_true, y_pred, positive_label=positive_label)


def evaluate_results(y_true, y_pred, positive_label=1):
    res = y_true == y_pred
    t_gn = float(res.sum()) / len(res)
    mask = y_true != positive_label
    t_mj = float(res[mask].sum()) / len(res[mask])
    mask = y_true == positive_label
    t_mn = float(res[mask].sum()) / len(res[mask])
    # approximation used for instance reduction algorithms
    auc = (1.0 + t_mn - (1.0 - t_mj)) / 2
    return t_gn, t_mj, t_mn, auc


def auc_score(y_true, y_pred, positive_label=1):
    fp_rate, tp_rate, thresholds = sk.metrics.roc_curve(
        y_true, y_pred, pos_label=positive_label)
    return sk.metrics.auc(fp_rate, tp_rate)
