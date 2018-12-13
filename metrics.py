import numpy as np


class Metrics(object):

    @staticmethod
    def precision(ranks):
        return np.array([sum(ranks[:idx + 1]) / (idx + 1) for idx, r in enumerate(ranks) if r != 0])

    @staticmethod
    def average_precision(ranks):
        precisions = Metrics.precision(ranks)
        return np.mean(precisions)

    @staticmethod
    def interpolated_average_precision(ranks):
        precisions = Metrics.precision(ranks)
        return np.mean([np.max(precisions[idx:]) for idx, x in enumerate(precisions)])

    @staticmethod
    def interpolated_average_precision_at_n_points(ranks, n):
        precisions = Metrics.precision(ranks)
        recalls = np.arange(1, precisions.size + 1, dtype=np.float32) / precisions.size
        pr = list(zip(precisions, recalls))
        return np.mean([max([p for p, r in pr if r >= s]) for s in np.linspace(0., 1., n)])
