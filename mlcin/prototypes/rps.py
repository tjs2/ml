import numpy as np
import random
import sklearn

from .base import InstanceReductionMixIn


class RandomPrototypeSelection(InstanceReductionMixIn):

    '''
    This is just an example, this classe is not a actual
    prototype selection algorithm and should not be used
    as such. It is only for reference use.
    '''

    def __init__(self, n_samples_per_class=None):
        if n_samples_per_class:
            self.n_samples_per_class = n_samples_per_class
        else:
            self.n_samples_per_class = 1

    def reduce_data(self):
        prototypes = []
        prototypes_labels = []
        for label in self.labels:
            mask = self.y == label
            for i in range(self.n_samples_per_class):
                sample = random.choice(self.X[mask])
                prototypes = prototypes + [sample]
                prototypes_labels = prototypes_labels + [label]

        self.prototypes = np.asarray(prototypes)
        self.prototypes_labels = np.asarray(prototypes_labels)
        self.reduction_ratio = 1 - float(len(self.prototypes_labels)) / len(self.y)

        return self.prototypes, self.prototypes_labels
