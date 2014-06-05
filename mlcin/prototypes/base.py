

class InstanceReductionMixIn(object):

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = set(y)
        self.prototypes = None
        self.prototypes_labels = None
        self.reduction_ratio = 0.0
        return self

    def reduce_data(self):
        return self.prototypes, self.prototypes_labels

    def get_prototypes(self):
        return self.prototypes, self.prototypes_labels
