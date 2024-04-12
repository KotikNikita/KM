import numpy as np

class MinMaxScaler():
    def __init__(self, feature_range=(0,1), clip=False):
        self.min_range = feature_range[0]
        self.max_range = feature_range[1]
        self.clip = clip

    def fit(self, X):
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        return

    def transform(self, X):
        X_transformed = (X - self.x_min) / (self.x_max - self.x_min)
        if self.clip:
            X_transformed = np.clip(X_transformed, self.min_range, self.max_range)
        X_transformed = X_transformed * (self.max_range - self.min_range) + self.min_range
        return X_transformed

    def fit_and_transform(self, X):
        self.fit(X)
        return self.transform(X)
