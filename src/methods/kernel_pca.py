import numpy as np
from scipy import linalg


class Kernel_PCA():
    def __init__(self, n_components : int, kernel_fnct):
        """
        n_components: int, number of components to keep
        kernel_fn: function, kernel function
        """
        self.kernel = kernel_fnct
        self.n_components = n_components

    def fit_and_transform(self, X):
        self.X_fit = X
    
        # Compute gram
        K = self.kernel(X, X)

        # Centriation of Gram. Maybe not needed
        tmp = np.identity(K.shape[0]) - np.ones(K.shape) / K.shape[0]
        K_c = tmp @ K @ tmp
        eigenvals, eigenvecs = linalg.eigh(K_c)
        idxs = eigenvals.argsort()[::-1]
        self.eigenvals = eigenvals[idxs][:self.n_components]
        self.eigenvectors = eigenvecs[:, idxs][:, :self.n_components]
        non_zeros = np.flatnonzero(self.eigenvals)
        self.alphas = np.zeros(self.eigenvectors.shape)
        self.alphas[:, non_zeros] = np.divide(self.eigenvectors[:, non_zeros], np.sqrt(self.eigenvals[non_zeros]))

        return self.alphas

    def transform(self, X):
        # Calculate K
        K = self.kernel(X, self.X_fit)

        # Center K
        tmp0 = np.identity(K.shape[0]) - np.ones((K.shape[0], K.shape[0])) / K.shape[0]
        tmp1 = np.identity(K.shape[1]) - np.ones((K.shape[1], K.shape[1])) / K.shape[1]
        K_c = tmp0 @ K @ tmp1

        return K_c @ self.alphas
