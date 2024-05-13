"""Contains the main predictor model."""

# System imports.
import numpy as np
from numpy import ndarray

# Scikit-learn
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA



class Predictor(object):
    """Predictor used during inference to manage data preprocessing, prediction
    and data postprocessing.

    Parameters
    ----------
    clf : BaseEstimator
        Fitted Scikit-learn like classifier estimator with a `predict` method.
    pca_fs1 : PCA
        Fitted principal component analysis object used during the training step
        for dimensionality reduction on FS1 sensor data.
    pca_ps2 : PCA
        Fitted principal component analysis object used during the training step
        for dimensionality reduction on PS2 sensor data.
    """
    def __init__(self, clf: BaseEstimator, pca_fs1: PCA, pca_ps2: PCA):
        self.clf = clf
        self.pca_fs1 = pca_fs1
        self.pca_ps2 = pca_ps2
        self._encoder = {100: 3, 90: 2, 80: 1, 73: 0}
        self._decoder = {3: 100, 2: 90, 1: 80, 0: 73}

    def _preprocess_data(self, X_fs1: ndarray, X_ps2: ndarray) -> ndarray:
        # Features
        X = np.hstack([
            self.pca_fs1.transform(X_fs1.astype(np.float32)),
            self.pca_ps2.transform(X_ps2.astype(np.float32))
        ])
        return X

    def _postprocess_data(self, y: ndarray) -> ndarray:
        y = np.array([self._decoder[y_val] for y_val in y])
        return y

    def predict(self, X_fs1: ndarray, X_ps2: ndarray) -> ndarray:
        """Pipeline to run a prediction over new cycle(s).

        Parameters
        ----------
        X_fs1 : ndarray
            Input from the volume flow sensor FS1 (with units in l/min and a
            sampling rate of 10Hz), shape of (batch_size, 600).
        X_ps2 : ndarray
            Input from the pressure sensor PS2 (with units in bar and a
            sampling rate of 100Hz), shape of (batch_size, 6000).

        Returns
        -------
        y : ndarray
            Predicted value, shape of (batch_size, 1).
        """
        X = self._preprocess_data(X_fs1, X_ps2)
        y = self.clf.predict(X)
        y = self._postprocess_data(y)
        return y
