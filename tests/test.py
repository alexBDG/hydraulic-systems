"""Testing the hydraulic_systems package methods."""

# System imports.
import os
import sys
import unittest
import numpy as np

# Scikit-Learn imports.
from sklearn.decomposition import PCA

# LGBM imports.
from lightgbm import LGBMClassifier

# Package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hydraulic_systems.models import Predictor
from hydraulic_systems.models import dump_model
from hydraulic_systems.models import load_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")



def generate_data(size):
    # Generate data
    X_fs1 = np.random.random_sample((size, 600))
    X_ps2 = np.random.random_sample((size, 6000))
    y = np.random.choice(4, size)
    return X_fs1, X_ps2, y


class TestPredictor(unittest.TestCase):
    """Test the `Predictor` model."""

    def test_model_creation_and_saving(self):
        """Test the procedure to create a Predictor model."""
        X_fs1, X_ps2, y = generate_data(size=100)

        # Preprocess data
        pca_fs1 = PCA(2).fit(X_fs1)
        pca_ps2 = PCA(2).fit(X_ps2)
        X = np.hstack([pca_fs1.transform(X_fs1), pca_ps2.transform(X_ps2)])
        clf = LGBMClassifier(random_state=0, verbose=-1)
        clf.fit(X, y)

        # Save the model
        model = Predictor(
            clf=clf,
            pca_fs1=pca_fs1,
            pca_ps2=pca_ps2,
        )
        model_path = os.path.join(DATA_PATH, "tmp.bin")
        dump_model(model, model_path)
        self.assertTrue(os.path.exists(model_path))
        os.remove(model_path)

    def test_model_loading(self):
        """Test the procedure to load a Predictor model."""
        X_fs1, X_ps2, _ = generate_data(size=10)

        # Load the model
        model_path = os.path.join(DATA_PATH, "model.bin")
        model = load_model(model_path)
        y = model.predict(X_fs1=X_fs1, X_ps2=X_ps2)
        for y_val in y:
            self.assertIn(y_val, [73, 80, 90, 100])



if __name__ == "__main__":
    unittest.main()