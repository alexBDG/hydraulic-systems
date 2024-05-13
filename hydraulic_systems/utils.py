"""Contains the predictor model saver/loader."""

# System imports.
import pickle

# Local imports.
from .models import Predictor



def dump_model(model: Predictor, path: str) -> None:
    """Save the input model into pickle format."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path) -> Predictor:
    """Load a model from pickle format."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
