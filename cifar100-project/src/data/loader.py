
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np


def load_pickle_data(file_path: str | Path) -> Dict[bytes, Any]:
    """Load a CIFAR pickle file and return its dictionary payload."""
    file_path = Path(file_path)
    with file_path.open("rb") as file_obj:
        return pickle.load(file_obj, encoding="bytes")


def load_features_and_labels(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened images and fine labels from a CIFAR-100 train file."""
    data = load_pickle_data(file_path)
    features = np.asarray(data[b"data"], dtype=np.uint8)
    labels = np.asarray(data[b"fine_labels"], dtype=np.int64)
    return features, labels