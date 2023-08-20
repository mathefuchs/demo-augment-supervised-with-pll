""" Module for loading data. """

from glob import glob
from typing import Dict

from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 

UCI_DATA = list(sorted(
    glob("./ucipp/uci/*.arff")
))
UCI_DATA_LABELS = [
    path.split("/")[-1].split(".")[0] for path in UCI_DATA
]
UCI_LABEL_TO_PATH = {
    label: path for label, path in zip(UCI_DATA_LABELS, UCI_DATA)
}


class Dataset:
    """ A dataset. """

    def __init__(
        self, x_full: np.ndarray, y_full: np.ndarray, y_true: np.ndarray,
        n_samples: int, m_features: int, l_classes: int,
    ) -> None:
        self.x_full = x_full
        self.y_full = y_full
        self.y_true = y_true
        self.n_samples = n_samples
        self.m_features = m_features
        self.l_classes = l_classes


class Datasplit:
    """ A data split. """

    def __init__(
        self, x_train: np.ndarray, x_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        y_true_train: np.ndarray, y_true_test: np.ndarray,
        orig_dataset: Dataset,
        normalize: str = "minmax",
    ) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test
        self.orig_dataset = orig_dataset
        self.normalize = normalize

        if self.normalize == "minmax":
            self.x_train_min = np.min(self.x_train, axis=0)
            self.x_train_max = np.max(self.x_train, axis=0)
            self.x_train_min = np.where(
                self.x_train_min == self.x_train_max, 0, self.x_train_min)
            self.x_train_max = np.where(
                self.x_train_min == self.x_train_max, 1, self.x_train_max)
        elif self.normalize == "normal":
            self.x_train_mean = np.mean(self.x_train, axis=0)
            self.x_train_std = np.std(self.x_train, axis=0)
            self.x_train_std = np.where(
                self.x_train_std == 0, 1, self.x_train_std)

        self.x_train = self._transform(self.x_train)
        self.x_test = self._transform(self.x_test)

    def _transform(self, x_data: np.ndarray) -> np.ndarray:
        """ Normalizes the given data.

        Args:
            x_data (np.ndarray): The data.

        Returns:
            np.ndarray: The normalized data.
        """

        if self.normalize == "minmax":
            return (
                (x_data - self.x_train_min) /
                (self.x_train_max - self.x_train_min)
            )
        if self.normalize == "normal":
            return (
                (x_data - self.x_train_mean) /
                self.x_train_std
            )
        return x_data

    @classmethod
    def create_random_split_from_dataset(
        cls, dataset: Dataset, rng: np.random.Generator, test_size: float = 0.5,
    ) -> "Datasplit":
        """ Create random split from dataset.

        Args:
            dataset (Dataset): The dataset.
            seed (int): The random seed.
            test_size (float): The test size. Defaults to 0.5.

        Returns:
            Datasplit: The data split.
        """

        x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
            dataset.x_full, dataset.y_full, dataset.y_true, test_size=test_size,
            shuffle=True, random_state=rng.integers(int(1e6)),
        )
        return Datasplit(
            x_train, x_test, y_train, y_test,
            y_true_train, y_true_test, dataset,
        )


def get_all_datasets() -> Dict[str, Dataset]:
    """ Retrieves all UCI datasets.

    Returns:
        Dict[str, Dataset]: Maps names to datasets.
    """

    all_datasets = {}
    for name, path in UCI_LABEL_TO_PATH.items():
        # Load dataset
        try:
            data, meta = arff.loadarff(path)
        except:
            continue
        if len(data) < 10 or len(data) > 100000 or len(meta.names()) > 1000:
            continue

        df = pd.DataFrame.from_records(data)
        for col, type in zip(
            map(str, meta.names()),
            map(str, meta.types())
        ):
            if col == "Class":
                df["Class"] = pd.Categorical(df["Class"]).codes.astype(int)
            elif type == "nominal":
                # Onehot encode column
                onehot_df = df[col].astype(str).str.get_dummies()
                for i, onehot_col in enumerate(map(str, onehot_df)):
                    df[f"{col}_{i}"] = onehot_df[onehot_col].astype(float)
                    df = df.copy()
                df.drop(col, axis=1, inplace=True)
            elif type == "numeric":
                # Parse as float
                cols = list(df.columns)
                cols.remove(col)
                cols.append(col)
                df[col] = df[col].astype(float)
                df = df[cols]
            else:
                # Unknown type
                raise ValueError(f"Unknown column type: {type}")

        # Extract values
        x_raw = df.loc[:, df.columns != "Class"].values
        y_raw = df["Class"].values

        # Exclude zero-variance features
        x_raw = x_raw[:, x_raw.var(axis=0) >= 1e-10]

        # Extract cardinalities
        n_samples = x_raw.shape[0]
        m_features = x_raw.shape[1]
        l_classes = np.unique(y_raw).shape[0]

        # Partial label vector
        pl_vec = np.zeros((n_samples, l_classes), dtype=int)
        for i, y_val in enumerate(y_raw):
            pl_vec[i, y_val] = 1

        # Store dataset
        all_datasets[name] = Dataset(
            x_raw, pl_vec, y_raw, n_samples, m_features, l_classes,
        )

    return all_datasets
