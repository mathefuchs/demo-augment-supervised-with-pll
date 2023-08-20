""" Module for a simle kNN baseline. """

from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from supervised_and_pll.data import Datasplit


class KnnPll:
    """ Simple partial label learning with k-NN. """

    def __init__(
        self, data: Datasplit,
        n_neighbors: int = 10,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.data = data

        # Compute nearest neighbors
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn.fit(self.data.x_train)

    def _get_knn_y_pred(
        self, nn_dists: np.ndarray, nn_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        y_voting = np.zeros(
            (nn_indices.shape[0], self.data.orig_dataset.l_classes))
        for i, (nn_dist, nn_idx) in enumerate(zip(nn_dists, nn_indices)):

            if nn_dist.sum() < 1e-6:
                sims = np.ones_like(nn_dist)
            else:
                sims = 1 - nn_dist / nn_dist.sum()

            for sim, idx in zip(sims, nn_idx):
                y_neighbors = self.data.y_train[idx, :]
                y_voting[i, :] += np.where(y_neighbors == 1, sim, 0)

        y_pred = np.argmax(y_voting, axis=1)
        return y_pred, y_voting

    def get_train_pred(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the label predictions on the training set.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of predictions and voting scores.
        """

        return self._get_knn_y_pred(*self.knn.kneighbors())

    def get_test_pred(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the label predictions on the test set.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of predictions and voting scores.
        """

        return self._get_knn_y_pred(*self.knn.kneighbors(self.data.x_test))
