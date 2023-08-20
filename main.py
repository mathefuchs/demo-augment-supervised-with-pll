""" Main module. """

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

from supervised_and_pll.data import Datasplit, get_all_datasets
from supervised_and_pll.knn_pll import KnnPll

if __name__ == "__main__":
    COLUMNS = ["dataset", "knn", "logreg", "logreg_with_pll"]
    datasets = get_all_datasets()

    res = []
    for dataset_name, dataset in tqdm(sorted(datasets.items())):
        for seed in range(5):
            rng = np.random.Generator(np.random.PCG64(seed))

            # Create data split
            datasplit = Datasplit.create_random_split_from_dataset(
                dataset, rng, test_size=0.2)
            row = [dataset_name]

            # Train KNN classifier
            n_neighbors = int(math.ceil(np.log2(datasplit.x_train.shape[0])))
            knn_clf = KnnPll(datasplit, n_neighbors=n_neighbors)
            knn_test_pred = knn_clf.get_test_pred()[0]
            score = matthews_corrcoef(datasplit.y_true_test, knn_test_pred)
            row.append(score)

            # Train logistic regression classifier
            clf = LogisticRegression(
                max_iter=10000, random_state=rng.integers(1000), n_jobs=-1)
            clf.fit(datasplit.x_train, datasplit.y_true_train)
            proba_test = clf.predict_proba(datasplit.x_test)
            score = matthews_corrcoef(
                datasplit.y_true_test, np.argmax(proba_test, axis=1))
            row.append(score)

            # Improve indecisive predictions with partial label learning
            datasplit.y_test = np.where(
                (proba_test.T >= 0.9 * np.max(proba_test, axis=1)).T, 1, 0).copy()
            knn_pll = KnnPll(datasplit, n_neighbors=n_neighbors)
            score = matthews_corrcoef(
                datasplit.y_true_test, knn_pll.get_test_pred()[0])
            row.append(score)
            res.append(row)

    res_df = pd.DataFrame.from_records(res, columns=COLUMNS)
    res_df.to_csv("results.csv", index=None)
    print(res_df.describe())
