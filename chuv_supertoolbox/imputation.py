"""Static and temporal imputation.

(1) Static imputation (6 options)
- BasicImputation: mean, median
- StandardImputation: mice, missforest, knn
- NNImputation: gain

(2) Temporal imputation (8 options)
- BasicImputation: mean, median
- Interpolation: linear, quadratic, cubic, spline
- NNImputation: tgain, mrnn
"""

# Necessary packages
import numpy as np
import pandas as pd

# Static imputation
import sklearn.neighbors._base
import sys

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer


def rounding(data, data_imputed):
    """Use rounding for categorical variables.

    Args:
        - data: incomplete original data
        - data_imputed: complete imputed data

    Returns:
        - data_imputed: imputed data after rounding
    """
    for i in range(data.shape[1]):
        # If the feature is categorical (category < 20)  # TODO: This is questionable - arbitrary threshold!
        if len(np.unique(data[:, i])) < 20:
            # If values are integer
            if np.all(
                np.round(data[:, i][~np.isnan(data[:, i])])
                == data[:, i][~np.isnan(data[:, i])]
            ):
                # Rounding
                data_imputed[:, i] = np.round(data_imputed[:, i])

    return data_imputed


class BasicImputation:
    """CHUV toolbox imputation"""

    def __init__(self):
        pass

    def impute(self, x):
        x = x.fillna(method="bfill")
        x = x.fillna(method="ffill")
        return x

    def fit(self, dataset):
        """Compute mean or median values.

        Args:
            - dataset: incomplete dataset
        """
        pass

    def transform(self, dataset):
        """Return  imputed dataset.

        Args:
            - dataset: incomplete dataset

        Returns:
            - dataset: imputed dataset
        """
        return dataset.apply(self.impute)

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data.

        Args:
            - dataset: incomplete dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


class StandardImputation:
    """Standard imputation method for static data.

    Reference 1: https://pypi.org/project/missingpy/
    Reference 2: https://s3.amazonaws.com/assets.datacamp.com/production/course_17404/slides/chapter4.pdf

    Attributes:
        - imputation_model_name: 'mice', 'missforest', 'knn'
    """

    def __init__(
        self,
        imputation_model_name,
        n_neighbors=5,
        weights="distance",
        impute_val=None,
    ):
        # Only allow for certain options
        assert imputation_model_name in ["mice", "missforest", "knn"]
        self.imputation_model_name = imputation_model_name
        # Initialize the imputation model
        self.imputation_model = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.impute_val = impute_val

    def fit(self, dataset):
        """Train standard imputation model.

        Args:
            - dataset: incomplete dataset
        """
        # MICE
        if self.imputation_model_name == "mice":
            self.imputation_model = IterativeImputer()
        # MissForest
        elif self.imputation_model_name == "missforest":
            assert dataset.shape[1] >= 2
            self.imputation_model = MissForest()
        # KNN
        elif self.imputation_model_name == "knn":
            self.imputation_model = KNNImputer(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
            )

        self.imputation_model = self.imputation_model.fit(dataset)

        return

    def transform(self, dataset):
        """Return imputed dataset by standard imputation.

        Args:
            - dataset: incomplete dataset

        Returns:
            - dataset: imputed dataset by standard imputation.
        """
        assert self.imputation_model is not None

        # Standard imputation
        data_imputed = self.imputation_model.transform(dataset)

        # Rounding
        dataset_arr = rounding(dataset.to_numpy(), data_imputed)

        dataset_arr = pd.DataFrame(
            dataset_arr, index=dataset.index, columns=dataset.columns
        )

        return dataset_arr

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data

        Args:
            - dataset: incomplete dataset
        """
        # if it is not NaN values that we want to replace but another specific value
        if self.impute_val is not None:
            # replace all these values by nan so they can be imputed
            dataset = dataset.replace(self.impute_val, np.nan)

        # remove complete null columns as they will be removed by the imputation model
        null_cols = dataset.columns[dataset.isnull().all(axis=0)]
        non_null_cols = [x for x in dataset.columns if x not in null_cols]
        dataset = dataset[non_null_cols]

        self.fit(dataset)
        return self.transform(dataset)
