""" Machine learning training helpers
"""

# Necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
import keras


def cast_to_numpy_if_necessary(x):
    """Returns a numpy array object if given a pandas dataframe or a pandas series.
    Returns the original input otherwise.

    :param x: input pandas dataframe, series, or numpy array
    :return: x casted as a numpy array
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.to_numpy()
    return x


class CrossValPredict:
    """Returns the predictions of a model performed over an entire dataset X, after
    a series of cv-fold cross-validation training and prediction rounds.

        Attributes:
            - model: keras neural network model or sklearn model
            - X: input data matrix
            - y: reference output matrix
            - cv: number of folds for cross-validation
            - batch_size: model's batch size
            - epochs: number of training epochs
            - sample_weights: optional Numpy array of weights for the training samples, used for weighting the loss function
            - verbose: verbose level, corresponds to the keras verbose level in train and predict calls
            - data_augmentation: boolean flag (defaults to False), if True, augments the training data to equalize labels distribution
            - n_predict: number of predictions steps (for uncertainty estimation with dropout layers and call with training=True)
    """

    def __init__(
        self,
        model,
        y,
        cv=6,
        batch_size=32,
        epochs=60,
        sample_weights=None,
        verbose=0,
        data_augmentation=False,
        n_predict=1,
        return_model=False,
    ):
        self.model = model
        self.y = y
        self.cv = cv
        self.batch_size = batch_size
        self.epochs = epochs
        self.sample_weights = sample_weights
        self.verbose = verbose
        self.data_augmentation = data_augmentation
        self.n_predict = n_predict
        self.return_model = return_model
        # assert self.model_name in ["sk", "nn"]

    def keras_or_tf_model(self):
        model_classes = (tf.keras.Model, tf.estimator.Estimator, keras.Model)
        return isinstance(self.model, model_classes)

    def fit(self, dataset):
        """Fit the whole pipeline.

        Args:
            - dataset: Input data for fitting
        """
        pass

    def crossval_predict_NN(self, X):
        """Returns the predictions of a keras neural network model performed over an entire dataset X, after
        a series of cv-fold cross-validation training and prediction rounds.

        :param X: input data matrix

        :return: the predicted values over the entire input dataset X, with n_predict outcomes
        """
        # safety first
        X = cast_to_numpy_if_necessary(X)
        self.y = cast_to_numpy_if_necessary(self.y)
        self.sample_weights = cast_to_numpy_if_necessary(self.sample_weights)

        print(f"Running a {self.cv}-fold cross-validation prediction")
        predictions = np.zeros(
            (X.shape[0], self.n_predict) + self.model.layers[-1].output_shape[1:]
        )
        w0 = self.model.get_weights()
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        k = 1
        for train_idx, test_idx in kf.split(X):
            print(f"iteration {k}/{self.cv}...")
            # reset model
            self.model.set_weights(w0)
            if self.sample_weights is not None:
                w_train = self.sample_weights[train_idx]
            else:
                w_train = None
            # resize classes if needed with data augmentation
            if self.data_augmentation:
                if len(self.y.shape) > 1:
                    print(
                        f"WARNING: multi-index labels detected, using max(y) to stratify samples for data augmentation"
                    )
                    labels = np.round(self.y[train_idx]).max(axis=1)
                else:
                    labels = np.round(self.y[train_idx])
                if self.sample_weights is not None:
                    class_equalizer = ClassEqualizer(
                        y=self.y[train_idx],
                        labels=labels,
                        n_scaling=1,
                        sample_weights=w_train,
                    )
                    X_train, y_train, w_train = class_equalizer.fit_transform(
                        X[train_idx]
                    )
                else:
                    class_equalizer = ClassEqualizer(
                        y=self.y[train_idx], labels=labels, n_scaling=1
                    )
                    X_train, y_train = class_equalizer.fit_transform(X[train_idx])
            else:
                X_train = X[train_idx]
                y_train = self.y[train_idx]
            if self.verbose > 0:
                validation_data = (X[test_idx], self.y[test_idx])
            else:
                validation_data = None
            # train
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_data=validation_data,
                sample_weight=w_train,
            )
            # predict
            pred = []
            for _ in range(self.n_predict):
                pred.append(
                    self.model.predict(
                        X[test_idx], batch_size=self.batch_size, verbose=self.verbose
                    )
                )
            predictions[test_idx] = np.swapaxes(np.array(pred), 0, 1)
            k += 1
        if self.return_model:
            return [self.model, predictions]
        else:
            return predictions

    def crossval_predict_sk(self, X):
        """Returns the perdictions of a sklearn model performed over an entire dataset X, after
        a series of cv-fold cross-validation training and prediction rounds.

        :param X: input data matrix

        :return: the predicted values over the entire input dataset X, with n_predict outcomes
        """
        # safety first
        X = cast_to_numpy_if_necessary(X)
        self.y = cast_to_numpy_if_necessary(self.y)
        self.sample_weights = cast_to_numpy_if_necessary(self.sample_weights)

        print(f"Running a {self.cv}-fold cross-validation prediction")
        predictions = np.zeros((X.shape[0], self.n_predict))
        params = self.model.get_params()
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        k = 1
        for train_idx, test_idx in kf.split(X):
            print(f"iteration {k}/{self.cv}...")
            # reset model
            self.model = self.model.__class__()
            self.model = self.model.set_params(**params)
            if self.sample_weights is not None:
                w_train = self.sample_weights[train_idx]
            else:
                w_train = None
            # resize classes if needed with data augmentation
            if self.data_augmentation:
                if self.sample_weights is not None:
                    class_equalizer = ClassEqualizer(
                        y=self.y[train_idx],
                        labels=np.round(self.y[train_idx]),
                        n_scaling=1,
                        sample_weights=w_train,
                    )
                    X_train, y_train, w_train = class_equalizer.fit_transform(
                        X[train_idx]
                    )
                else:
                    class_equalizer = ClassEqualizer(
                        y=self.y[train_idx],
                        labels=np.round(self.y[train_idx]),
                        n_scaling=1,
                    )
                    X_train, y_train = class_equalizer.fit_transform(X[train_idx])

            else:
                X_train = X[train_idx]
                y_train = self.y[train_idx]
            # train
            self.model.fit(X_train, y_train, sample_weight=w_train)
            # predict
            pred = []
            for _ in range(self.n_predict):
                pred.append(self.model.predict(X[test_idx]))

            # if the model returned one dimension too many (e.g (1,1296,1) instead of (1296,1))
            pred = np.array(pred)
            if pred.ndim == 3:
                pred = pred[:, :, 0]

            predictions[test_idx] = np.swapaxes(pred, 0, 1)
            k += 1
            if self.verbose > 0:
                print(
                    f"score on test set: {self.model.score(X[test_idx], self.y[test_idx]):2.2f}"
                )
        if self.return_model:
            return [self.model, predictions]
        else:
            return predictions

    def transform(self, dataset):
        if self.keras_or_tf_model():
            return self.crossval_predict_NN(dataset)
        else:
            return self.crossval_predict_sk(dataset)

    def fit_transform(self, dataset):
        """Fit the whole pipeline and apply the transform.

        Args:
            - dataset: Input data for fit and transform
        """
        self.fit(dataset)
        return self.transform(dataset)


class ClassEqualizer:
    """Returns X, y, and weights augmented with new samples such that all classes in the output (y) are equally represented.

    Attributes:
        - y: outputs matrix (target model outputs)
        - labels: label, based on y values, that will be equalized in the output
        - n_scaling: scaling ratio (defaults to 1, i.e., all classes are represented equally)
        - sample_weights: optional associated sample weights
    """

    def __init__(self, y, labels, n_scaling=1, sample_weights=None):
        self.y = y
        self.labels = labels
        self.n_scaling = n_scaling
        self.sample_weights = sample_weights

    def fit(self, dataset):
        """Fit the whole pipeline.

        Args:
            - dataset: Input data for fitting
        """
        pass

    def transform(self, dataset):
        """Returns X, y, and weights augmented with new samples such that all classes in the output (y) are equally represented.
        New samples are chosen randomly from the X matrix, as in regular bootstrapping.
        NOTE: 'y' and 'labels' are 2 different objects since y could be a float matrix, but one might still want to
        equalize some high, medium and low 'y' values.

        :param dataset: inputs data matrix (model inputs)
        :return: X, y, and weights (if provided to the function) augmented with new data
        """
        # safety first
        X = cast_to_numpy_if_necessary(dataset)
        self.y = cast_to_numpy_if_necessary(self.y)
        self.labels = cast_to_numpy_if_necessary(self.labels)
        self.sample_weights = cast_to_numpy_if_necessary(self.sample_weights)

        has_weights = self.sample_weights is not None
        X_aug = X.copy()
        y_aug = self.y.copy()
        if has_weights:
            weights_aug = self.sample_weights.copy()

        labu = np.unique(self.labels)
        for l in labu:
            ind = self.labels == l
            Nnew = int((self.y.shape[0] - np.sum(ind)) / self.n_scaling)
            idx = np.random.choice(
                np.arange(self.y.shape[0])[ind], replace=True, size=Nnew
            ).astype(np.int)
            X_aug = np.concatenate([X_aug, X[idx]])
            y_aug = np.concatenate([y_aug, self.y[idx]])
            if has_weights:
                weights_aug = np.concatenate([weights_aug, self.sample_weights[idx]])
        if has_weights:
            return X_aug, y_aug, weights_aug
        return X_aug, y_aug

    def fit_transform(self, dataset):
        """Fit the whole pipeline and apply the transform.

        Args:
            - dataset: Input data for fit and transform
        """
        self.fit(dataset)
        return self.transform(dataset)


class PipelineComposer:
    """Composing a pipeline from stages.

    Attributes:
        - *stage: individual stages in the pipeline
    """

    def __init__(self, *stage):
        self.stage = stage

    def fit(self, dataset):
        """Fit the whole pipeline.

        Args:
            - dataset: Input data for fitting
        """
        for s in self.stage:
            s.fit(dataset)

    def transform(self, dataset):
        """Use the whole pipeline to transform the data set.

        Args:
            - dataset: Input data for transform
        """
        for s in self.stage:
            dataset = s.transform(dataset)
        return dataset

    def fit_transform(self, dataset):
        """Fit the whole pipeline and apply the transform.

        Args:
            - dataset: Input data for fit and transform
        """
        for s in self.stage:
            dataset = s.fit_transform(dataset)
        return dataset
