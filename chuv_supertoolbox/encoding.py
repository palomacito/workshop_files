"""
Filtering, cleaning, normalization and parallel application
"""

# Necessary packages
import pandas as pd
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed


class OneHotEncoder:
    """Transforms categorical columns into numerical one-hot encoded columns.

    :param columns: categorical columns we want to transform
    :param drop_columns: whether we want to drop the original columns 
    :param dummy_na: if we want a column that states whether the value in the column was nan
    """
    def __init__(self, columns, drop_columns=True, dummy_na=True):
        """Constructor method for the OneHotEncoder.

        :param columns: categorical columns we want to transform
        :param drop_columns: whether we want to drop the original columns 
        :param dummy_na: if we want a column that states whether the value in the column was nan
        """
        self.columns = columns
        self.drop_columns = drop_columns
        self.dummy_na = dummy_na

    def fit(self, dataset):
        """Does nothing.
        :param dataset: the dataset we fit
        """
        pass

    def transform(self, dataset):
        """Transform original dataset to one-hot encoded data.

        :param dataset: original dataframe
        :return: one-hot encoded dataframe
        """
        if self.columns is not None and not dataset.empty:
            df_cat = pd.get_dummies(
                dataset[self.columns],
                columns=self.columns,
                prefix=self.columns,
                drop_first=True,
                dummy_na=self.dummy_na,
            )
            dataset = pd.concat([dataset, df_cat], axis=1)
            if self.drop_columns:
                dataset = dataset.drop(columns=self.columns)
        return dataset

    def fit_transform(self, dataset):
        """Fit Transform original dataset to one-hot encoded data.

        :param dataset: original dataframe
        :return: one-hot encoded dataframe
        """
        self.fit(dataset)
        return self.transform(dataset)


class DuplicateIndexRemover:
    """Returns the input dataframe without any duplicated index. By security, the function keep the row index that contains the less NaN values.
    """

    def __init__(self):
        """Constructor method for the DuplicateIndexRemover.
        """
        pass

    def fit(self, dataset):
        """Fit method for the DuplicateIndexRemover.
        :param dataset: input dataframe
        """
        pass

    def transform(self, dataset):
        """Returns the input dataframe without any duplicated index.
        By security, the function keep the row index that contains the less NaN values.

        :param dataset: input dataframe
        :return: the input dataframe without duplicated indices
        """
        if dataset.index.name is not None and not dataset.empty:
            i_name = dataset.index.name  # original index
            dat = dataset.reset_index()  # reindex with unique indices

            # sort data based on number of NaN values in each row (the ones with the most at the bottom of the list)
            nna = dat.isna().sum(axis="columns").sort_values(ascending=True)

            # drop duplicated indices and keep only the 1st occurrence (i.e. the one with the lest NaN values)
            # this assumes that the pandas .loc[] operator preserve ordering (and hence avoid the usage of the
            # relatively time consuming reindex() function)
            idx = dat.loc[nna.index, i_name].drop_duplicates(keep="first").index
            return dat.loc[idx].set_index(i_name)
        else:
            return dataset

    def fit_transform(self, dataset):
        """Returns the input dataframe without any duplicated index.
        By security, the function keep the row index that contains the less NaN values.

        :param dataset: input dataframe
        :return: the input dataframe without duplicated indices
        """
        self.fit(dataset)
        return self.transform(dataset)


class ColumnsExtract:
    """Returns the reformated dataframe where keywords have been extracted and reshaped into new columns.
    Note that non matching rows are not returned, unless keep_all_rows is set to True.

    :param col_keywords:  list of tuple (column, value) or (columns, value, target_column) or (columns, list_of_values, target_column) that has to match for the data to be extracted
    :param keep_all_rows: boolean flag (defaults to False), if True, returns the same index as in the original dataframe
    """

    def __init__(self, col_keywords, keep_all_rows=False):
        """Constructor method for the ColumnsExtract.
        :param col_keywords:  list of tuple (column, value) or (columns, value, target_column) or (columns, list_of_values, target_column) that has to match for the data to be extracted
        :param keep_all_rows: boolean flag (defaults to False), if True, returns the same index as in the original dataframe
        """
        self.col_keywords = col_keywords
        self.keep_all_rows = keep_all_rows

    def fit(self, dataset):
        """Fit method for ColumnsExtract, does nothing here.
        :param dataset: input dataframe
        """
        pass

    def transform(self, dataset):
        """Returns the reformated dataframe where keywords have been extracted and reshaped into new columns.
        :param dataset:  input dataframe
        :return: transformed dataframe where keywords have been extracted and reshaped into new columns
        """
        df = pd.DataFrame()
        if not dataset.empty:
            if len(self.col_keywords[0]) == 2:
                self.col_keywords = [(c[0], c[0], c[1]) for c in self.col_keywords]
            for col, vcol, code in self.col_keywords:
                d = dataset.loc[dataset[col] == code, vcol]
                if type(vcol) == list:
                    d.columns = [f"{code}_{v}" for v in vcol]
                else:
                    d = d.rename(code)
                df = df.join(d, how="outer")
            if self.keep_all_rows:
                idx = dataset.index.difference(df.index)
                df = pd.concat([df, pd.DataFrame(index=idx)])
        return df

    def fit_transform(self, dataset):
        """Returns the reformated dataframe where keywords have been extracted and reshaped into new columns.

        :param dataset:  input dataframe
        :return: transformed dataframe where keywords have been extracted and reshaped into new columns
        """
        self.fit(dataset)
        return self.transform(dataset)


class ZNormalizer:
    """Normalize the data such that mean = 0 and std = 1
    :param thres: threshold value (default to 1e-4) under which std values will not be able to go, to avoid divisions by 0
    :param return_normalization_constants: boolean flag (defaults to False), if True, returns the normalization constants (means and stds)
    :param std: we'll use this variable to store the standard deviations
    :param means: we'll use this variable to store the means
    """

    def __init__(self, thres=1e-4, return_normalization_constants=False):
        """Constructor of ZNormalizer
        :param thres: threshold value (default to 1e-4) under which std values will not be able to go, to avoid divisions by 0
        :param return_normalization_constants: boolean flag (defaults to False), if True, returns the normalization constants (means and stds)
        """
        self.thres = thres
        self.return_normalization_constants = return_normalization_constants
        self.std = None
        self.means = None

    def fit(self, dataset):
        """Fit method for ZNormalizer, initialize standard deviations and means
        :param dataset: input dataframe
        """
        self.std = dataset.std()
        self.means = dataset.mean()
        self.std[self.std <= self.thres] = self.thres

    def transform(self, dataset):
        """Normalize dataframe
        :param dataset: input dataframe
        :return: normalized dataframe
        """
        dataset = dataset - self.means
        dataset = dataset / self.std
        if self.return_normalization_constants:
            return dataset, self.std, self.means
        return dataset

    def fit_transform(self, dataset):
        """Returns a Z-normalized dataframe (i.e., mean = 0 and std = 1) for all columns of the dataframe.
        If return_normalization_constants = True, returns the normalization constants (means and stds), this
        can be useful to reconstruct the original data.

        :param dataset: input dataframe
        :return: Z-normalized dataframe
        """
        self.fit(dataset)
        return self.transform(dataset)


class MinMaxNormalizer:
    """Normalize the data to make the range within [0, 1]."""

    def __init__(self, thres=1e-4, return_normalization_constants=False):
        self.thres = thres
        self.return_normalization_constants = return_normalization_constants
        self.mins = None
        self.maxs = None

    def fit(self, dataset):
        self.mins = dataset.min()
        self.maxs = dataset.max()
        self.maxs[self.maxs <= self.thres] = self.thres

    def transform(self, dataset):
        """Returns a minmax-normalized dataframe (i.e., min = 0 and max = 1) for all columns of the dataframe.
        If return_normalization_constants = True, returns the normalization constants (mins and maxs), this
        can be useful to reconstruct the original data.

        :param data: input dataframe
        :param thres: threshold value (default to 1e-4) under which std values will not be able to go, to avoid divisions by 0
        :param return_normalization_constants: boolean flag (defaults to False), if True, returns the normalization constants (mins and maxs)
        :return: minmax-normalized dataframe
        """

        dataset = dataset - self.mins
        dataset = dataset / self.maxs

        if self.return_normalization_constants:
            return dataset, self.mins, self.maxs
        return dataset

    def fit_transform(self, dataset):
        """Transform original dataset to MinMax normalized dataset.

        Args:
            - dataset: original PandasDataset

        Returns:
            - dataset: normalized PandasDataset
            - norm_parameters: normalization parameters for renomalization
        """
        self.fit(dataset)
        return self.transform(dataset)


class CategoryCleaner:
    """Removal of categorical data not present in the top thres percentile of the data.
    :param columns: columns containing the categorical data and that will be transformed
    :param thres: percentage of the data that will be represented by original categories, before using the 'Other' category label (defaults to 80%)
    :param top_n: the top number of categories we want to keep in the dataset
    :param verbose: boolean flag (defaults to False), if True, prints the categories distribution and which ones are kept after cleanup
    """

    def __init__(self, columns, thres=None, top_n=None, verbose=False):
        self.columns = columns
        self.thres = thres
        self.verbose = verbose
        self.top_n = top_n

    def fit(self, dataset):
        pass

    def transform(self, dataset_):
        """Inplace removal of categorical data not present in the top thres percentile of the data. Discarded
        categories are indicated with the 'Other' label.

        :param dataset: input dataframe
        :return: transformed dataframe
        """
        dataset = dataset_.copy()
        if self.columns is not None and not dataset.empty:
            for column in self.columns:
                prevalence = (
                    dataset.groupby(column).size().sort_values(ascending=False)
                    / dataset[column].dropna().shape[0]
                )
                if self.verbose:
                    print("categories and prevalence:")
                    print(prevalence * 100)
                top = []
                prev = 0
                k = 0
                # if we're working with a threshold
                if self.thres is not None and self.top_n is None:
                    while prev < self.thres:
                        top.append(prevalence.index[k])
                        prev += prevalence[k]
                        k += 1
                # else if we want the top n categories
                else:
                    top = (
                        dataset.groupby(column)
                        .size()
                        .sort_values(ascending=False)[: self.top_n]
                        .index
                    )
                if self.verbose:
                    print("categories kept after cleanup:")
                    print(top)
                # data replacement
                na_idx = dataset[column].isna()
                dataset.loc[np.logical_not(dataset[column].isin(top)), column] = "Other"
                dataset.loc[na_idx, column] = np.nan
        return dataset

    def fit_transform(self, dataset):
        """Transform original dataset to MinMax normalized dataset.

        Args:
            - dataset: original PandasDataset

        Returns:
            - dataset: normalized PandasDataset
            - norm_parameters: normalization parameters for renomalization
        """
        self.fit(dataset)
        return self.transform(dataset)


class ParallelApply:
    """Applies a function func to the entire dataframe using multiple threads."""

    def __init__(
        self, func, params=None, axis=None, n_cores=None, grp=None, keep_index=False
    ):
        self.func = func
        self.params = params
        self.n_cores = n_cores
        self.grp = grp
        self.keep_index = keep_index
        self.axis = axis

    def apply_f(self, *arg):

        if len(arg) > 1:
            return arg[0].apply(self.func, args=arg[1:], axis=self.axis)
        else:
            return arg[0].apply(self.func, axis=self.axis)

    def fit(self, dataset):

        if self.n_cores is None:
            # if the dataset is grouped
            if self.grp:
                self.n_cores = int(np.minimum(mp.cpu_count(), len(dataset.groups)))
            else:
                self.n_cores = int(np.minimum(mp.cpu_count(), dataset.shape[0]))
            # make sure n_cores is at least 1
            self.n_cores = int(np.maximum(1, self.n_cores))

    def parallel_apply(self, dataset):
        """Applies a function func to the entire dataframe using multiple threads.

        :param df: input dataframe
        :param func: function to be applied
        :param n_cores: number of cores to use for computation
        :return: transformed dataframe
        """

        df_split = np.array_split(dataset, self.n_cores)
        pool = mp.Pool(self.n_cores)

        if self.params is None or (
            (type(self.params) == tuple or type(self.params) == list)
            and len(self.params) == 0
        ):
            if self.axis is not None:
                dataset = pd.concat(pool.map(self.apply_f, df_split))
            else:
                dataset = pd.concat(pool.map(self.func, df_split))
            # if we want to apply the function to each column: we aggregate the results of each core
            if self.axis == 0:
                dataset = dataset.groupby(dataset.index).agg(self.func)
        else:
            params_list = [
                (x, *tuple(self.params))
                if type(self.params) == tuple or type(self.params) == list
                else (x, self.params)
                for x in df_split
            ]
            if self.axis is not None:
                dataset = pd.concat(pool.starmap(self.apply_f, params_list))
            else:
                dataset = pd.concat(pool.starmap(self.func, params_list))
            # if we want to apply the function to each column: we aggregate the results of each core
            if self.axis == 0:
                dataset = dataset.groupby(dataset.index).agg(self.func)
        pool.close()
        pool.join()
        return dataset

    def parallel_grp_apply(self, dataset):
        """Applies a function func to the entire grouped dataframe using multiple threads.

        :param df_grouped: groupby pandas dataframe
        :param func: function to be applied
        :param keep_index: boolean flag, defaults to False, if True, the groupby index column is returned
        :param n_cores: number of cores to use for computation
        :return: transformed dataframe
        """

        if self.params is None or (
            (type(self.params) == tuple or type(self.params) == list)
            and len(self.params) == 0
        ):
            if self.axis is not None:
                ret_lst = Parallel(n_jobs=self.n_cores)(
                    delayed(self.apply_f)(group) for name, group in dataset
                )
            else:
                ret_lst = Parallel(n_jobs=self.n_cores)(
                    delayed(self.func)(group) for name, group in dataset
                )
        else:

            if type(self.params) == tuple or type(self.params) == list:
                if self.axis is not None:
                    ret_lst = Parallel(n_jobs=self.n_cores)(
                        delayed(self.apply_f)(group, *self.params)
                        for name, group in dataset
                    )
                else:
                    ret_lst = Parallel(n_jobs=self.n_cores)(
                        delayed(self.func)(group, *self.params)
                        for name, group in dataset
                    )
            else:
                if self.axis is not None:
                    ret_lst = Parallel(n_jobs=self.n_cores)(
                        delayed(self.apply_f)(group, self.params)
                        for name, group in dataset
                    )
                else:
                    ret_lst = Parallel(n_jobs=self.n_cores)(
                        delayed(self.func)(group, self.params)
                        for name, group in dataset
                    )
        if self.keep_index:
            grp_idx = [name for name in dataset.groups]
            grp_idx_name = dataset.head(1).index.name
            for r, g in zip(ret_lst, grp_idx):
                r[grp_idx_name] = g

        res = pd.concat(ret_lst)
        # if we want to apply the function to each column: we aggregate the results of each core
        if self.axis == 0:
            res = res.groupby(res.index).agg(self.func)

        if self.keep_index:
            index_0 = res.index.name
            res = res.reset_index().set_index([grp_idx_name, index_0])

        return res

    def transform(self, dataset):
        """Applies a function func to the entire dataframe using multiple threads.
        Note: for grouped dataframe, uses the function parallel_grp_apply.
        :param dataset: input dataframe
        :return: transformed dataframe
        """
        if self.grp:
            return self.parallel_grp_apply(dataset)
        else:
            return self.parallel_apply(dataset)

    def fit_transform(self, dataset):
        """Applies a function func to the entire dataframe using multiple threads.
        Note: for grouped dataframe, uses the function parallel_grp_apply.
        :param dataset: input dataframe
        :return: transformed dataframe
        """
        self.fit(dataset)
        return self.transform(dataset)


# class ParallelApply:
#     """Applies a function func to the entire dataframe using multiple threads."""

#     def __init__(self, func, params=None, n_cores=None, grp=None, keep_index=False):
#         self.func = func
#         self.params = params
#         self.n_cores = n_cores
#         self.grp = grp
#         self.keep_index = keep_index

#     def fit(self, dataset):

#         if self.n_cores is None:
#             # if the dataset is grouped
#             if self.grp:
#                 self.n_cores = int(np.minimum(mp.cpu_count(), len(dataset.groups)))
#             else:
#                 self.n_cores = int(np.minimum(mp.cpu_count(), dataset.shape[0]))
#             # make sure n_cores is at least 1
#             self.n_cores = int(np.maximum(1, self.n_cores))

#     def parallel_apply(self, dataset):
#         """Applies a function func to the entire dataframe using multiple threads.

#         :param df: input dataframe
#         :param func: function to be applied
#         :param n_cores: number of cores to use for computation
#         :return: transformed dataframe
#         """

#         df_split = np.array_split(dataset, self.n_cores)
#         pool = mp.Pool(self.n_cores)
#         if self.params:
#             params_list = [
#                 (x, *tuple(self.params))
#                 if type(self.params) == tuple or type(self.params) == list
#                 else (x, self.params)
#                 for x in df_split
#             ]
#             dataset = pd.concat(pool.starmap(self.func, params_list))

#         else:
#             dataset = pd.concat(pool.map(self.func, df_split))
#         pool.close()
#         pool.join()
#         return dataset

#     def parallel_grp_apply(self, dataset):
#         """Applies a function func to the entire grouped dataframe using multiple threads.

#         :param df_grouped: groupby pandas dataframe
#         :param func: function to be applied
#         :param keep_index: boolean flag, defaults to False, if True, the groupby index column is returned
#         :param n_cores: number of cores to use for computation
#         :return: transformed dataframe
#         """
#         if self.params:
#             if type(self.params) == tuple or type(self.params) == list:
#                 ret_lst = Parallel(n_jobs=self.n_cores)(
#                     delayed(self.func)(group, *self.params) for name, group in dataset
#                 )
#             else:
#                 ret_lst = Parallel(n_jobs=self.n_cores)(
#                     delayed(self.func)(group, self.params) for name, group in dataset
#                 )
#         else:
#             ret_lst = Parallel(n_jobs=self.n_cores)(
#                 delayed(self.func)(group) for name, group in dataset
#             )

#         if self.keep_index:
#             grp_idx = [name for name in dataset.groups]
#             grp_idx_name = dataset.head(1).index.name
#             for r, g in zip(ret_lst, grp_idx):
#                 r[grp_idx_name] = g

#         res = pd.concat(ret_lst)

#         if self.keep_index:
#             index_0 = res.index.name
#             res = res.reset_index().set_index([grp_idx_name, index_0])

#         return res

#     def transform(self, dataset):
#         """Applies a function func to the entire dataframe using multiple threads.
#         Note: for grouped dataframe, uses the function parallel_grp_apply.
#         :param dataset: input dataframe
#         :return: transformed dataframe
#         """
#         if self.grp:
#             return self.parallel_grp_apply(dataset)
#         else:
#             return self.parallel_apply(dataset)

#     def fit_transform(self, dataset):
#         """Applies a function func to the entire dataframe using multiple threads.
#         Note: for grouped dataframe, uses the function parallel_grp_apply.
#         :param dataset: input dataframe
#         :return: transformed dataframe
#         """
#         self.fit(dataset)
#         return self.transform(dataset)
