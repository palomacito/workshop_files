"""
Extract statistically relevant features from a complex dataset.
"""


from chuv_supertoolbox.timedevents import extract_timed_data
from chuv_supertoolbox.encoding import DuplicateIndexRemover
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def select_discriminative_feature(
    data,
    target_col,
    value_col,
    lib_col,
    time_col,
    y_col,
    idx,
    count_thres=200,
    fname=None,
    data_type="numerical",
    plot=False,
):
    """Writes and if necessary plots features from the input dataframe that are statistically discriminant for a
    given y (label) target value. This function relies on the extract_timed_data() method to reshape the original
    dataframe. Currently supported types of data are:
    1. categorical (t-test with alpha=5%)
    2. numerical (absolute correlation test)
    Examples::

      select_discriminative_feature(df_mol, target_col='code_test_molis', value_col='resultat_numerique',
                      lib_col='lib_test_molis', time_col='date_impression_rapport', y_col='duree_sejour',
                      data_type='numerical', fname='feature_selection/features_molis.csv')

      select_discriminative_feature(df_soa, target_col='fnd_code', value_col='fnd_value',
                      lib_col='fnd_libelle', time_col='obsv_dt', y_col='duree_sejour', data_type='categorical',
                      fname='feature_selection/features_soarian.csv')

    :param data: input dataframe
    :param target_col: column indexing the rows of interest 'value_col' (e.g. 'code_test_molis' or 'fnd_code')
    :param value_col: column containing the result values that will be used for statistical testing
    :param lib_col: column containing the text labels corresponding to the 'target_col' column
    :param time_col: column containing the time information
    :param y_col: column containing the target values (labels) that will be used to perform the statistical tests
    :param idx: column indexing the dataframe
    :param count_thres: minimal number of valid measurements required in order to keep the feature
    :param fname: filename in which the results will be printed (defaults to None, i.e., no file output is printed)
    :param data_type: string defining the type of data to be analyzed (supported are currently 'numerical and 'categorical')
    :param plot: boolean flag (default to False), if True, a plot of each significant features is plotted
    :return: the filename in which the output (list of selected features) was written
    """

    if data_type not in ["numerical", "categorical"]:
        raise NameError(f"Unknown data_type: {data_type}")

    # extract codes that have count above the threshold
    ind = data.groupby(target_col)[value_col].count() >= count_thres
    codes = list(ind[ind].index)

    # remove outliers (stays longer than 20 days)
    data = data[
        data[y_col] <= 20
    ]  # TODO: remove this line that has nothing to do here and move it back in the notebooks

    y = data.set_index(idx)[y_col]
    duplicate_remover = DuplicateIndexRemover()
    y = duplicate_remover.fit_transform(y).dropna()

    if fname:
        fid = open(fname, "w", buffering=1)
        fid.write('"code","label","label_value","test_result","pvalue"\n')

    for c in codes:
        if data_type == "categorical":
            res = data.loc[data[target_col] == c, value_col].dropna()
            vals = res.unique()
            sz = res.size
            # check: if the total number of different labels represents more than 5% of the total number of labels
            # we assume that the data corresponding to this code is not categorical and continue to next code
            if vals.size / sz > 0.05:
                continue

        dat = extract_timed_data(
            data.reset_index(),
            idx,
            target_col,
            [c],
            value_col,
            time_col,
            n_iter=-1,
        ).dropna()
        dat = dat.join(y, how="inner")
        test_result = None

        if data_type == "categorical":
            for v in dat[f"{c}_00"].unique():
                ind = dat[f"{c}_00"] == v
                # make sure there are enough individual in both groups
                if np.sum(ind) >= count_thres and np.sum(~ind) >= count_thres:
                    r1 = dat.loc[ind, y_col]
                    r2 = dat.loc[~ind, y_col]
                    tval, pval = stats.ttest_ind(
                        r1, r2, nan_policy="raise"
                    )  # equal_var=False (for Welch’s t-test)
                    if pval < 0.05:
                        test_result = {
                            "code": c,
                            "label": data.loc[data[target_col] == c, lib_col].iloc[0],
                            "label_value": v,
                            "test_result": tval,
                            "pvalue": pval,
                        }
                        if plot:
                            plt.figure()
                            plt.boxplot([r1, r2])
                            plt.xticks(np.arange(1, 3), ["yes", "no"])
                            plt.ylabel(y_col)
                            plt.title(
                                f"{data.loc[data[target_col] == c, lib_col].iloc[0]} = {v}"
                                + f" | p={pval:2.6f}  t={tval:2.4f}"
                            )
                            plt.grid()
                            plt.show()

        elif data_type == "numerical":
            # perform the test in a try/except block to avoid errors in case the data is not numeric
            try:
                dat = dat.astype(np.float)
                # TODO implement more advanced selection method (e.g. mutual information)
                # mi = mutual_info_regression(dat[[f'{c}_00']], dat[y_col])
                corr = dat.corr().values[1, 0]
            except:
                print(
                    f'WARNING: "{c}" cannot be tested, its value is probably not numeric and it will be discarded.'
                )
                continue
            if np.abs(corr) > 0.1:
                test_result = {
                    "code": c,
                    "label": data.loc[data[target_col] == c, lib_col].iloc[0],
                    "label_value": np.nan,
                    "test_result": corr,
                    "pvalue": np.nan,
                }
                if plot:
                    plt.figure()
                    plt.scatter(dat[[f"{c}_00"]], dat[y_col])
                    plt.xlabel(c)
                    plt.ylabel(y_col)
                    plt.title(f"{c} | corr={corr:2.6f}")
                    plt.grid()
                    plt.show()

        # Write results to file
        if fname and test_result:
            fid.write(",".join(f'"{x}"' for x in test_result.values()) + "\n")

    if fname:
        fid.close()
    return fname
