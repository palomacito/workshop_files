"""
Functions to plot dataframes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chuv_supertoolbox.preprocessing import get_most_common, get_sorted_val


def boxplot_sorted(df, by, column, rot=0, ascending=True, ax=None):
    """Uses the pandas boxplot() function to plot a boxplot where the medians of the values are sorted
    in ascending or descending order.

    :param df: input dataframe
    :param by: column used to group rows of the input dataframe
    :param column: column containing the values to be plotted
    :param rot: x-labels rotation (defaults to 0, i.e., no rotation)
    :param ascending: boolean flag (defaults to True), whether to sort the medians in ascending (or descending) order
    :param ax: optional matplotlib figure axes (defaults to None)
    :return: boxplot figure
    """
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col: vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values(ascending=ascending)
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return df2[meds.index].boxplot(rot=rot, return_type="axes", ax=ax)


def plot_histogram(df, col_name, scale=1.0 / 100, cutoff=30, figsize=(8, 10)):
    """Plots a histogram using the matplotlib barh() function where bars are sorted from most common to
    least common.

    :param df: input dataframe
    :param col_name: column name of the dataframe to be plotted
    :param scale: scaling factor applied on each histogram bar (defaults to 1/100)
    :param cutoff: threshold count value under which no value will be plotted
    :param figsize: plot figsize (defaults to (8,10))
    :return: None
    """
    u_mots, cnt_mots = get_most_common(
        df, col_name, cutoff, return_counts=True, verbose=True
    )
    cnt_mots = cnt_mots / scale * 100

    plt.figure(figsize=figsize)
    plt.barh(range(cnt_mots.size), cnt_mots)
    plt.yticks(range(u_mots.size), u_mots)
    plt.title(f"{col_name}")
    plt.xlabel("% of occurences")
    plt.grid()
    plt.show()
    return None


def col_boxplot(
    data,
    goal_col,
    grouping_col,
    cutoff=30,
    min_count=0,
    agg_func="median",
    ylim=None,
    figsize=(16, 6),
    ascending=False,
):
    """Plots a histogram of the goal column using the boxplot_sorted() and get_sorted_val() functions.

    :param data: input dataframe
    :param goal_col: column name of the dataframe that will be plotted
    :param grouping_col: column of the dataframe that will be used to group the dataframe rows
    :param cutoff: number of unique values that will be plotted (defaults to 30)
    :param min_count: minimum number of valid values (defaults to 0) that the column value must have in order to be returned
    :param agg_func: aggregation function used to compare the LOS among groups (defaults to median)
    :param ylim: figure's y-axis upper limit (defaults to None)
    :param figsize: plot figsize (defaults to (16,6))
    :param ascending: boolean flag (defaults to True), whether to sort the medians in ascending (or descending) order
    :return: None
    """
    u_mots = get_sorted_val(
        data,
        goal_col,
        grouping_col,
        cutoff=cutoff,
        ascending=ascending,
        min_count=min_count,
        agg_func=agg_func,
    )

    fig, ax = plt.subplots(figsize=figsize)
    boxplot_sorted(
        data[data[grouping_col].isin(u_mots)].reset_index(),
        column=goal_col,
        by=grouping_col,
        rot=90,
        ax=ax,
        ascending=ascending,
    )
    ax.set_ylim(ylim)
    ax.set_ylabel(goal_col)
    ax.set_title(f"{grouping_col} with constraint: minimum occurences={min_count}")
    return None


def plot_importance(coeffs, col_labels, thres=0.1):
    """Plot a barplot representing the importance of each feature in a sklearn prediction model (e.g. random forest).
    The importance is normalized on a [0, 1] scale.

    :param coeffs: importance coefficients
    :param col_labels: labels corresponding to the coefficients
    :param thres: threshold above which coefficients will be plotted
    :return: None
    """
    order = coeffs.argsort()[::-1]
    imp = coeffs[order] / np.max(coeffs)
    cols = col_labels[order]

    cutoff = imp > thres
    imp = imp[cutoff]
    cols = cols[cutoff]

    plt.figure(figsize=(16, 4))
    plt.bar(range(imp.size), imp)
    plt.xticks(range(imp.size), cols, rotation=90)
    plt.ylabel(f"scaled variable importance (above {thres * 100:2.2f}% only)")
    plt.grid()
    plt.show()
    return None
