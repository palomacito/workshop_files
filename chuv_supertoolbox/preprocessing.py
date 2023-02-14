"""
Methods sort, get the most common element and work with soarian values (CHUV)
"""

import numpy as np
import pandas as pd
import multiprocessing as mp


def get_most_common(data, column, cutoff=30, return_counts=False, verbose=False):
    """Returns the most common values appearing in the column 'column' up to cutoff.

    :param data: input dataframe
    :param column: target column of the dataframe containing the values of interest
    :param cutoff: number of unique values to return (defaults to 30)
    :param return_counts: boolean flag (defaults to False), if True, returns the counts in addition to the values
    :param verbose: boolean flag (defaults to False), if True, prints the number of unique values found in the column
    :return: unique array of n=cutoff values ordered by frequency in the column
    """
    assert column in data.columns
    assert data is not None
    if verbose:
        print(f'{data[column].dropna().unique().size} unique "{column}" found')

    u_mots, cnt_mots = np.unique(data[column].dropna(), return_counts=True)
    ind = np.argsort(cnt_mots)
    ind = ind[-cutoff:]  # cutoff for infrequent terms
    u_mots = u_mots[ind]
    if return_counts:
        cnt_mots = cnt_mots[ind]
        return u_mots, cnt_mots
    return u_mots


def get_sorted_val(
    data,
    goal_column,
    grouper,
    cutoff=30,
    ascending=False,
    min_count=0,
    agg_func="median",
):
    """Returns the values with the highest (or lowest) LOS appearing in the column 'column'
    it is possible to set a minimum number of occurences with the variable 'min_count'.

    :param data: input dataframe
    :param grouper: column of the dataframe that will be used to group the dataframe rows
    :param cutoff: number of unique values to return (defaults to 30)
    :param ascending:  boolean flag (defaults to False), if True, returns the column values corresponding to the smallest LOS.
    :param min_count: minimum number of valid values (defaults to 0) that the column value must have in order to be returned
    :param agg_func: aggregation function used to compare the LOS among groups
    :return: unique array of n=cutoff column values ordered by LOS value (computed by 'agg_fun')
    """
    los = data.groupby(grouper).apply(lambda x: x[goal_column].agg(agg_func))
    los = los[data.groupby(grouper).size() > min_count].sort_values(ascending=ascending)
    u_mots = los.iloc[:cutoff].index
    return u_mots


def soarian_dependances_to_scores(data, finding_code, inplace=False):
    """Returns a transformed dataframe where the dependances finding identified by 'finding_code' has been
    replaced by its numerical values.
    Tested on findings: 'C2_INF_Stat1_020', 'C2_INF_Stat1_021', 'C2_INF_Stat1_022', 'C2_INF_Stat1_023',
    'C2_INF_Stat1_024', 'C2_INF_Stat_148', 'C2_INF_Stat_149', 'C2_INF_Stat_150', 'C2_INF_Stat_151',
    'C2_INF_Stat_152', 'C2_INF_Stat_154', 'C2_INF_Stat_155'

    :param data: input dataframe
    :param finding_code: code of the finding that will be replaced by a numerical value
    :param inplace: boolean flag (defaults to False), if True, performs the transformation in place and returns None
    :return: transformed dataframe where the dependances finding has been replaced by its numerical values
    """
    dep_to_score = {
        None: None,
        "Indépendant (fait seul)": 0,
        "Stimulation, surveillance (nécessite une stimulation)": 1,
        "Dépendance partielle ou importante (parfois ou toujours aidé)": 2,
        "Dépendance totale (fait par autrui)": 3,
        "0. Indépendance": 0,
        "1. Indépendance avec aide pour la préparation seulement": 0.5,
        "2. Supervision": 1,
        "3. Aide limitée": 1.5,
        "4. Aide considérable": 2,
        "5. Aide maximale": 2.5,
        "6. Dépendance totale": 3,
        "8. L'activité n'a pas eu lieu au cours de la période": None,
        "Indépendant": 0,
        "Suppléance partielle": 1,
        "Suppléance importante": 2,
        "Suppléance totale": 3,
        "Préparation/Installation": None,
    }
    ind = data.fnd_code == finding_code
    if inplace:
        data.loc[ind, "fnd_value"] = data.loc[ind, "fnd_value"].apply(
            lambda x: dep_to_score[x]
        )
        return None
    return data.loc[ind, "fnd_value"].apply(lambda x: dep_to_score[x])
