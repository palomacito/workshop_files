"""
Functions to preprocess data
that contains temporal information
"""

import numpy as np
import pandas as pd
import holidays
import datetime
from functools import partial
from chuv_supertoolbox.encoding import ParallelApply, DuplicateIndexRemover


def get_first_of(data, grouper, date_col, dropna=False):
    """Returns the 1st occurrence of a measurement within a group (defined by the 'grouper').
    Time is computed based on the 'date_col' data.
    If dropna is True, all values in a group that contain at least one NaN are discarded. This can be useful
    to retain for instance only the first valid (i.e. non-NaN) occurrence of an event.

    :param data: input dataframe
    :param grouper: column used to regroup rows
    :param date_col: column containing the time information
    :param dropna: boolean flag (defaults to False) that, if True, drops all value containing at least a NaN value
    :return: the 1st occurrence of each event, indexed by group
    """

    def get_date_min(x):

        if dropna:
            return x[x[date_col] == x.dropna()[date_col].min()]
        return x[x[date_col] == x[date_col].min()]

    parallel = ParallelApply(func=get_date_min, grp=True)
    res = parallel.fit_transform(data.groupby(grouper))
    res = res.set_index(grouper)
    if res.index.duplicated().sum() > 0:
        print(f"WARNING: duplicated indices in the data (indexed by {grouper})")
    return res


def get_last_of(data, grouper, date_col, dropna=False):
    """Returns the last occurrence of a measurement within a group (defined by the 'grouper').
    Time is computed based on the 'date_col' data.
    If dropna is True, all values in a group that contain at least one NaN are discarded. This can be useful
    to retain for instance only the last valid (i.e. non-NaN) occurrence of an event.

    :param data: input dataframe
    :param grouper: column used to regroup rows
    :param date_col: column containing the time information
    :param dropna: boolean flag (defaults to False) that, if True, drops all value containing at least a NaN value
    :return: the last occurrence of each event, indexed by group
    """

    def get_date_max(x):

        if dropna:
            return x[x[date_col] == x.dropna()[date_col].max()]
        return x[x[date_col] == x[date_col].max()]

    parallel = ParallelApply(func=get_date_max, grp=True)
    res = parallel.fit_transform(data.groupby(grouper))
    res = res.set_index(grouper)
    if res.index.duplicated().sum() > 0:
        print(f"WARNING: duplicated indices in the data (indexed by {grouper})")
    return res


def tryget(series, idx):
    """Safe method to return an indexed value from a pandas series object

    :param series: input pandas series
    :param idx: index
    :return: the series value at index 'idx' or NaN if the index does not exist
    """
    try:
        return series.iloc[idx]
    except:
        return np.nan


# TODO change from 0 to 1 for first ?
def get_nth_of(data, grouper, date_col, n, dropna=False):
    """Returns the nth occurrence of a measurement within a group (defined by the 'grouper').
    Time is computed based on the 'date_col' data.
    If dropna is True, all values in a group that contain at least one NaN are discarded. This can be useful
    to retain for instance only the nth valid (i.e. non-NaN) occurrence of an event.

    :param data: input dataframe
    :param grouper: column used to regroup rows
    :param date_col: column containing the time information
    :param n: nth index of occurrence
    :param dropna: boolean flag (defaults to False) that, if True, drops all value containing at least a NaN value
    :return: the nth occurrence of each event, indexed by group
    """
    assert n >= 0

    def get_date_sorted(x):

        if dropna:
            return x[x[date_col] == tryget(x.dropna()[date_col].sort_values(), n)]
        return x[x[date_col] == tryget(x[date_col].sort_values(), n)]

    parallel = ParallelApply(func=get_date_sorted, grp=True)
    res = parallel.fit_transform(data.groupby(grouper))
    res = res.set_index(grouper)
    if res.index.duplicated().sum() > 0:
        print(f"WARNING: duplicated indices in the data (indexed by {grouper})")
    return res


def extract_timed_data(
    data,
    grouper,
    target_col,
    target_col_values,
    res_col,
    date_col,
    n_iter=1,
    return_time=False,
    textsearch=False,
):
    """Returns a table ordered chronologically and containing as columns the 'target_col_values' that appear in the
    original 'target_col' column. The data goes back n_iter steps in the past. Columns are labelled with a
    '_XX' suffix, where XX corresponds to the iteration.
    Example::

      extract_timed_data(df_mol, 'numero_sejour', 'code_test_molis', mol_tests, 'resultat_numerique',
                         'date_impression_rapport', n_iter=4, return_time=True, textsearch=False)

    :param data: input dataframe
    :param grouper: column used to regroup rows
    :param target_col: column containing the codes of target values 'target_col_values' (e.g. 'code_test_molis' or 'fnd_code')
    :param target_col_values: list of values for 'target_col' that will serve as columns for the returned dataframe
    :param res_col: column containing the result values that will be used to fill the rows
    :param date_col: column containing the time information
    :param n_iter: number of iterations to retrieve [0..n] (0 for the 1st, -1 for the last)
    :param return_time: boolean flag (default False) if True, adds an additional column to the output with the timestep of the measure, taken from 'date_col'
    :param textsearch: boolean flag (default False) if True, 'res_col' values will be kept if 'target_col_values' is found in the text of the 'target_col'. If False, 'target_col' has to match exactly with the 'target_col_values'
    :return: new dataframe ordered chronologically and containing as columns the 'target_col_values' that appear in the original 'target_col' column
    """
    if not (
        grouper in data.columns
        and target_col in data.columns
        and date_col in data.columns
        and res_col in data.columns
    ):
        raise ValueError("One of the given columns is not present in the dataframe.")

    if data[[grouper, target_col, date_col]].isnull().values.any():
        raise ValueError(
            "The target, date and grouper column should not contain any null values."
        )

    target_col_values = list(target_col_values)  # safety first
    return_last_only = False
    if n_iter == -1:
        return_last_only = True
        n_iter = 1
    l0 = len(target_col_values)
    res = pd.DataFrame([])
    cols = [res_col, date_col, grouper]
    for k, t in enumerate(target_col_values * n_iter):
        l = k // l0
        if textsearch:
            # make sure we are dealing with a string column
            assert data[target_col].dtypes == object
            ind = data[target_col].apply(lambda x: x.find(t) == 0)
        else:
            ind = data[target_col] == t
        if return_last_only:
            r = get_last_of(data.loc[ind, cols], grouper, date_col, dropna=True)
        elif l == 0:
            r = get_first_of(data.loc[ind, cols], grouper, date_col, dropna=True)
        else:
            r = get_nth_of(data.loc[ind, cols], grouper, date_col, l)
        remover = DuplicateIndexRemover()
        r = remover.fit_transform(r)
        res = pd.concat(
            [res, r[res_col].rename(str(t) + f"_{l:02d}")],
            axis=1,
            join="outer",
            sort=True,
        )
        if return_time:
            res = pd.concat(
                [res, r[date_col].rename(str(t) + f"_time_{l:02d}")],
                axis=1,
                join="outer",
                sort=True,
            )
    return res


def get_previous_stays(data, ipp, start_stay_date, suppl_cols=None):
    """Returns the previous stays in chronological order
       NOTE: The input data has to be indexed with the column 'numero_sejour'

    :param data: input dataframe, indexed by 'numero_sejour'
    :param ipp: patient's ipp of interest
    :param start_stay_date: date of interest from which previous stays will be computed
    :param suppl_cols: columns to include in the returned dataframe, in addition to 'date_sortie_sejour'
    :return: dataframe containing the rows corresponding to the previous stays
    """
    if suppl_cols is None:
        suppl_cols = []
    dates = data.loc[
        data.ipp == ipp, ["date_sortie_sejour"] + suppl_cols
    ].drop_duplicates()
    idx = dates.date_sortie_sejour < start_stay_date
    return dates.loc[idx]


def compute_nb_events(data, col_start, col_end, t_now):
    """Return the number of occurrences of an event between two dates.
    Example of usage: count the number of beds occupied using the date of admission and date of discharge::

      df['date_sortie_sejour'].apply(
              partial(compute_nb_events, df, 'date_entree_sejour', 'date_sortie_sejour')).rename('nbr_lits')

    :param data: input dataframe
    :param col_start: column of the dataframe containing the events start date (e.g. date_entree_sejour)
    :param col_end: column of the dataframe containing the events start date (e.g. date_sortie_sejour)
    :param t_now: reference time from which occurrences are computed
    :return: number of event occurrences
    """
    ind = np.logical_and(data[col_end] >= t_now, data[col_start] <= t_now)
    return np.sum(ind)


def days_to_next_holiday(date_0):
    """Returns the number of days remaining until a holiday in the Canton of Vaud from a given date.

    :param date_0: start date
    :return: number of days until the next holiday in the Canton of Vaud
    """
    for k in range(1, 365):
        date = date_0 + pd.to_timedelta(f"{k} days")
        if date in holidays.Switzerland(prov="VD", years=date.year):
            return k
    print(f"WARNING: no next holiday was found from starting date: {date_0}")
    return np.nan


def compute_dow_month_holiday(data, date_column):
    """Returns a dataframe where 5 new columns are added:
    1. weekday: day of the week number [0 = monday, ..., 6 = sunday]
    2. month: month of the year [1 = janvier, ..., 12 = december]
    3. holiday: holiday indicator [0 = business day, 1 = holiday]
    4. vacation: school vacation indicator for the canton of vaud [0 = school day, 1 = vacation day]
    5. days_to_next_holiday: number of days left until next holiday in the Canton of Vaud

    source vacances: https://www.feiertagskalender.ch/ferien.php?geo=2283&jahr=2018&klasse=0&hl=fr

    :param data: input dataframe
    :param date_column: column containing the time information
    :return: dataframe with the 5 new coumns: weekday, month, holiday, vacation, and days_to_next_holiday
    """
    vacances = [
        ["2017-12-23", "2018-01-07"],
        ["2018-02-17", "2018-02-25"],
        ["2018-03-30", "2018-04-15"],
        ["2018-05-10", "2018-05-13"],
        ["2018-05-19", "2018-05-21"],
        ["2018-07-07", "2018-08-26"],
        ["2018-09-15", "2018-09-17"],
        ["2018-10-13", "2018-10-28"],
        ["2018-12-22", "2019-01-06"],
        ["2019-02-23", "2019-03-03"],
        ["2019-04-13", "2019-04-28"],
        ["2019-05-30", "2019-06-02"],
        ["2019-06-08", "2019-06-10"],
        ["2019-09-14", "2019-09-16"],
        ["2019-07-06", "2019-08-25"],
        ["2019-10-12", "2019-10-27"],
        ["2019-12-21", "2020-01-05"],
        ["2020-02-15", "2020-02-23"],
        ["2020-03-16", "2020-04-30"],
    ]
    # compute vacation dataframe
    df_vacation = pd.DataFrame()
    for k, v in enumerate(vacances):
        df_vacation.loc[k, "start"] = pd.to_datetime(v[0])
        df_vacation.loc[k, "stop"] = pd.to_datetime(v[1])

    res = pd.DataFrame(
        [], columns=["weekday", "month", "holiday", "days_to_next_holiday"]
    )
    res["weekday"] = data[date_column].apply(lambda x: x.weekday())
    res["month"] = data[date_column].apply(lambda x: x.month)
    res["holiday"] = data[date_column].apply(
        lambda x: x in holidays.Switzerland(prov="VD", years=x.year)
    )
    res["vacation"] = data[date_column].apply(
        lambda x: compute_nb_events(df_vacation, "start", "stop", x)
    )
    res["days_to_next_holiday"] = data[date_column].apply(
        lambda x: days_to_next_holiday(x)
    )
    return res


def compute_time_of_day(data, date_column, circular=False):
    """Returns a new dataframe containing the time of day.
    If circular = False, the returned dataframe contains a single 'h' column containing the time of day h = [0..24].
    If circular = True, the returned dataframe contains 2 columns 'h_cos' and 'h_sin' containing the coordinates
    similar to the tip of the hour hand on a clock. This representation makes sense for cyclic events.

    :param data: input dataframe
    :param date_column: column containing the time information
    :param circular: boolean flag (defaults to False), if True, return the time in circular representation
    :return: dataframe with the 5 new coumns: weekday, month, holiday, vacation, and days_to_next_holiday
    """
    if circular:
        res = pd.DataFrame([], columns=["h_cos", "h_sin"])
        res["h_cos"] = data[date_column].apply(
            lambda x: np.cos(2 * np.pi * (x.hour + x.minute / 60) / 24)
        )
        res["h_sin"] = data[date_column].apply(
            lambda x: np.sin(2 * np.pi * (x.hour + x.minute / 60) / 24)
        )
    else:
        res = pd.DataFrame([], columns=["h"])
        res["h"] = data[date_column].apply(lambda x: x.hour + x.minute / 60)
    return res


def remove_time_from_datetime(date):
    """Returns a datetime object that contains only date information and no time (i.e. hours, minutes, seconds).

    :param date: input datetime
    :return: datetime striped from hours, minutes, and seconds
    """
    return datetime.datetime(date.year, date.month, date.day)


def get_idxmin(t, t0, forward):
    """Returns the index of the timestamp t closest to the reference t0.

    :param t: timestamps dataframe
    :param t0: reference timestamp
    :param forward: boolean flag, if False, only timestamps smaller than t0 are considered
    :return: index corresponding to the timestamp closet to t0, or None if no data is found to match
    """
    dt = t0 - t
    if forward:
        # allow measurements taken in the future (w.r.t t0) to be considered
        return dt.abs().idxmin()
    ind = dt >= pd.to_timedelta("0 hours")
    if ind.sum() > 0:
        return dt[ind].idxmin()
    return None


def get_nearest_value(
    data, grouper, timecol, timestamp, dt_max=pd.to_timedelta("1 hour"), forward=True
):
    """Returns a DataFrame containing the closest row in the data dataframe with respect to the provided timsetamp.
    Using a grouper allows the function to return multiple rows corresponding to the desired groups.
    Example::

      get_nearest_value(X.loc[np.logical_and(idx, X.fnd_code=='C4_INF_Suivi_192')], ['numero_sejour', 'sequence_mouvement'],
                       'soarian_display_date', X.loc[i, 'soarian_display_date'],
                       dt_max = pd.to_timedelta('14 days'), forward=True)

    :param data: dataFrame in which the closest datapoint will be searched
    :param grouper: column or list of columns that will be used to group the data before taking the nearest datapoint
    :param timecol: name of the column of the dataframe to use for time measurements
    :param timestamp: time point of reference
    :param dt_max: maximum allowed time delta. If this value is exceeded, no data is returned
    :param forward: boolean flag (defaults to True), if False, ignore data in the future (w.r.t. the timestamp)
    :return: dataframe with each row (corresponding to each group) containinig the closest measurement w.r.t. the timestamp
    """
    grp = data.groupby(grouper)
    kmin = grp.apply(lambda x: get_idxmin(x[timecol], timestamp, forward)).dropna()
    if kmin.shape[0] == 0:
        return pd.DataFrame([])
    res = data.loc[kmin]
    res["dt"] = res[timecol] - timestamp
    res = res[res["dt"].abs() <= dt_max]
    return res


def replace_missing_nearest(
    data,
    grouper,
    val_col,
    timecol,
    dt_max=pd.to_timedelta("1 hour"),
    forward=True,
    val_num_col=None,
):
    """Returns a DataFrame where all the missing values have been replaced with the closest result from the same group (grouper) within dt_max.
    When no other results from the same group are found within dt_max then the result stays null.

    :param data: dataFrame with missing values
    :param grouper: column or list of columns that will be used to group the data before taking the nearest datapoint
    :param val_col: the column of the dataframe that contains the missing values
    :param timecol: name of the column of the dataframe to use for time measurements
    :param dt_max: maximum allowed time delta. If this value is exceeded, no data is returned
    :param forward: boolean flag (defaults to True), if False, ignore data in the future (w.r.t. the timestamp)
    :param val_num_col: the name of the column containing the numerical version of the value, if exists.
    :return: dataframe with each row (corresponding to each group) containinig the closest measurement w.r.t. the timestamp
    """
    # get the inital missing values
    dat = data.copy()
    missing_results = dat[dat[val_col].isna()]
    while len(missing_results) > 0:  # for as long as there are missing lab results
        idx = missing_results.iloc[
            0, :
        ].name  # get the index of the first missing value

        # find the nearest lab result of the patient
        nearest = get_nearest_value(
            data=dat,
            grouper=grouper,
            timecol=timecol,
            timestamp=dat.loc[idx, timecol],
            dt_max=dt_max,
            forward=forward,
        )
        nearest = nearest[nearest[grouper].isin(missing_results[grouper])]

        # sort by closeness to the lab result
        nearest["total_seconds"] = nearest["dt"].apply(
            lambda x: np.abs(x.total_seconds())
        )
        nearest = nearest.sort_values(by="total_seconds")

        # replace missing value with the nearest one
        dat.loc[idx, val_col] = nearest.head(1)[val_col].values
        if val_num_col:
            dat.loc[idx, val_num_col] = nearest.head(1)[val_num_col].values
        # reinstantiate the missing results
        missing_results = dat[dat[val_col].isna()]

    return dat


def get_timeseries(
    data,
    grouper,
    target_col,
    target_col_values,
    res_col,
    date_col,
    freq,
    agg_func,
    origin="start",
    origin_col=None,
):
    """Returns a timeseries in a dataframe format. All values are aggregated using the 'agg_func' function(s) over the
    desired time period defined by 'freq'.
    If 'origin' is set to 'start', the computed time periods start at the time of the first measurement.
    If 'origin' is set to 'start_day' the computed time periods start at the beginning of the day.
    Note: in order to obtain a single aggregated result (1 line per stay), the date_col can be set to a constant value
    for all rows (e.g. data[date_col] = pd.to_datetime('2020-01-01')).
    Example:
      get_timeseries(df_soa, 'numero_sejour', 'fnd_code', soa_numeric_codes, 'fnd_value',
                             'soarian_display_date', '24H', ['mean', 'std'])
    :param data: input dataframe
    :param grouper: column used to regroup rows
    :param target_col: column indexing the rows of interest 'target_col_values' (e.g. 'code_test_molis' or 'fnd_code')
    :param target_col_values: list of values for 'target_col' that will serve as columns for the returned dataframe
    :param res_col: column containing the result values that will be used to fill the rows
    :param date_col: column containing the time information
    :param freq: frequency at which the time period will be computed (i.e. size of the time window)
    :param agg_func: aggregation function (or list of functions) to apply to all the measurements in the same time window
    :param origin: can be 'start' (i.e. the period stat at the 1st measurements) or 'start_day' (i.e., the period start at the beginning of the day)
    :param origin_col: if origin == None, this parameter can be set to select a date column that will be used to compute the period start
    :return: return the timeseries in a dataframe format, indexed by grouper and time period
    """
    if origin_col is not None and origin is not None:
        print(
            "ERROR: parameters 'origin_col' and 'origin' cannot be used at the same time"
        )
        return None
    # safety first
    if type(agg_func) is not list:
        agg_func = [agg_func]
    # dataframe that will contain the resulting timeseries
    r = pd.DataFrame()
    r.index = pd.Index([], name=grouper)

    if origin == "start" or origin_col is not None:
        # gather the aggregated data by individual 'grouper' value (slower but unavoidable in order to compute the
        # individual date0 used as origin for pd.Grouper())
        def aggregate(df):
            idx = df[grouper].iloc[0]  # save the index of the current group
            if origin == "start":
                date0 = df[date_col].min()
            else:
                date0 = df[origin_col].min()
            df = df.groupby(
                [
                    target_col,
                    pd.Grouper(key=date_col, freq=freq, origin=date0, sort=True),
                ]
            )[res_col].agg(agg_func)
            df = df.unstack(level=0)  # reshape the dataframe
            newcols = [
                f"{code}_{f}" for f, code in df.columns
            ]  # merge the tow column levels into one name
            df.columns = newcols
            df[grouper] = idx
            df = df.set_index(
                grouper, append=True
            ).swaplevel()  # set new index, in the right order
            return df

        df = data[data[target_col].isin(target_col_values)]
        parallel = ParallelApply(func=aggregate, grp=True)
        return parallel.fit_transform(
            df.groupby(grouper)
        )  # parallel implementation for speedup

    for code in target_col_values:
        # loop over the codes and gather the aggregated data (faster)
        df = data[data[target_col] == code]
        df = df.groupby(
            [grouper, pd.Grouper(key=date_col, freq=freq, origin=origin, sort=True)]
        )[res_col].agg(agg_func)
        df = df.rename({a: f"{code}_{a}" for a in list(agg_func)}, axis=1)
        r = r.join(df, how="outer")
    return r


def fill_with_nearest(
    data,
    code,
    date_col,
    val_col,
    agg_func="median",
    resolution="1H",
    maxspan=pd.to_timedelta("12 hours"),
    forward=True,
    no_agg=False,
):
    """Returns a dataframe that contains, for each measurement of the original dataframe 'data', all other measurements
    in the maxspan time interval. If multiple values of the same 'code' are found, those are aggregated using the
    'agg_function' to a single row if 'agg_func' is specified, otherwise, the closest match is used ('no_agg'=False).
    The 'agg_date' corresponding to the measurement period and the delta time 'dt' between the aggregation time and
    the real measurement are returned as well.
    Example::

      fill_with_nearest(df_soa, 'fnd_code', 'soarian_display_date', 'fnd_value')

    Parallel implementation example (using the functools partial)::

      parallel_grp_apply(df_test.groupby('numero_sejour'),
                         partial(df_soa, code='fnd_code', date_col='soarian_display_date',
                         val_col='fnd_value', agg_fun='median', resolution='1H', maxspan=pd.to_timedelta('12 hours')))

    :param data: input dataframe
    :param code: column containing all the measurements names (e.g. 'code_test_molis' or 'fnd_code')
    :param date_col: column containing the time information
    :param val_col: column containing the measurements values that will be used to fill the rows
    :param agg_func: function used on the val_col to aggregate multiple measurements in the same time interval
    :param resolution: time resolution of the returned dataframe
    :param maxspan: maximum timespan where neighboring measurements are searched
    :param forward: boolean flag (defaults to True), if False, ignore data in the future (w.r.t. the measurement's timestamp)
    :param no_agg: boolean flag, defaults to False, if True, potential duplicated rows will not be aggregated but the closest one (smallest dt) will be chosen
    :return: dataframe containing the original measurements and the aggregated closest neighbor measurements
    """
    if no_agg is not None and agg_func is not None:
        print(
            f"WARNING: agg_func was set to {agg_func} but will not be used since no_agg has been specified"
        )
    res = pd.DataFrame()
    dfindex = data.index.name
    for idx in data.index.unique():
        dat = data.loc[[idx]]
        for t in dat[date_col]:  # TODO optimization?
            n = get_nearest_value(
                dat.reset_index(), code, date_col, t, dt_max=maxspan, forward=forward
            )
            n[date_col] = t
            res = pd.concat([res, n], ignore_index=True)
        if no_agg:
            # return the closest (non-null) value to the original timestamp
            # note: the first() function returns only non-nan values (as opposed to head())
            # copy the date column so that we keep a trace of it (otherwise it will be erased by the pd.Grouper() function)
            res["agg_date"] = res[date_col].copy()
            res = (
                res.sort_values(by="dt")
                .groupby([dfindex, code, pd.Grouper(key="agg_date", freq=resolution)])
                .first()
            )
            # transform the indices to have the final dataframe indexed only by dfindex
            res = res.reset_index(level=code, drop=False)
            res = res.reset_index(level="agg_date")
        else:
            # aggregate values on potentially duplicated rows
            agg_rules = {c: "max" for c in res.columns}
            agg_rules[val_col] = agg_func
            res = res.groupby(
                [dfindex, code, pd.Grouper(key=date_col, freq=resolution)]
            ).agg(agg_rules)
            # remove duplicated columns
            res = res.drop(columns=[dfindex, code])
            # transform the indices to have the final dataframe indexed only by dfindex
            res.index = res.index.set_names("agg_date", level=date_col)
            res = res.reset_index(level=["agg_date", code])
    return res


def time_rolling_window(data, date_col, period, agg_fun, target_col=None):
    """Returns a DataFrame containing rolling window values of the target columns representing the state of
    the dataframe 'data' for each period of time 'period'. The data is aggregated in each window using the
    agg_fun function(s).

    :param data: input dataframe
    :param date_col: name of the column containing the datetime information
    :param period: size of the time-window interval (a timedelta interval)
    :param agg_fun: (list or single function name) of the aggregation function that will be used to aggregate values over a time-window
    :param target_col: (list or a single column label) that contains the data of interest. If None, all columns of the original dataframe will be kept
    :return:
    """
    # if a target column is not defined, return the entire dataframe
    if target_col is None:
        target_col = list(data.columns)
        target_col.remove(date_col)
    return (
        data.set_index(date_col).sort_index()[target_col].rolling(period).agg(agg_fun)
    )


def grptime_rolling_window(data, grouper, date_col, period, agg_fun, target_col=None):
    """Returns a DataFrame grouped by 'grouper' containing rolling window values of the target columns
    representing the state of the dataframe 'data' for each period of time 'period'. The data is aggregated
    in each window using the agg_fun function(s).
    This function is based on the implementation of the function time_rolling_window().

    :param data: input dataframe
    :param grouper: name of the column that will be used to group data rows together
    :param date_col: name of the column containing the datetime information
    :param period: size of the time-window interval
    :param agg_fun: (list or single function name) of the aggregation function that will be used to aggregate values over a time-window
    :param target_col: (list or a single column label) that contains the data of interest. If None, all columns of the original dataframe will be kept
    :return: the values of the dataframe in each rolling window
    """
    res = data.groupby(grouper).apply(
        partial(
            time_rolling_window,
            date_col=date_col,
            period=period,
            agg_fun=agg_fun,
            target_col=target_col,
        )
    )
    return res


def data_in_interval(
    data, idx, date_col, date0, dt, forward_only=False, backward_only=False
):
    """Returns the rows of the input dataframe that are in the desired temporal interval with respect to date0.
    forward_only and backward_only flags can further restrict the interval properties.

    :param data: input dataframe
    :param idx: data index
    :param date_col: label of the dataframe column containing the time information
    :param date0: reference date
    :param dt: length of time interval to consider
    :param forward_only: boolean flag, defaults to False, if True, only events in the future (w.r.t. date0) are considered
    :param backward_only: boolean flag, defaults to False, if True, only events in the past (w.r.t. date0) are considered
    :return: the rows of the input dataframe that are in the desired time interval
    """
    try:
        res = data.loc[[idx]]
    except:
        return pd.DataFrame(columns=data.columns)
    ind = [False] * res.shape[0]
    if not backward_only:
        ind_b = np.logical_and(res[date_col] >= date0, res[date_col] <= date0 + dt)
        ind = np.logical_or(ind, ind_b)
    if not forward_only:
        ind_f = np.logical_and(res[date_col] >= date0 - dt, res[date_col] <= date0)
        ind = np.logical_or(ind, ind_f)
    return res[ind]
