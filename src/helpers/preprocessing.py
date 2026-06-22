#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Data preprocessing
#
#################################################################################################################

import os
import numpy as np
import pandas as pd
import tables

from sklearn.model_selection import train_test_split


# ========================================= Convert NILM dataset to TSER ========================================= #
def nilmdataset_to_tser(data):
    """
    Input:
        data: Convention of 4D array obtained with every NILM Databuilder available in data_processing

    Output:
        X: 2D array of input time series
        y: 2D array as (len(X), 1) -> sum of energy consumed in each window
    """
    X = data[:, 0, 0, :]
    y = np.sum(data[:, 1, 0, :], axis=-1)

    return X, y


def split_train_valid_test(data, test_size=0.2, valid_size=0, nb_label_col=1):
    if isinstance(data, pd.core.frame.DataFrame):
        if valid_size != 0:
            X_train_valid, X_test, y_train_valid, y_test = train_test_split(
                data.iloc[:, :-nb_label_col],
                data.iloc[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid, y_train_valid, test_size=valid_size, random_state=0
            )

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data.iloc[:, :-nb_label_col],
                data.iloc[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )
            return X_train, y_train, X_test, y_test

    elif isinstance(data, np.ndarray):
        if valid_size != 0:
            X_train_valid, X_test, y_train_valid, y_test = train_test_split(
                data[:, :-nb_label_col],
                data[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid, y_train_valid, test_size=valid_size, random_state=0
            )
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data[:, :-nb_label_col],
                data[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )

            return X_train, y_train, X_test, y_test
    else:
        raise Exception("Please provide pandas Dataframe or numpy array object.")


def split_train_valid_test_pdl(
    df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0, return_df=False
):
    """
    Split DataFrame based on index ID (ID PDL for example)

    - Input : df_data -> DataFrame
              test_size -> Percentage data for test
              valid_size -> Percentage data for valid
              nb_label_col -> Number of columns of label
              seed -> Set seed
              return_df -> Return DataFrame instances, or Numpy Instances
    - Output:
            np.arrays or DataFrame Instances
    """

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[: int(len(list_pdl) * (1 - test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1 - test_size)) :]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[: int(len(pdl_train_valid) * (1 - valid_size))]

    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)

    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1 - valid_size)) :]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)

    if return_df:
        if valid_size != 0:
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
    else:
        X_train = df_train.iloc[:, :-nb_label_col].to_numpy().astype(np.float32)
        y_train = df_train.iloc[:, -nb_label_col:].to_numpy().astype(np.float32)
        X_test = df_test.iloc[:, :-nb_label_col].to_numpy().astype(np.float32)
        y_test = df_test.iloc[:, -nb_label_col:].to_numpy().astype(np.float32)

        if valid_size != 0:
            X_valid = df_valid.iloc[:, :-nb_label_col].to_numpy().astype(np.float32)
            y_valid = df_valid.iloc[:, -nb_label_col:].to_numpy().astype(np.float32)

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test


def split_train_test_pdl_nilmdataset(
    data,
    st_date,
    seed=0,
    nb_house_test=None,
    perc_house_test=None,
    nb_house_valid=None,
    perc_house_valid=None,
):
    assert nb_house_test is not None or perc_house_test is not None
    assert len(data) == len(st_date)
    assert isinstance(st_date, pd.DataFrame)

    np.random.seed(seed)

    if nb_house_valid is not None or perc_house_valid is not None:
        assert (nb_house_test is not None and nb_house_valid is not None) or (
            perc_house_test is not None and perc_house_valid is not None
        )

    if len(data.shape) > 2:
        tmp_shape = data.shape
        data = data.reshape(data.shape[0], -1)

    data = pd.concat([st_date.reset_index(), pd.DataFrame(data)], axis=1).set_index(
        "index"
    )
    list_pdl = np.array(data.index.unique())
    np.random.shuffle(list_pdl)

    if nb_house_test is None:
        nb_house_test = max(1, int(len(list_pdl) * perc_house_test))
        if perc_house_valid is not None and nb_house_valid is None:
            nb_house_valid = max(1, int(len(list_pdl) * perc_house_valid))

    if nb_house_valid is not None:
        assert len(list_pdl) > nb_house_test + nb_house_valid
    else:
        assert len(list_pdl) > nb_house_test

    pdl_test = list_pdl[:nb_house_test]

    if nb_house_valid is not None:
        pdl_valid = list_pdl[nb_house_test : nb_house_test + nb_house_valid]
        pdl_train = list_pdl[nb_house_test + nb_house_valid :]
    else:
        pdl_train = list_pdl[nb_house_test:]

    df_train = data.loc[pdl_train, :].copy()
    df_test = data.loc[pdl_test, :].copy()

    st_date_train = df_train.iloc[:, :1]
    data_train = df_train.iloc[:, 1:].values.reshape(
        (len(df_train), tmp_shape[1], tmp_shape[2], tmp_shape[3])
    )
    st_date_test = df_test.iloc[:, :1]
    data_test = df_test.iloc[:, 1:].values.reshape(
        (len(df_test), tmp_shape[1], tmp_shape[2], tmp_shape[3])
    )

    if nb_house_valid is not None:
        df_valid = data.loc[pdl_valid, :].copy()
        st_date_valid = df_valid.iloc[:, :1]
        data_valid = df_valid.iloc[:, 1:].values.reshape(
            (len(df_valid), tmp_shape[1], tmp_shape[2], tmp_shape[3])
        )

        return (
            data_train,
            st_date_train,
            data_valid,
            st_date_valid,
            data_test,
            st_date_test,
        )
    else:
        return data_train, st_date_train, data_test, st_date_test


def split_train_test_nilmdataset(data, st_date, perc_house_test=0.2, seed=0):
    np.random.seed(seed)

    data_len = np.arange(len(data))
    np.random.shuffle(data_len)

    split_index = int(len(data_len) * (1 - perc_house_test))
    train_idx, test_idx = data_len[:split_index], data_len[split_index:]

    data_train, st_date_train = data[train_idx], st_date.iloc[train_idx]
    data_test, st_date_test = data[test_idx], st_date.iloc[test_idx]

    return data_train, st_date_train, data_test, st_date_test


def normalize_exogene(x, xmin, xmax, newRange):
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    norm = (x - xmin) / (xmax - xmin)
    if newRange == (0, 1):
        return norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]


def create_exogene(
    values, st_date, list_exo_variables, freq, cosinbase=True, new_range=(-1, 1)
):
    if cosinbase:
        n_var = 2 * len(list_exo_variables)
    else:
        n_var = len(list_exo_variables)

    np_extra = np.zeros(
        (1, n_var, len(values[-1]) if len(values.shape) > 1 else len(values))
    ).astype(np.float32)

    tmp = pd.date_range(
        start=st_date,
        periods=len(values[-1]) if len(values.shape) > 1 else len(values),
        freq=freq,
    )

    k = 0
    for exo_var in list_exo_variables:
        if exo_var == "month":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.month.values / 12.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.month.values / 12.0)
                k += 2
            else:
                np_extra[k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=12, newRange=new_range
                )
                k += 1
        elif exo_var == "dom":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.day.values / 31.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.day.values / 31.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=12, newRange=new_range
                )
                k += 1
        elif exo_var == "dow":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.dayofweek.values / 7.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.dayofweek.values / 7.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=7, newRange=new_range
                )
                k += 1
        elif exo_var == "hour":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.hour.values / 24.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.hour.values / 24.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=0, xmax=24, newRange=new_range
                )
                k += 1
        elif exo_var == "minute":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.minute.values / 60.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.minute.values / 60.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.minute.values, xmin=0, xmax=60, newRange=new_range
                )
                k += 1
        else:
            raise ValueError(
                "Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour', 'minute' supported, received {}".format(
                    exo_var
                )
            )

    if len(values.shape) == 1:
        values = np.expand_dims(np.expand_dims(values, axis=0), axis=0)
    elif len(values.shape) == 2:
        values = np.expand_dims(values, axis=0)

    values = np.concatenate((values, np_extra), axis=1)

    return values


# ===================== UKDALE DataBuilder =====================#
class UKDALE_DataBuilder(object):
    def __init__(
        self,
        data_path,
        mask_app,
        sampling_rate,
        window_size,
        window_stride=None,
        soft_label=False,
        use_status_from_kelly_paper=True,
        synth_aggregate_apps=None,
    ):
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.soft_label = soft_label
        self.synth_aggregate_apps = synth_aggregate_apps

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size == "week":
                self.flag_week = True
                self.flag_day = False
                if (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 10080
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 1008
                else:
                    raise ValueError(
                        f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}"
                    )
            elif window_size == "day":
                self.flag_week = False
                self.flag_day = True
                if self.sampling_rate == "30s":
                    self.window_size = 2880
                elif (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 1440
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 144
                else:
                    raise ValueError(
                        f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}"
                    )
            else:
                raise ValueError(
                    f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}'
                )
        else:
            self.flag_week = False
            self.flag_day = False
            self.window_size = window_size

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ["aggregate"] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 6000

        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        # All parameters are in Watts and taken from Kelly et all. NeuralNILM paper
        if self.use_status_from_kelly_paper:
            # Threshold parameters are in Watts and time parameter in 10sec (minimum resampling)
            self.appliance_param = {
                "kettle": {
                    "min_threshold": 2000,
                    "max_threshold": 3100,
                    "min_on_duration": 1,
                    "min_off_duration": 0,
                    "min_activation_time": 1,
                },
                "fridge": {
                    "min_threshold": 50,
                    "max_threshold": 300,
                    "min_on_duration": 6,
                    "min_off_duration": 1,
                    "min_activation_time": 1,
                },
                "washing_machine": {
                    "min_threshold": 20,
                    "max_threshold": 2500,
                    "min_on_duration": 180,
                    "min_off_duration": 16,
                    "min_activation_time": 12,
                },
                "microwave": {
                    "min_threshold": 200,
                    "max_threshold": 3000,
                    "min_on_duration": 1,
                    "min_off_duration": 3,
                    "min_activation_time": 1,
                },
                "dishwasher": {
                    "min_threshold": 10,
                    "max_threshold": 2500,
                    "min_on_duration": 180,
                    "min_off_duration": 180,
                    "min_activation_time": 12,
                },
            }
        else:
            # Threshold parameters are in Watts
            self.appliance_param = {
                "kettle": {"min_threshold": 500, "max_threshold": 6000},
                "washing_machine": {"min_threshold": 300, "max_threshold": 3000},
                "dishwasher": {"min_threshold": 300, "max_threshold": 3000},
                "microwave": {"min_threshold": 200, "max_threshold": 6000},
                "fridge": {"min_threshold": 50, "max_threshold": 300},
            }

    def get_house_data(self, house_indicies):
        assert len(house_indicies) == 1, (
            "get_house_data() implemented to get data from 1 house only at a time."
        )

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset

        Return :
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date

    def get_nilm_dataset(self, house_indicies):
        """
        Process data to build NILM usecase

        Return :
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :

                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """

        output_data = np.array([])
        st_date = pd.DataFrame()

        for indice in house_indicies:
            tmp_list_st_date = []

            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if self.window_size == self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

            X = np.empty(
                (len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size)
            )

            cpt = 0
            for i in range(n_wins):
                tmp = stems[
                    :,
                    i * self.window_stride : i * self.window_stride + self.window_size,
                ]

                if not self._check_anynan(
                    tmp
                ):  # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i * self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key + 1, :]
                        key += 2

                    cpt += 1  # Add one subsequence

            tmp_st_date = pd.DataFrame(
                data=tmp_list_st_date,
                index=[indice for _ in range(cpt)],
                columns=["start_date"],
            )
            output_data = (
                np.concatenate((output_data, X[:cpt, :, :, :]), axis=0)
                if output_data.size
                else X[:cpt, :, :, :]
            )
            st_date = (
                pd.concat([st_date, tmp_st_date], axis=0)
                if st_date.size
                else tmp_st_date
            )

        return output_data, st_date

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on]
            off_events = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status

    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.

        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app) - 1) * 2, dataframe.shape[0]))
        stems[0, :] = (
            dataframe["synth_aggregate"].values
            if "synth_aggregate" in dataframe.columns
            else dataframe["aggregate"].values
        )

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key + 1, :] = dataframe[appliance + "_status"].values
            key += 2

        return stems, list(dataframe.index)

    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.

        Return : pd.core.frame.DataFrame instance
        """
        path_house = self.data_path + "House" + str(indice) + os.sep
        self._check_if_file_exist(
            path_house + "labels.dat"
        )  # Check if labels exist at provided path

        # House labels
        house_label = pd.read_csv(path_house + "labels.dat", sep=" ", header=None)
        house_label.columns = ["id", "appliance_name"]
        _ukdale_label_map = {"dish_washer": "dishwasher"}
        house_label["appliance_name"] = house_label["appliance_name"].map(
            lambda x: _ukdale_label_map.get(x, x)
        )

        # Load aggregate load curve and resample to lowest sampling rate
        house_data = pd.read_csv(path_house + "channel_1.dat", sep=" ", header=None)
        house_data.columns = ["time", "aggregate"]
        house_data["time"] = pd.to_datetime(house_data["time"], unit="s")
        house_data = house_data.set_index("time")  # Set index to time
        house_data = (
            house_data.resample("10s").mean().ffill(limit=6)
        )  # Resample to minimum of 10s and ffill for 1min30
        house_data[house_data < 5] = 0  # Remove small value
        house_data = house_data.clip(lower=0, upper=self.cutoff)

        if self.flag_week:
            tmp_min = house_data[
                (house_data.index.weekday == 1)
                & (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[
                (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]

        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if (
                len(
                    house_label.loc[house_label["appliance_name"] == appliance][
                        "id"
                    ].values
                )
                != 0
            ):
                i = house_label.loc[house_label["appliance_name"] == appliance][
                    "id"
                ].values[0]

                # Load aggregate load curve and resample to lowest sampling rate
                appl_data = pd.read_csv(
                    path_house + "channel_" + str(i) + ".dat", sep=" ", header=None
                )
                appl_data.columns = ["time", appliance]
                appl_data["time"] = pd.to_datetime(appl_data["time"], unit="s")
                appl_data = appl_data.set_index("time")
                appl_data = appl_data.resample("10s").mean()
                appl_data[appliance] = self._fill_long_gaps_with_zero(appl_data[appliance])
                appl_data = appl_data.ffill(limit=6)
                appl_data[appl_data < 5] = 0  # Remove small value

                # Merge aggregate load curve with appliance load curve
                house_data = pd.merge(house_data, appl_data, how="inner", on="time")
                del appl_data
                house_data = house_data.clip(
                    lower=0, upper=self.cutoff
                )  # Apply general cutoff
                house_data = house_data.sort_index()

                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                # Creating status
                initial_status = (
                    (
                        (
                            house_data[appliance]
                            >= self.appliance_param[appliance]["min_threshold"]
                        )
                        & (
                            house_data[appliance]
                            <= self.appliance_param[appliance]["max_threshold"]
                        )
                    )
                    .astype(int)
                    .values
                )

                if self.use_status_from_kelly_paper:
                    house_data[appliance + "_status"] = self._compute_status(
                        initial_status,
                        self.appliance_param[appliance]["min_on_duration"],
                        self.appliance_param[appliance]["min_off_duration"],
                        self.appliance_param[appliance]["min_activation_time"],
                    )
                else:
                    house_data[appliance + "_status"] = initial_status

                # Finally replacing nan values put to -1 by nan
                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate != "10s":
            house_data = house_data.resample(self.sampling_rate).mean()

        for appliance in self.mask_app[1:]:
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance + "_status"] = (
                        house_data[appliance + "_status"] > 0
                    ).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance + "_status"] = 0

        if self.synth_aggregate_apps is not None:
            # Load extra appliance channels not already in mask_app, then sum all
            # synth_aggregate_apps to form a clean aggregate over the 5 evaluated appliances.
            already_loaded = set(self.mask_app[1:])
            for appliance in self.synth_aggregate_apps:
                if appliance in already_loaded:
                    continue
                app_ids = house_label.loc[
                    house_label["appliance_name"] == appliance, "id"
                ].values
                if len(app_ids) == 0:
                    house_data[appliance] = 0.0
                    continue
                appl_data = pd.read_csv(
                    path_house + "channel_" + str(app_ids[0]) + ".dat",
                    sep=" ",
                    header=None,
                )
                appl_data.columns = ["time", appliance]
                appl_data["time"] = pd.to_datetime(appl_data["time"], unit="s")
                appl_data = appl_data.set_index("time")
                appl_data = appl_data.resample("10s").mean()
                appl_data[appliance] = self._fill_long_gaps_with_zero(appl_data[appliance])
                appl_data = appl_data.ffill(limit=6)
                appl_data[appl_data < 5] = 0
                appl_data = appl_data.clip(lower=0, upper=self.cutoff)
                if self.sampling_rate != "10s":
                    appl_data = appl_data.resample(self.sampling_rate).mean()
                house_data = pd.merge(house_data, appl_data, how="left", on="time")
                house_data[appliance] = house_data[appliance].fillna(0.0)

            house_data["synth_aggregate"] = sum(
                house_data[app].fillna(0.0) for app in self.synth_aggregate_apps
            )

        return house_data

    def _fill_long_gaps_with_zero(self, series, gap_threshold_steps=12):
        """Fill NaN runs > 120s (12 × 10s steps) with 0.0 — silence means appliance OFF per Kelly & Knottenbelt."""
        is_nan = series.isna()
        nan_group = (is_nan != is_nan.shift()).cumsum()
        nan_group_sizes = is_nan.groupby(nan_group).transform("sum")
        long_gap_mask = is_nan & (nan_group_sizes > gap_threshold_steps)
        series = series.copy()
        series[long_gap_mask] = 0.0
        return series

    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in [
                "washing_machine",
                "cooker",
                "dishwasher",
                "kettle",
                "fridge",
                "microwave",
                "electric_heater",
            ], f"Selected applicance unknow for UKDALE Dataset, got: {appliance}"
        return

    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return

    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))


# ===================== REFIT DataBuilder =====================#
class REFIT_DataBuilder(object):
    def __init__(
        self,
        data_path,
        mask_app,
        sampling_rate,
        window_size,
        window_stride=None,
        use_status_from_kelly_paper=False,
        soft_label=False,
        synth_aggregate_apps=None,
    ):
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate
        self.soft_label = soft_label
        self.synth_aggregate_apps = synth_aggregate_apps

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size == "week":
                self.flag_week = True
                self.flag_day = False
                if (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 10080
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 1008
                else:
                    raise ValueError(
                        f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}"
                    )
            elif window_size == "day":
                self.flag_week = False
                self.flag_day = True
                if self.sampling_rate == "30s":
                    self.window_size = 2880
                elif (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 1440
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 144
                else:
                    raise ValueError(
                        f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}"
                    )
            else:
                raise ValueError(
                    f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}'
                )
        else:
            self.flag_week = False
            self.flag_day = False
            self.window_size = window_size

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ["Aggregate"] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 10000

        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        # All parameters are in Watts and adapted from Kelly et all. NeuralNILM paper
        if self.use_status_from_kelly_paper:
            # Threshold parameters are in Watts and time parameter in 10sec
            self.appliance_param = {
                "Kettle": {
                    "min_threshold": 1000,
                    "max_threshold": 6000,
                    "min_on_duration": 1,
                    "min_off_duration": 0,
                    "min_activation_time": 0,
                },
                "WashingMachine": {
                    "min_threshold": 20,
                    "max_threshold": 3500,
                    "min_on_duration": 6,
                    "min_off_duration": 16,
                    "min_activation_time": 12,
                },
                "Dishwasher": {
                    "min_threshold": 50,
                    "max_threshold": 3000,
                    "min_on_duration": 2,
                    "min_off_duration": 180,
                    "min_activation_time": 12,
                },
                "Microwave": {
                    "min_threshold": 200,
                    "max_threshold": 6000,
                    "min_on_duration": 1,
                    "min_off_duration": 3,
                    "min_activation_time": 0,
                },
            }
        else:
            # Threshold parameters are in Watts
            self.appliance_param = {
                "Kettle": {"min_threshold": 500, "max_threshold": 6000},
                "WashingMachine": {"min_threshold": 300, "max_threshold": 4000},
                "Dishwasher": {"min_threshold": 300, "max_threshold": 4000},
                "Microwave": {"min_threshold": 200, "max_threshold": 6000},
            }

    def get_house_data(self, house_indicies):
        assert len(house_indicies) == 1, (
            "get_house_data() implemented to get data from 1 house only at a time."
        )

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset

        Return :
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date

    def get_nilm_dataset(self, house_indicies):
        """
        Process data to build NILM usecase

        Return :
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :

                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """

        output_data = np.array([])
        st_date = pd.DataFrame()

        for indice in house_indicies:
            tmp_list_st_date = []

            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if self.window_size == self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

            X = np.empty(
                (len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size)
            )

            cpt = 0
            for i in range(n_wins):
                tmp = stems[
                    :,
                    i * self.window_stride : i * self.window_stride + self.window_size,
                ]

                if not self._check_anynan(
                    tmp
                ):  # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i * self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key + 1, :]
                        key += 2

                    cpt += 1  # Add one subsequence

            tmp_st_date = pd.DataFrame(
                data=tmp_list_st_date,
                index=[indice for j in range(cpt)],
                columns=["start_date"],
            )
            output_data = (
                np.concatenate((output_data, X[:cpt, :, :, :]), axis=0)
                if output_data.size
                else X[:cpt, :, :, :]
            )
            st_date = (
                pd.concat([st_date, tmp_st_date], axis=0)
                if st_date.size
                else tmp_st_date
            )

        return output_data, st_date

    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.

        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app) - 1) * 2, dataframe.shape[0]))
        stems[0, :] = (
            dataframe["synth_aggregate"].values
            if "synth_aggregate" in dataframe.columns
            else dataframe["Aggregate"].values
        )

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key + 1, :] = dataframe[appliance + "_status"].values
            key += 2

        return stems, list(dataframe.index)

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on]
            off_events = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status

    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.

        Return : pd.core.frame.DataFrame instance
        """
        file = self.data_path + "CLEAN_House" + str(indice) + ".csv"
        self._check_if_file_exist(file)
        labels_houses = pd.read_csv(self.data_path + "HOUSE_Labels").set_index(
            "House_id"
        )

        house_data = pd.read_csv(file)
        house_data.columns = list(labels_houses.loc[int(indice)].values)
        house_data = house_data.set_index("Time").sort_index()
        house_data.index = pd.to_datetime(house_data.index)
        idx_to_drop = house_data[house_data["Issues"] == 1].index
        house_data = house_data.drop(index=idx_to_drop, axis=0)
        house_data = (
            house_data.resample(rule="10s").mean().ffill(limit=9)
        )  # Resample to minimum of 10sec and ffill for 1min30
        house_data[house_data < 5] = 0  # Remove small value
        house_data = house_data.clip(lower=0, upper=self.cutoff)  # Apply general cutoff
        house_data = house_data.sort_index()

        if self.flag_week:
            tmp_min = house_data[
                (house_data.index.weekday == 1)
                & (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[
                (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]

        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if appliance in house_data:
                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                # Creating status
                initial_status = (
                    (
                        (
                            house_data[appliance]
                            >= self.appliance_param[appliance]["min_threshold"]
                        )
                        & (
                            house_data[appliance]
                            <= self.appliance_param[appliance]["max_threshold"]
                        )
                    )
                    .astype(int)
                    .values
                )

                if self.use_status_from_kelly_paper:
                    house_data[appliance + "_status"] = self._compute_status(
                        initial_status,
                        self.appliance_param[appliance]["min_on_duration"],
                        self.appliance_param[appliance]["min_off_duration"],
                        self.appliance_param[appliance]["min_activation_time"],
                    )
                else:
                    house_data[appliance + "_status"] = initial_status

                # Finally replacing nan values put to -1 by nan
                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate != "10s":
            house_data = house_data.resample(self.sampling_rate).mean()

        if self.synth_aggregate_apps is not None:
            house_data["synth_aggregate"] = sum(
                house_data[app].fillna(0.0)
                if app in house_data.columns
                else pd.Series(0.0, index=house_data.index)
                for app in self.synth_aggregate_apps
            )

        tmp_list = ["Aggregate"]
        if "synth_aggregate" in house_data.columns:
            tmp_list.append("synth_aggregate")
        for appliance in self.mask_app[1:]:
            tmp_list.append(appliance)
            tmp_list.append(appliance + "_status")
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance + "_status"] = (
                        house_data[appliance + "_status"] > 0
                    ).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance + "_status"] = 0

        house_data = house_data[tmp_list]

        return house_data

    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in [
                "WashingMachine",
                "Dishwasher",
                "Kettle",
                "Microwave",
            ], f"Selected applicance unknow for REFIT Dataset, got: {appliance}"
        return

    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return

    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))


# ===================== REDD DataBuilder =====================#
class REDD_DataBuilder(object):
    # Mains meters (two phases) for all buildings
    REDD_MAINS = [1, 2]

    # Map: house_int -> canonical_appliance -> [meter_ints]
    REDD_METER_MAP = {
        1: {
            "Fridge": [5],
            "Dishwasher": [6],
            "Microwave": [11],
            "WashingMachine": [10, 20],
            "WasherDryer": [10, 20],
        },
        2: {
            "Fridge": [9],
            "Dishwasher": [10],
            "Microwave": [6],
            "WashingMachine": [7],
            "WasherDryer": [7],
        },
        3: {
            "Fridge": [7],
            "Dishwasher": [9],
            "Microwave": [16],
            "WashingMachine": [13, 14],
            "WasherDryer": [13, 14],
        },
        4: {
            "Dishwasher": [15],
            "WashingMachine": [7],
            "WasherDryer": [7],
        },
        5: {
            "Fridge": [18],
            "Dishwasher": [20],
            "Microwave": [3],
            "WashingMachine": [8, 9],
            "WasherDryer": [8, 9],
        },
        6: {
            "Fridge": [8],
            "Dishwasher": [9],
            "WashingMachine": [4],
            "WasherDryer": [4],
        },
    }

    def __init__(
        self,
        data_path,
        mask_app,
        sampling_rate,
        window_size,
        window_stride=None,
        use_status_from_kelly_paper=True,
        soft_label=False,
        synth_aggregate_apps=None,
    ):
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate
        self.soft_label = soft_label
        self.synth_aggregate_apps = synth_aggregate_apps

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size == "week":
                self.flag_week = True
                self.flag_day = False
                if (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 10080
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 1008
                else:
                    raise ValueError(
                        f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}"
                    )
            elif window_size == "day":
                self.flag_week = False
                self.flag_day = True
                if self.sampling_rate == "30s":
                    self.window_size = 2880
                elif (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 1440
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 144
                else:
                    raise ValueError(
                        f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}"
                    )
            else:
                raise ValueError(
                    f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}'
                )
        else:
            self.flag_week = False
            self.flag_day = False
            self.window_size = window_size

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ["aggregate"] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 6000

        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        # All parameters are in Watts and adapted from Kelly et al. NeuralNILM paper
        if self.use_status_from_kelly_paper:
            # Threshold parameters are in Watts and time parameter in 10sec
            self.appliance_param = {
                "Fridge": {
                    "min_threshold": 50,
                    "max_threshold": 300,
                    "min_on_duration": 6,
                    "min_off_duration": 1,
                    "min_activation_time": 1,
                },
                "Microwave": {
                    "min_threshold": 200,
                    "max_threshold": 3000,
                    "min_on_duration": 1,
                    "min_off_duration": 3,
                    "min_activation_time": 1,
                },
                "Dishwasher": {
                    "min_threshold": 10,
                    "max_threshold": 2500,
                    "min_on_duration": 180,
                    "min_off_duration": 180,
                    "min_activation_time": 12,
                },
                "WashingMachine": {
                    # REDD "washer dryer" is a combined unit; the dryer element
                    # peaks ~3-5kW, so max_threshold must be well above 2500W or
                    # those steps read as off and no activation registers.
                    "min_threshold": 20,
                    "max_threshold": 6000,
                    "min_on_duration": 30,
                    "min_off_duration": 16,
                    "min_activation_time": 12,
                },
                "WasherDryer": {
                    # REDD "washer dryer" is a combined unit; the dryer element
                    # peaks ~3-5kW, so max_threshold must be well above 2500W or
                    # those steps read as off and no activation registers.
                    "min_threshold": 20,
                    "max_threshold": 6000,
                    "min_on_duration": 30,
                    "min_off_duration": 16,
                    "min_activation_time": 12,
                },
            }
        else:
            # Threshold parameters are in Watts
            self.appliance_param = {
                "Fridge": {"min_threshold": 50, "max_threshold": 300},
                "Microwave": {"min_threshold": 200, "max_threshold": 3000},
                "Dishwasher": {"min_threshold": 10, "max_threshold": 2500},
                "WashingMachine": {"min_threshold": 20, "max_threshold": 6000},
                "WasherDryer": {"min_threshold": 20, "max_threshold": 6000},
            }

    def get_house_data(self, house_indicies):
        assert len(house_indicies) == 1, (
            "get_house_data() implemented to get data from 1 house only at a time."
        )

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset

        Return :
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date

    def get_nilm_dataset(self, house_indicies):
        """
        Process data to build NILM usecase

        Return :
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :

                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """

        output_data = np.array([])
        st_date = pd.DataFrame()

        for indice in house_indicies:
            tmp_list_st_date = []

            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if self.window_size == self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

            X = np.empty(
                (len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size)
            )

            cpt = 0
            for i in range(n_wins):
                tmp = stems[
                    :,
                    i * self.window_stride : i * self.window_stride + self.window_size,
                ]

                if not self._check_anynan(
                    tmp
                ):  # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i * self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key + 1, :]
                        key += 2

                    cpt += 1  # Add one subsequence

            tmp_st_date = pd.DataFrame(
                data=tmp_list_st_date,
                index=[indice for j in range(cpt)],
                columns=["start_date"],
            )
            output_data = (
                np.concatenate((output_data, X[:cpt, :, :, :]), axis=0)
                if output_data.size
                else X[:cpt, :, :, :]
            )
            st_date = (
                pd.concat([st_date, tmp_st_date], axis=0)
                if st_date.size
                else tmp_st_date
            )

        return output_data, st_date

    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.

        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app) - 1) * 2, dataframe.shape[0]))
        stems[0, :] = (
            dataframe["synth_aggregate"].values
            if "synth_aggregate" in dataframe.columns
            else dataframe["aggregate"].values
        )

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key + 1, :] = dataframe[appliance + "_status"].values
            key += 2

        return stems, list(dataframe.index)

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on]
            off_events = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status

    def _read_meter_series(self, h5file, indice, meter):
        """
        Read a single meter's power series directly via PyTables.

        Reading through ``tables`` (rather than ``pd.HDFStore``) is required because
        pandas 3.0's HDFStore reader fails on this legacy NILMTK PyTables file
        ('numpy.bytes_' object has no attribute 'get'); PyTables reads it in any
        pandas version and handles the blosc decompression.

        Return : pd.Series indexed by tz-aware (US/Eastern) DatetimeIndex
        """
        node = h5file.get_node(f"/building{indice}/elec/meter{meter}/table")
        idx = node.col("index")  # int64 ns, UTC
        vals = node.col("values_block_0")[:, 0]  # float32 power, shape (N, 1)
        index = pd.to_datetime(idx, utc=True).tz_convert("US/Eastern")
        return pd.Series(vals.astype(float), index=index).sort_index()

    def _fill_long_gaps_with_zero(self, series, gap_threshold_steps=12):
        """Fill NaN runs > 120s (12 x 10s steps) with 0.0 -- silence means appliance OFF."""
        is_nan = series.isna()
        nan_group = (is_nan != is_nan.shift()).cumsum()
        nan_group_sizes = is_nan.groupby(nan_group).transform("sum")
        long_gap_mask = is_nan & (nan_group_sizes > gap_threshold_steps)
        series = series.copy()
        series[long_gap_mask] = 0.0
        return series

    def _get_dataframe(self, indice):
        """
        Load house data from redd.h5 and return one dataframe with aggregate and
        appliance resampled at chosen time step.

        Return : pd.core.frame.DataFrame instance
        """
        self._check_if_file_exist(self.data_path)
        h5file = tables.open_file(self.data_path, mode="r")
        try:
            # Build aggregate: sum of two mains phases, resampled to 10s
            agg_series = None
            for m in self.REDD_MAINS:
                s = self._read_meter_series(h5file, indice, m)
                s = s.resample("10s").mean().ffill(limit=6)
                if agg_series is None:
                    agg_series = s
                else:
                    agg_series = agg_series.add(s, fill_value=0)

            house_data = pd.DataFrame({"aggregate": agg_series})
            house_data[house_data < 5] = 0
            house_data = house_data.clip(lower=0, upper=self.cutoff)
            house_data = house_data.sort_index()

            if self.flag_week:
                tmp_min = house_data[
                    (house_data.index.weekday == 1)
                    & (house_data.index.hour == 0)
                    & (house_data.index.minute == 0)
                    & (house_data.index.second == 0)
                ]
                house_data = house_data[house_data.index >= tmp_min.index[0]]
            elif self.flag_day:
                tmp_min = house_data[
                    (house_data.index.hour == 0)
                    & (house_data.index.minute == 0)
                    & (house_data.index.second == 0)
                ]
                house_data = house_data[house_data.index >= tmp_min.index[0]]

            for appliance in self.mask_app[1:]:
                # Check if appliance is available in this house
                house_map = self.REDD_METER_MAP.get(indice, {})
                if appliance in house_map:
                    meter_ids = house_map[appliance]
                    app_series = None
                    for m in meter_ids:
                        s = self._read_meter_series(h5file, indice, m)
                        s = s.resample("10s").mean()
                        if app_series is None:
                            app_series = s
                        else:
                            app_series = app_series.add(s, fill_value=0)

                    # Long NaN gaps mean the appliance was off (silence -> 0);
                    # short gaps are forward-filled. Without this, sparse REDD
                    # submeters leave scattered NaNs that drop whole windows.
                    app_series = self._fill_long_gaps_with_zero(app_series)
                    app_series = app_series.ffill(limit=6)
                    app_series[app_series < 5] = 0

                    appl_data = pd.DataFrame({appliance: app_series})

                    # Merge on index (inner join to align timestamps)
                    house_data = pd.merge(
                        house_data, appl_data, how="inner", left_index=True, right_index=True
                    )
                    house_data = house_data.clip(lower=0, upper=self.cutoff)
                    house_data = house_data.sort_index()

                    # Replace nan values by -1 during appliance activation status filtering
                    house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                    # Creating status
                    initial_status = (
                        (
                            (
                                house_data[appliance]
                                >= self.appliance_param[appliance]["min_threshold"]
                            )
                            & (
                                house_data[appliance]
                                <= self.appliance_param[appliance]["max_threshold"]
                            )
                        )
                        .astype(int)
                        .values
                    )

                    if self.use_status_from_kelly_paper:
                        house_data[appliance + "_status"] = self._compute_status(
                            initial_status,
                            self.appliance_param[appliance]["min_on_duration"],
                            self.appliance_param[appliance]["min_off_duration"],
                            self.appliance_param[appliance]["min_activation_time"],
                        )
                    else:
                        house_data[appliance + "_status"] = initial_status

                    # Finally replacing nan values put to -1 by nan
                    house_data[appliance] = house_data[appliance].replace(-1, np.nan)

            if self.synth_aggregate_apps is not None:
                already_loaded = set(self.mask_app[1:])
                for appliance in self.synth_aggregate_apps:
                    if appliance in already_loaded:
                        continue
                    house_map = self.REDD_METER_MAP.get(indice, {})
                    if appliance not in house_map:
                        house_data[appliance] = 0.0
                        continue
                    meter_ids = house_map[appliance]
                    app_series = None
                    for m in meter_ids:
                        s = self._read_meter_series(h5file, indice, m)
                        s = s.resample("10s").mean()
                        if app_series is None:
                            app_series = s
                        else:
                            app_series = app_series.add(s, fill_value=0)
                    app_series = self._fill_long_gaps_with_zero(app_series)
                    app_series = app_series.ffill(limit=6)
                    app_series[app_series < 5] = 0
                    app_series = app_series.clip(lower=0, upper=self.cutoff)
                    appl_data = pd.DataFrame({appliance: app_series})
                    house_data = pd.merge(
                        house_data, appl_data, how="left", left_index=True, right_index=True
                    )
                    house_data[appliance] = house_data[appliance].fillna(0.0)

        finally:
            h5file.close()

        if self.sampling_rate != "10s":
            house_data = house_data.resample(self.sampling_rate).mean()

        if self.synth_aggregate_apps is not None:
            house_data["synth_aggregate"] = sum(
                house_data[app].fillna(0.0)
                if app in house_data.columns
                else pd.Series(0.0, index=house_data.index)
                for app in self.synth_aggregate_apps
            )

        tmp_list = ["aggregate"]
        if "synth_aggregate" in house_data.columns:
            tmp_list.append("synth_aggregate")
        for appliance in self.mask_app[1:]:
            tmp_list.append(appliance)
            tmp_list.append(appliance + "_status")
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance + "_status"] = (
                        house_data[appliance + "_status"] > 0
                    ).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance + "_status"] = 0

        house_data = house_data[tmp_list]

        return house_data

    def _check_appliance_names(self):
        """
        Check appliances names for REDD case.
        """
        for appliance in self.mask_app:
            assert appliance in [
                "Fridge",
                "Dishwasher",
                "Microwave",
                "WashingMachine",
                "WasherDryer",
            ], f"Selected appliance unknown for REDD Dataset, got: {appliance}"
        return

    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return

    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))
