#
# ------------------------------------------------------------------------------
# Classifies given time series demand data.
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Mod that changes endpoint identification for better convergence
# ------------------------------------------------------------------------------
#
#
# Author: Sven Berendsen
#
# Changelog:
#
# 2021.10.18 - SBerendsen
# 2022.07.10 - SBerendsen - changed into a package
#
# ------------------------------------------------------------------------------
#
# Copyright 2022, Sven Berendsen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------------
#
# Classifies the given time series into several classes:
#
# - two peaks per day
# - peaking at the start of the day
# - peaking at the end of the day
# - no peak
# - does not make sense
#
# A more modular version, ready to be re-used in parameter adjustment.
#
# ------------------------------------------------------------------------------
#

# 0. Imports ===================================================================

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.graph_objects as go

# 1. Global vars ===============================================================

_logging = False
_extensive_log = False

# settings
_complex_bar_id = False  # whether it should use complex id-bar location determination
_max_search_range = 5  # how far to search for a local maxima
_max_search_range_check = 1  # how far to search for finding the local changed values
_max_allowed_concentration = 50.0  # [%] maximum allowed volume part at one timestep
_min_non_zero_needed = 8  # minimum number of non-zero entries required
_min_peak_distance = 6  # how far peaks should be apart to be counted as "distinct"
_min_num_max_level = 8  # how many entries need to be at least at top level for classing it as "one long peak"
_min_sum_viable = 60.0  # minimum daily sum from which the time series might be sensible
_min_hour_viable = 0.01  # beneath which the hourly value is taken as "zero"
_limit_nearly = 0.05  # what the cut-off point is for being "nearly" in another class
_lower_division = 2.0 / 3.0  # at which ratio to break between the lower two classes
_upper_division = 1.0 / 2.0  # at which ration to break between the upper two classes

# directory names
_dir_two_peaks = 'two_peaks/'
_dir_morning_peak = 'morning_peak/'
_dir_evening_peak = 'evening_peak/'
_dir_long_peak = 'long_peak/'
_dir_unclassified = 'unclassified/'
_dir_crap = 'crap/'

# valid classes
valid_classes = ['long_peak', 'two_peaks', 'one_peak']

# class label translators
dict_class = {
    'one_peak': 'One Peak',
    'two_peaks': 'Two Peaks',
    'crap': 'Data Problem',
    'no_class': 'Unclassified',
    'long_peak': 'Long Peak'
}
dict_subclass = {
    'morning_peak': 'Morning Peak',
    'evening_peak': 'Evening Peak',
    'None': '',
    'no_demand': 'No Demand',
    'too_little_demand': 'Little Demand',
    'too_much_demand': 'Extreme Demand',
    'high_concentration': 'Demand Spike',
    'no_morning_bar': 'Step Problem',
    'one_peak_equal_sides': 'Undecidable Peak Position'
}


# 1.1 Classes ##################################################################
class _hline(object):

    def __init__(self, x_min: float, x_max: float, y: float):
        self.x_max = x_max
        self.x_min = x_min
        self.y = y


class _position(object):

    def __init__(self, x: int, y: float):
        self.x = x
        self.y = y


class ClassificationInfo(object):

    def __init__(self, name: str, group: str, filename: str = ''):

        # general info
        self.name = name
        self.group = group  # grouped source of data
        self.classed = None
        self.sub_classed = 'None'

        # stats info
        self.daily_mean = None  # diurnal mean
        self.daily_max = None
        self.diurnal_mean = None  # diurnal mean
        self.diurnal_max = None
        self.diurnal_min = None
        self.df_diurnal = None  # df with the hourly mean

        # number of folks in household
        self.persons_min = None
        self.person_nearest = None  # number of persons nearest to mean demand
        self.persons_max = None

        # classes ys
        self.lower_spread = None
        self.upper_spread = None

        # position of the minima/maxima for adjustment
        # Could be improved by using a points class
        self.morning_max_x = None
        self.morning_max_y = None
        self.day_min_x = None
        self.day_min_y = None
        self.evening_max_x = None
        self.evening_max_y = None
        self.night_min_x = None
        self.night_min_y = None

        # other
        self.filename = filename

        # filled later
        self.hlines = []

    @classmethod
    def load_classify(cls,
                      file,
                      name: str,
                      group: str,
                      filename: str = '',
                      df_persons: pd.DataFrame = None,
                      testing: bool = False,
                      nearest_persons: bool = False):

        cls = ClassificationInfo(name, group, filename)

        df = pd.read_csv(file, parse_dates=True, index_col=0)

        if df_persons is not None:
            # set min viable from the table
            global _min_sum_viable
            _min_sum_viable = df_persons['Estimate Lower'].min()

        cls.classify(df, testing=testing)
        cls.abstraction_comparison()

        if df_persons is not None:

            # add person info
            cls.add_person_info(df_persons, nearest_persons)

        return cls

    @classmethod
    def load_df(cls,
                df: pd.DataFrame,
                name: str,
                group: str,
                testing=False,
                fn=''):

        cls = ClassificationInfo(name, group, fn)

        cls.classify(df, station_name=name, testing=testing)
        cls.abstraction_comparison(testing=testing)

        return (cls)

    def access_field(self, field_name: str):

        if (field_name == "morning_max_x"):
            return self.morning_max_x

        elif (field_name == "morning_max_y"):
            return self.morning_max_y

        elif (field_name == "day_min_x"):
            return self.day_min_x

        elif (field_name == "day_min_y"):
            return self.day_min__y

        elif (field_name == "evening_max_x"):
            return self.morning_max_x

        elif (field_name == "evening_max_y"):
            return self.morning_max_y

        elif (field_name == "night_min_x"):
            return self.night_min_x

        elif (field_name == "night_min_y"):
            return self.night_min_y

        else:
            print(f'Error: access_field: unknown field ID: {field_name}')
            exit(255)

    def classify(self,
                 df: pd.DataFrame,
                 station_name: str = '',
                 no_low_demand: bool = False,
                 testing: bool = False,
                 ignore_low_demand: bool = False):

        # sanity check
        if (len(df.columns) != 1):
            if station_name in df.columns:
                df_internal = pd.DataFrame(df[station_name])
            else:
                print('\nClassificationInfo.classify: given stationame is not '
                      'in df, taking first column')
                print(f'Station: {station_name}')
                if testing:
                    print(df)
                df_internal = pd.DataFrame(df[df.columns[0]])
        else:
            df_internal = df

        # prep
        if len(station_name) > 0:
            station = station_name
        else:
            station = self.name

        # daily stuff
        df_resampled_day = df_internal.resample("1D").sum()
        self.daily_mean = df_resampled_day.mean()[0]
        self.daily_max = df_resampled_day.max()[0]

        # get the diurnal mean
        df_hourly = df_internal.resample('1H').sum()
        self.df_diurnal = df_hourly.groupby(df_hourly.index.hour).mean()
        self.diurnal_mean = self.df_diurnal.mean()[0]
        self.diurnal_max = self.df_diurnal.max()[0]
        self.diurnal_min = self.df_diurnal.min()[0]

        # self.lower_spread = (1.0 - _lower_division) * self.diurnal_mean
        self.upper_spread = _upper_division * (
            self.diurnal_max - self.diurnal_mean) + self.diurnal_mean
        self.lower_spread = (1.0 - _lower_division) * (
            self.diurnal_mean - self.diurnal_min) + self.diurnal_min

        # ascertain that the minimum positive value is lower than "lower spread"
        if self.df_diurnal[self.name].min() > self.lower_spread:
            min_val = 10000000
            min2_val = 10000000
            for i in range(0, len(self.df_diurnal.index)):
                if self.df_diurnal.iloc[i, 0] < min_val:
                    min2_val = min_val
                    min_val = self.df_diurnal.iloc[i, 0]

            self.lower_spread = min_val + 0.5 * (min(
                abs(min_val - self.lower_spread), abs(min2_val - min_val)))

        # filter out one crap one: if the daily mean is too low
        if self.daily_mean <= 0.0 or self.diurnal_min < 0.0:
            self.classed = 'crap'
            self.sub_classed = 'no_demand'
            return

        # if the input values are too concentrated
        if _check_concentration(self.df_diurnal) >= _max_allowed_concentration:
            self.classed = 'crap'
            self.sub_classed = 'high_concentration'
            return

        # if no diurnal pattern has been found
        if self.df_diurnal.index[0] != 0 or self.df_diurnal.index[-1] != 23:
            self.classed = 'crap'
            self.sub_classed = 'nan_in_diurnal'
            return

        # classify in parts
        # doesn't matter for so small frames, but make it able to use SSE_X...
        self.df_diurnal["classified"] = -1
        self.df_diurnal["classified_nearly"] = -1
        pos_classified = self.df_diurnal.columns.get_loc("classified")
        pos_classified_nearly = self.df_diurnal.columns.get_loc(
            "classified_nearly")

        for i in range(0, len(self.df_diurnal.index)):
            self.df_diurnal.iat[i, pos_classified] = _classify_four_parts(
                self.df_diurnal.iloc[i][0], self.lower_spread,
                self.diurnal_mean, self.upper_spread, self.diurnal_max)

            self.df_diurnal.iat[
                i, pos_classified_nearly] = _classify_nearly_four_parts(
                    self.df_diurnal.iloc[i][0], self.lower_spread,
                    self.diurnal_mean, self.upper_spread, self.diurnal_max)

        # go through and get min/max points
        self.df_diurnal = _get_point_info(self.df_diurnal)

        self.refilter_classing(station,
                               no_low_demand=no_low_demand,
                               testing=testing,
                               ignore_low_demand=ignore_low_demand)

        return

    def refilter_classing(self,
                          station: str,
                          no_low_demand: bool = False,
                          testing: bool = False,
                          ignore_low_demand: bool = False):

        # reset what needs to be reset
        self.hlines = []

        # get supplementary info
        pos_point_info = self.df_diurnal.columns.get_loc("point_info")

        # start sorting.........................................................

        # first filter ones out which are plain crap

        # too low daily mean
        if (_min_sum_viable >= self.daily_mean) and not (no_low_demand
                                                         or ignore_low_demand):
            self.classed = 'crap'
            self.sub_classed = 'low_demand'
            return

        # if too many entries are zero
        counts_classes = self.df_diurnal['classified'].value_counts()

        if 0 in counts_classes.index:
            if (_min_non_zero_needed <
                    len(self.df_diurnal.index) - counts_classes[0]):
                self.classed = 'crap'
                self.sub_classed = 'few_data_points'
                return

        # otherwise distinguish between the main classes
        counts_point_info = self.df_diurnal['point_info'].value_counts()

        # long peak - add fuzzy "a bit of one level lower is ok"-test
        flag = 4 in counts_classes.index
        if _logging:
            print(f'Check for level 4 in distribution {flag}')

        if flag:
            spread, pos_start, pos_end = _get_level_spread(self.df_diurnal, 4)
            if (counts_classes[4] >= _min_num_max_level and spread >= 0.75):

                self.classed = 'long_peak'

                # get the characteristica
                if pos_start < pos_end:
                    df_temp = self.df_diurnal.iloc[pos_start:pos_end + 1]
                    mean_val = df_temp[df_temp.columns[0]].mean()

                    self.day_min_x = pos_start + 1
                    self.day_min_y = mean_val

                    self.hlines += [_hline(pos_start, pos_end, mean_val)]

                elif pos_end < pos_start:
                    df_temp = self.df_diurnal.iloc[pd.np.r_[:pos_end + 1,
                                                            pos_start:0]]
                    mean_val = df_temp[df_temp.columns[0]].mean()

                    self.day_min_x = pos_start
                    self.day_min_y = mean_val

                    self.hlines += [
                        _hline(0, pos_end, mean_val),
                        _hline(pos_start,
                               len(df_temp.index) - 1, mean_val)
                    ]

                # Common ones
                # TODO refactor for it!
                self.morning_max_x = pos_start
                self.morning_max_y = mean_val

                self.evening_max_x = pos_end
                self.evening_max_y = mean_val

                for i in range(0, len(self.df_diurnal.index)):
                    if self.df_diurnal.iloc[i, pos_point_info] == "min":
                        self.night_min_x = i
                        self.night_min_y = self.df_diurnal.iloc[i, 0]
                        break

                return

        # prep
        troughs = _get_troughs(self.df_diurnal)
        num_peaks, peaks = _get_num_distinct_peaks(self.df_diurnal)

        flag = ('max' in counts_point_info.index)
        if _logging:
            print(f'Check for absolute maxima in distribution {flag}')
            print(
                f'Num_peaks {num_peaks} and max points {counts_point_info["max"]}'
            )
        if flag:

            # TODO add fuzzy "if peak is just about above"
            # Case: Two Peaked
            if (num_peaks == 2 and counts_point_info["max"] >= 2):

                self.classed = 'two_peaks'

                # get stuff
                assert len(
                    peaks
                ) == 2, f'work_on_file: station {station}: too many peaks'

                # position of minimums
                pos_min_a = None
                pos_min_b = None

                for item in troughs:
                    if item.x < peaks[0].x or item.x > peaks[1].x:
                        if pos_min_a is None:
                            pos_min_a = item
                        elif pos_min_a.y > item.y:
                            pos_min_a = item

                    elif item.x > peaks[0].x and item.x < peaks[1].x:
                        if pos_min_b is None:
                            pos_min_b = item
                        elif pos_min_b.y > item.y:
                            pos_min_b = item

                if not (pos_min_a is None or pos_min_b is None):

                    # get night / day, assume discharge at night is lower
                    # TODO: add case analysis for "equal low day/night"
                    if pos_min_a.y < pos_min_b.y:
                        pos_night = pos_min_a
                        pos_day = pos_min_b

                        self.morning_max_x = peaks[0].x
                        self.morning_max_y = peaks[0].y
                        self.day_min_x = pos_day.x
                        self.day_min_y = pos_day.y
                        self.evening_max_x = peaks[1].x
                        self.evening_max_y = peaks[1].y
                        self.night_min_x = pos_night.x
                        self.night_min_y = pos_night.y

                    elif pos_min_a.y >= pos_min_b.y:
                        pos_day = pos_min_a
                        pos_night = pos_min_b

                        self.morning_max_x = peaks[1].x
                        self.morning_max_y = peaks[1].y
                        self.day_min_x = pos_day.x
                        self.day_min_y = pos_day.y
                        self.evening_max_x = peaks[0].x
                        self.evening_max_y = peaks[0].y
                        self.night_min_x = pos_night.x
                        self.night_min_y = pos_night.y

                    else:
                        print(
                            f'TwoPeaks: uncaught day/night: station {station}')
                        exit(255)

                return

            # case: one peak, deciding on whether morning (start of period) or evening (end of period)
            if (num_peaks == 1 and counts_point_info["max"] == 1):

                typus = _get_peak_position(self.df_diurnal, testing)

                if typus == 'morning':
                    self.classed = "one_peak"
                    self.sub_classed = "morning_peak"

                    # set the known one
                    self.morning_max_x = peaks[0].x
                    self.morning_max_y = peaks[0].y

                    # do the rest
                    try:
                        self._get_morning_peak_bar(testing)
                    except KeyError:
                        raise KeyError(
                            f'Problem in getting Morning Bar on station {self.name}'
                        )
                    except IndexError:
                        raise IndexError(
                            f'Problem in getting Morning Bar on station {self.name}'
                        )

                    return

                elif typus == 'evening':
                    self.classed = "one_peak"
                    self.sub_classed = "evening_peak"

                    self.evening_max_x = peaks[0].x
                    self.evening_max_y = peaks[0].y

                    # do the rest
                    try:
                        self._get_evening_peak_bar(testing)
                    except KeyError:
                        raise KeyError(
                            f'Problem in getting Evening Bar on station {self.name}'
                        )
                    except IndexError:
                        raise IndexError(
                            f'Problem in getting Evening Bar on station {self.name}'
                        )

                    return

                # don't yet know what to make out of this one
                elif typus == 'equal':
                    self.classed = "crap"
                    self.sub_classed = "one_peak_equal_sides"
                    return

            else:
                self.classed = "no_class"

        return

    def add_person_info(self,
                        demand_persons: pd.DataFrame,
                        nearest: bool = False):
        """
        demand_persons - Df with demand information.
        nearest - whether only "nearest" persons are wanted or all within range (default)
        """

        # checks
        assert ('Estimate Lower' in demand_persons.columns
                ), 'The dataframe needs a column headed "Estimate Lower"'
        assert ('Estimate Upper' in demand_persons.columns
                ), 'The dataframe needs a column headed "Estimate Upper"'
        assert ('Daily' in demand_persons.columns
                ), 'The dataframe needs a column headed "Daily"'

        # skip, if bad data
        if self.classed is None or self.daily_mean <= 0.0:
            return

        # check for if unrealistic high usage - if so, leave empty
        if self.daily_mean > demand_persons["Estimate Upper"].max():
            self.classed = 'crap'
            self.sub_classed = 'too_much_demand'
            return

        # check for if unrealistic low usage - if so, leave empty
        if self.daily_mean < demand_persons["Estimate Lower"].min():
            self.classed = 'crap'
            self.sub_classed = 'too_little_demand'
            return

        # find the right values
        df = demand_persons.sort_index()

        # TODO: Replace by list comprehension - though here the impact should be small
        #       Would be something along the lines "filter all values which are" and
        #       then take min/max index
        if nearest:
            for i in range(min(df.index), max(df.index) + 1):
                if df.loc[i]["Daily"] > self.daily_mean:
                    self.persons_min = i
                    break

            for i in range(max(df.index), min(df.index) - 1, -1):
                if df.loc[i]["Daily"] < self.daily_mean:
                    self.persons_max = i + 1
                    break

        else:
            for i in range(min(df.index), max(df.index) + 1):
                if df.loc[i]["Estimate Upper"] >= self.daily_mean:
                    self.persons_min = i
                    break

            for i in range(max(df.index), min(df.index) - 1, -1):
                if df.loc[i]["Estimate Lower"] <= self.daily_mean:
                    self.persons_max = i
                    break

        # FIXME Use list comprehension, is faster
        dist_min = 26000
        pos_min = -1
        pos_daily = df.columns.get_loc("Daily")
        for i in range(0, len(df.index)):
            if abs(df.iloc[i, pos_daily] - self.daily_mean) < dist_min:
                dist_min = abs(df.iloc[i, pos_daily] - self.daily_mean)
                pos_min = i

        self.person_nearest = df.index[pos_min]

        # sanity checks
        if self.persons_min is None:
            self.persons_min = self.person_nearest

        if (self.person_nearest < self.persons_min):
            self.persons_min = self.person_nearest

        if self.persons_max is None:
            self.persons_max = self.person_nearest

        if (self.person_nearest > self.persons_max):
            self.persons_max = self.person_nearest

        if (self.persons_min < min(df.index)):
            self.persons_min = min(df.index)

        if (self.persons_max > max(df.index)):
            self.persons_max = max(df.index)

        return

    def plot(self,
             output_dir: str,
             postfix: str = '',
             comment: str = '',
             into_dirs: bool = False):
        """
        Plots an classification object.

        Inputs:
            into_dirs   Whether it should be put into subdirs according to classification.

        TODO Replace by the plotly version - plot_display is still missing some data output
        """

        if (self.df_diurnal is None) or (self.classed is
                                         None) or (self.diurnal_mean <= 0.0):
            return

        if ("classified" not in self.df_diurnal.columns) or (
                "point_info" not in self.df_diurnal.columns):
            return

        # prep
        pos_classified = self.df_diurnal.columns.get_loc("classified")
        pos_point_info = self.df_diurnal.columns.get_loc("point_info")

        # main plot
        fig = plt.figure(figsize=(11, 6))
        sns.lineplot(x=self.df_diurnal.index, y=self.df_diurnal[self.name])
        # self.df_diurnal[self.name].plot(kind='line',
        #                                 grid=True,
        #                                 legend=False,
        #                                 figsize=(11, 6))

        plt.xticks([0, 5, 11, 17, 23], ["1", "6", "12", "18", "24"])
        plt.minorticks_on()

        # add backgrounds
        # walk through and colour according to group
        start = 0
        val = int(self.df_diurnal.iloc[0][pos_classified])

        for i in range(1, len(self.df_diurnal.index)):

            if self.df_diurnal.iloc[i][pos_point_info] == 'max':
                plt.axvline(x=i, color="black")

            if self.df_diurnal.iloc[i][pos_point_info] == 'min':
                plt.axvline(x=i, color="violet")

            # if changes
            if (val != int(self.df_diurnal.iloc[i][pos_classified])):

                # do the background colourisation
                plt.axvspan(xmin=max(start - 0.5, 0),
                            xmax=i - 0.5,
                            color=_get_colour(val),
                            alpha=0.25)

                # setups for next instance
                val = int(self.df_diurnal.iloc[i][pos_classified])
                start = i

        # finish last box
        plt.axvspan(xmin=start - 0.5,
                    xmax=min(len(self.df_diurnal.index) - 0.5, 23),
                    color=_get_colour(val),
                    alpha=0.25)

        # num min & max
        num_min = 0
        num_max = 0

        for i in range(0, len(self.df_diurnal.index)):
            if self.df_diurnal.iloc[i][pos_point_info] == 'max':
                num_max += 1

            if self.df_diurnal.iloc[i][pos_point_info] == 'min':
                num_min += 1

        # horizontal info lines
        plt.axhline(y=self.lower_spread, color='lightgray')
        plt.axhline(y=self.diurnal_mean, color='darkgray')
        plt.axhline(y=self.upper_spread, color='lightgray')
        plt.axhline(y=self.diurnal_max, color='darkgray')

        # add extra lines
        for line_data in self.hlines:
            # print('hline_info', self.classed, self.sub_classed, line_data.y,
            #       line_data.x_min, line_data.x_max)
            plt.hlines(y=line_data.y,
                       xmin=line_data.x_min - 0.5,
                       xmax=line_data.x_max + 0.5,
                       linewidth=2,
                       color='royalblue')

        # title & labels
        title = (
            f'Classification plot for station {self.name}\n'
            f'Average Daily Demand: {self.df_diurnal[self.name].sum():.2f} l/day'
            f' Classes as {self.classed} with subclass {self.sub_classed}')

        if (len(comment) != 0):
            title += '\n' + comment

        if self.persons_min is not None and self.persons_max is not None:
            if (self.persons_min == self.persons_max):
                title += f'\n{self.persons_min} persons in the household.'
            else:
                title += (
                    f'\nThe household has between {self.persons_min} and '
                    f'{self.persons_max} persons.')

        plt.title(title)
        plt.xlabel('Hour')
        plt.ylabel('Water Demand [l/hour]')

        # add point info

        # maxima
        x = []
        y = []

        if (self.morning_max_x is not None and self.morning_max_y is not None):
            x += [self.morning_max_x]
            y += [self.morning_max_y]

        if (self.evening_max_x is not None and self.evening_max_y is not None):
            x += [self.evening_max_x]
            y += [self.evening_max_y]

        if len(x) > 0:
            plt.plot(x, y, 'gx')

        # minima
        x = []
        y = []

        if (self.day_min_x is not None and self.day_min_y is not None):
            x += [self.day_min_x]
            y += [self.day_min_y]

        if (self.night_min_x is not None and self.night_min_y is not None):
            x += [self.night_min_x]
            y += [self.night_min_y]

        if len(x) > 0:
            plt.plot(x, y, 'g+', markersize=10)

        # export
        if into_dirs:

            out_dir = self._get_output_dir(output_dir)

        else:
            out_dir = output_dir

        if (len(postfix) == 0):
            fn = f'{out_dir}{self.name}.png'
        else:
            fn = f'{out_dir}{self.name}_{postfix}.png'

        plt.savefig(fn)

        # cleanups
        plt.close(fig)
        plt.close('all')

        return

    def plot_display(self,
                     output_dir: str,
                     postfix: str = '',
                     comment: str = '',
                     into_dirs: bool = False):
        """
        Plots an classification object, setup for a nicer visual output.

        Inputs:
            into_dirs   Whether it should be put into subdirs according to classification.
        """

        if (self.df_diurnal is None) or (self.classed is
                                         None) or (self.diurnal_mean <= 0.0):
            return

        if ("classified" not in self.df_diurnal.columns) or (
                "point_info" not in self.df_diurnal.columns):
            return

        # prep
        pos_classified = self.df_diurnal.columns.get_loc("classified")
        pos_point_info = self.df_diurnal.columns.get_loc("point_info")

        # generate series for nicer display (adds a buffer)
        x = list(range(-1, 25))
        y = [self.df_diurnal.iloc[x, 0]
             for x in range(-1, 24)] + [self.df_diurnal.iloc[0, 0]]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

        fig.update_layout(autosize=False,
                          width=900,
                          height=600,
                          margin=dict(l=50, r=100, b=20, t=90, pad=4),
                          template='seaborn')

        # add backgrounds
        # walk through and colour according to group
        start = 0
        val = int(self.df_diurnal.iloc[0][pos_classified])

        for i in range(1, len(self.df_diurnal.index)):

            if self.df_diurnal.iloc[i][pos_point_info] == 'max':
                fig.add_vline(
                    x=i,
                    line_width=2,
                    line_dash="solid",
                    line_color="black",
                    opacity=0.5,
                    annotation=dict(
                        text="Maxima",
                        textangle=90,  # angle following the Plotly rule!!! 
                        xanchor="left",
                        yanchor="top",
                        y=0.2))

            if self.df_diurnal.iloc[i][pos_point_info] == 'min':
                fig.add_vline(
                    x=i,
                    line_width=2,
                    line_dash="solid",
                    line_color="violet",
                    opacity=0.6,
                    annotation=dict(
                        text="Minima",
                        textangle=90,  # angle following the Plotly rule!!! 
                        xanchor="left",
                        yanchor="top"))

            # if changes
            if (val != int(self.df_diurnal.iloc[i][pos_classified])):

                # do the background colourisation
                fig.add_vrect(x0=start - 0.5,
                              x1=i - 0.5,
                              fillcolor=_get_colour(val),
                              opacity=0.15,
                              line_width=0)

                # setups for next instance
                val = int(self.df_diurnal.iloc[i][pos_classified])
                start = i

        # finish last box
        fig.add_vrect(x0=start - 0.5,
                      x1=23.5,
                      fillcolor=_get_colour(val),
                      opacity=0.15,
                      line_width=0)

        # horizontal info lines
        fig.add_hline(y=self.df_diurnal[self.df_diurnal.columns[0]].min(),
                      line_width=2,
                      line_dash="solid",
                      line_color='darkgray',
                      opacity=0.8,
                      annotation=dict(text="Min Value",
                                      xanchor="left",
                                      yanchor="middle",
                                      x=1.005))
        fig.add_hline(y=self.lower_spread,
                      line_width=2,
                      line_dash="solid",
                      line_color='darkgray',
                      opacity=0.5,
                      annotation=dict(text="Lower Divisor",
                                      xanchor="left",
                                      yanchor="middle",
                                      x=1.005))
        fig.add_hline(y=self.diurnal_mean,
                      line_width=2,
                      line_dash="solid",
                      line_color='darkgray',
                      opacity=0.8,
                      annotation=dict(text="Mean Value",
                                      xanchor="left",
                                      yanchor="middle",
                                      x=1.005))
        fig.add_hline(y=self.upper_spread,
                      line_width=2,
                      line_dash="solid",
                      line_color='darkgray',
                      opacity=0.5,
                      annotation=dict(text="Upper Divisor",
                                      xanchor="left",
                                      yanchor="middle",
                                      x=1.005))
        fig.add_hline(y=self.diurnal_max,
                      line_width=2,
                      line_dash="solid",
                      line_color='darkgray',
                      opacity=0.8,
                      annotation=dict(text="Max Value",
                                      xanchor="left",
                                      yanchor="middle",
                                      x=1.005))

        # add extra lines
        for line_data in self.hlines:
            # print('hline_info', self.classed, self.sub_classed, line_data.y,
            #       line_data.x_min, line_data.x_max)
            fig.add_shape(type="line",
                          x0=line_data.x_min - 0.5,
                          y0=line_data.y,
                          x1=line_data.x_max + 0.5,
                          y1=line_data.y,
                          line=dict(color='royalblue', width=4, dash='dot'))

        if self.classed in dict_class:
            label_class = dict_class[self.classed]
        else:
            label_class = self.classed
        if self.sub_classed in dict_subclass:
            label_subclass = dict_subclass[self.sub_classed]
        else:
            label_subclass = self.sub_classed

        title = (
            f'Classification plot for station {self.name}<br>'
            f'Average Daily Demand: {self.df_diurnal[self.name].sum():.2f} l/day;'
            f' Classes as {label_class} with subclass {label_subclass}')

        if (len(comment) != 0):
            title += '<br>' + comment

        if self.persons_min is not None and self.persons_max is not None:
            if (self.persons_min == self.persons_max):
                title += f'<br>{self.persons_min} persons in the household.'
            else:
                title += (
                    f'<br>The household has between {self.persons_min} and '
                    f'{self.persons_max} persons.')

        fig.update_layout(
            title={
                'text': title,
                # 'y': 0.5,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            })

        fig.update_xaxes(range=[-0.5, 23.5],
                         title_text='Time of Day [h]',
                         tick0=0,
                         dtick=3)
        fig.update_yaxes(title_text='Water Demand [l/hour]')

        # export
        if into_dirs:
            out_dir = self._get_output_dir(output_dir)

        else:
            out_dir = output_dir

        if len(postfix) < 1:
            fn = f'{out_dir}{self.name}.png'
        else:
            fn = f'{out_dir}{self.name}_{postfix}.png'

        fig.write_image(fn)

        # # add point info
        # # maxima
        # x = []
        # y = []

        # if (self.morning_max_x is not None and self.morning_max_y is not None):
        #     x += [self.morning_max_x]
        #     y += [self.morning_max_y]

        # if (self.evening_max_x is not None and self.evening_max_y is not None):
        #     x += [self.evening_max_x]
        #     y += [self.evening_max_y]

        # if len(x) > 0:
        #     plt.plot(x, y, 'gx')

        # # minima
        # x = []
        # y = []

        # if (self.day_min_x is not None and self.day_min_y is not None):
        #     x += [self.day_min_x]
        #     y += [self.day_min_y]

        # if (self.night_min_x is not None and self.night_min_y is not None):
        #     x += [self.night_min_x]
        #     y += [self.night_min_y]

        # if len(x) > 0:
        #     plt.plot(x, y, 'g+', markersize=10)

        return

    def _get_output_dir(self, output_dir: str):
        if self.classed == 'crap':
            return output_dir + _dir_crap

        elif self.classed == 'two_peaks':
            return output_dir + _dir_two_peaks

        elif self.classed == 'long_peak':
            return output_dir + _dir_long_peak

        elif self.classed == 'one_peak':

            if self.sub_classed == 'morning_peak':
                return output_dir + _dir_morning_peak

            elif self.sub_classed == 'evening_peak':
                return output_dir + _dir_evening_peak

            else:
                print(
                    f'Classification.plot: Unsupport output sub class {self.sub_classed} for class {self.classed}'
                )
                return output_dir + _dir_unclassified

        elif self.classed == 'no_class':
            return output_dir + _dir_unclassified

        else:
            print(
                f'Classification.plot: Unsupport output class {self.classed}')
            exit(255)

    def to_list(self):

        if 'abstract' in self.df_diurnal.columns:
            sum_abstract = self.df_diurnal['abstract'].sum()
        else:
            sum_abstract = np.nan

        return [
            self.group, self.classed, self.sub_classed, self.persons_min,
            self.person_nearest, self.persons_max, self.daily_max,
            self.daily_mean, sum_abstract, self.diurnal_mean, self.diurnal_max,
            self.morning_max_x, self.morning_max_y, self.day_min_x,
            self.day_min_y, self.evening_max_x, self.evening_max_y,
            self.night_min_x, self.night_min_y, self.filename
        ]

    def print_char_points(self):

        print('\nChar Points:')
        print(f'Mean Demand (daily): {self.daily_mean}')
        print(f'Typus:               {self.classed} {self.sub_classed}')
        if not self.classed == 'crap':
            print(
                f'Morning (x, y):      {self.morning_max_x:>2} {self.morning_max_y:>05.3f}'
            )
            print(
                f'Day (x, y):          {self.day_min_x:>2} {self.day_min_y:>05.3f}'
            )
            print(
                f'Evening (x, y):      {self.evening_max_x:>2} {self.evening_max_y:>05.3f}'
            )
            print(
                f'Night (x, y):        {self.night_min_x:>2} {self.night_min_y:>05.3f}'
            )
            for hline in self.hlines:
                print('HLine:              ', hline.x_min, hline.x_max)

    def force(self,
              what: str,
              sub_what: str,
              state_info,
              logging: bool = False,
              exception_text: str = ''):
        # TODO needs safeties
        # TODO extract into own class in huum_household, extending this one - much cleaner!

        if self.classed == what and self.sub_classed == sub_what:
            return

        if what == 'two_peaks':
            try:
                self._force_two_peaks(state_info)
            except KeyError:
                raise KeyError(
                    f'KeyError while forcing two peaks{exception_text}')
            except TypeError:
                raise TypeError(
                    f'TypeError while forcing two peaks{exception_text}')
            except Exception as e:
                raise Exception(
                    f'Error while forcing evening peak (complex){exception_text}'
                )

        elif what == 'one_peak' and sub_what == 'morning_peak':
            try:
                self._force_morning_peak(state_info)
            except KeyError:
                raise KeyError(
                    f'KeyError while forcing morning peak{exception_text}')
            except TypeError:
                raise TypeError(
                    f'TypeError while forcing morning peak{exception_text}')
            except Exception as e:
                raise Exception(
                    f'Error while forcing morning peak{exception_text}')

        elif what == 'one_peak' and sub_what == 'evening_peak':
            try:
                self._force_evening_peak(state_info)
            except KeyError:
                raise KeyError(
                    f'KeyError while forcing evening peak{exception_text}')
            except TypeError:
                raise TypeError(
                    f'TypeError while forcing evening peak{exception_text}')
            except Exception as e:
                raise Exception(
                    f'Error while forcing evening peak{exception_text}')

        elif what == 'long_peak':
            try:
                self._force_long_peak(state_info)
            except KeyError:
                raise KeyError(
                    f'Error while forcing long peak{exception_text}')
            except TypeError:
                raise TypeError(
                    f'TypeError while forcing long peak{exception_text}')
            except Exception as e:
                raise Exception(
                    f'Error while forcing evening peak (complex){exception_text}'
                )

        else:
            print(f'Error: ClassificationInfo.force: Unsupported Case {what}')
            exit(255)

    def _force_two_peaks(self, info):
        # TODO: re-implement using list comprehension

        # do maxima ------------------------------------------------------------
        if info.day_min_x < info.night_min_x:
            # get morning peak
            pos_max = -1.0
            val_max = -1.0
            for i in range(0, info.day_min_x + 1):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

            for i in range(info.night_min_x, len(self.df_diurnal.index)):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

            self.morning_max_x = pos_max
            self.morning_max_y = val_max

            # get evening peak
            pos_max = -1.0
            val_max = -1.0
            for i in range(info.day_min_x, info.night_min_x + 1):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

            self.evening_max_x = pos_max
            self.evening_max_y = val_max

        if info.day_min_x > info.night_min_x:
            # get morning peak
            pos_max = -1.0
            val_max = -1.0

            for i in range(info.night_min_x, info.day_min_x + 1):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

            self.morning_max_x = pos_max
            self.morning_max_y = val_max

            # get evening peak
            pos_max = -1.0
            val_max = -1.0
            for i in range(0, info.night_min_x + 1):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

            for i in range(info.day_min_x, len(self.df_diurnal.index)):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

            self.evening_max_x = pos_max
            self.evening_max_y = val_max

        # do minima ------------------------------------------------------------
        if self.morning_max_x > self.evening_max_x:

            # get daytime min
            pos_min = -1.0
            val_min = 1000000
            for i in range(0, self.evening_max_x):
                if (self.df_diurnal.iloc[i][0] < val_min):
                    val_min = self.df_diurnal.iloc[i][0]
                    pos_min = i

            for i in range(self.morning_max_x, len(self.df_diurnal.index)):
                if (self.df_diurnal.iloc[i][0] < val_min):
                    val_min = self.df_diurnal.iloc[i][0]
                    pos_min = i

            self.day_min_x = pos_min
            self.day_min_y = val_min

            # get nightime min
            pos_min = -1.0
            val_min = 1000000

            for i in range(self.evening_max_x, self.morning_max_x):
                if (self.df_diurnal.iloc[i][0] < val_min):
                    val_min = self.df_diurnal.iloc[i][0]
                    pos_min = i

            self.night_min_x = pos_min
            self.night_min_y = val_min

        if self.morning_max_x < self.evening_max_x:

            # get daytime min
            pos_min = -1.0
            val_min = 1000000

            for i in range(self.morning_max_x, self.evening_max_x):
                if (self.df_diurnal.iloc[i][0] < val_min):
                    val_min = self.df_diurnal.iloc[i][0]
                    pos_min = i

            self.day_min_x = pos_min
            self.day_min_y = val_min

            # get nightime min
            pos_min = -1.0
            val_min = 1000000

            for i in range(0, self.morning_max_x):
                if (self.df_diurnal.iloc[i][0] < val_min):
                    val_min = self.df_diurnal.iloc[i][0]
                    pos_min = i

            for i in range(self.evening_max_x, len(self.df_diurnal.index)):
                if (self.df_diurnal.iloc[i][0] < val_min):
                    val_min = self.df_diurnal.iloc[i][0]
                    pos_min = i

            self.night_min_x = pos_min
            self.night_min_y = val_min

    def _force_morning_peak(self, info):
        # TODO check for "wrap around"
        # FIXME do proper checking & taking old data into account

        # get maxima location
        val_max = -1.0
        pos_max = -1.0
        for i in range(
                info.morning_max_x - _max_search_range,
                min(len(self.df_diurnal.index),
                    info.morning_max_x + _max_search_range + 1)):
            if (self.df_diurnal.iloc[i][0] > val_max):
                val_max = self.df_diurnal.iloc[i][0]
                pos_max = i

        # deal with forward wrap-around
        # (iloc can deal with negative indices so no need for backwards wrap)
        if len(self.df_diurnal.index
               ) < info.morning_max_x + _max_search_range + 1:
            for i in range(
                    0, info.morning_max_x + _max_search_range + 1 -
                    len(self.df_diurnal.index)):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

        if (pos_max >= 0):
            self.morning_max_x = pos_max
        else:
            self.morning_max_x = len(self.df_diurnal.index) + pos_max
        self.morning_max_y = val_max

        # get the bar
        try:
            self._get_morning_peak_bar(testing=False)
        except KeyError:
            raise KeyError('Problem while getting morning bar')

        pass

    def _force_evening_peak(self, info):
        # TODO check for "wrap around"

        # get maxima location
        val_max = -1.0
        pos_max = -1.0
        for i in range(
                info.evening_max_x - _max_search_range,
                min(len(self.df_diurnal.index),
                    info.evening_max_x + _max_search_range + 1)):
            if (self.df_diurnal.iloc[i][0] > val_max):
                val_max = self.df_diurnal.iloc[i][0]
                pos_max = i

        # deal with case "forward wrap around"
        if (info.evening_max_x + _max_search_range + 1) > len(
                self.df_diurnal.index):
            for i in range(
                    0, info.evening_max_x + _max_search_range + 1 -
                    len(self.df_diurnal.index)):
                if (self.df_diurnal.iloc[i][0] > val_max):
                    val_max = self.df_diurnal.iloc[i][0]
                    pos_max = i

        if (pos_max >= 0):
            self.evening_max_x = pos_max
        else:
            self.evening_max_x = len(self.df_diurnal.index) + pos_max
        self.evening_max_y = val_max

        # get the bar
        try:
            self._get_evening_peak_bar(testing=False)
        except KeyError:
            raise KeyError('Problem while getting evening bar')

        pass

    def _force_long_peak(self, info):
        # TODO check for "wrap around"

        pos_point_info = self.df_diurnal.columns.get_loc("point_info")

        spread, pos_start, pos_end = _get_level_spread(self.df_diurnal, 4)
        spread_info, pos_start_info, pos_end_info = _get_level_spread(
            info.df_diurnal, 4)

        if abs(pos_start_info - pos_start) > 1 or abs(pos_end_info -
                                                      pos_end) > 1:
            # TODO check whether going for half-way position might be more realistic
            spread, pos_start, pos_end = _get_level_spread(self.df_diurnal, 3)

        self.classed = 'long_peak'

        # get the characteristica
        mean_val = np.nan
        if pos_start < pos_end:
            df_temp = self.df_diurnal.iloc[pos_start:pos_end + 1]
            mean_val = df_temp[df_temp.columns[0]].mean()

            self.day_min_x = pos_start + 1
            self.day_min_y = mean_val

            self.hlines += [_hline(pos_start, pos_end, mean_val)]

        elif pos_end < pos_start:
            df_temp = self.df_diurnal.iloc[pd.np.r_[:pos_end + 1, pos_start:0]]
            mean_val = df_temp[df_temp.columns[0]].mean()

            self.day_min_x = pos_start
            self.day_min_y = mean_val

            self.hlines += [
                _hline(0, pos_end, mean_val),
                _hline(pos_start, max(len(df_temp.index) - 1, pos_start),
                       mean_val)
            ]

        # common settings
        if np.isnan(mean_val):
            mean_val = self.df_diurnal[self.df_diurnal.columns[0]].mean()
        self.morning_max_x = pos_start
        self.morning_max_y = mean_val

        self.evening_max_x = pos_end
        self.evening_max_y = mean_val

        for i in range(0, len(self.df_diurnal.index)):
            if self.df_diurnal.iloc[i, pos_point_info] == "min":
                self.night_min_x = i
                self.night_min_y = self.df_diurnal.iloc[i, 0]
                break

    def _get_morning_peak_bar(self, testing: bool):

        if testing and _extensive_log:
            print('\nMorningBar:')

        # prep
        try:
            pos_classified = self.df_diurnal.columns.get_loc("classified")
            pos_point_info = self.df_diurnal.columns.get_loc("point_info")
        except KeyError:
            raise KeyError('MorningBar: Needed columns do not exist')

        # get maxima & minima pos
        pos_min, pos_max = _get_maxima_pos(self.df_diurnal, testing)

        self.night_min_x = pos_min.x
        self.night_min_y = pos_min.y

        if testing and _extensive_log:
            print('   ', pos_min.x, pos_min.y, pos_max.x, pos_max.y)

        # get as separate dataframe
        if pos_min.x < pos_max.x:
            df = self.df_diurnal.iloc[np.r_[(
                pos_max.x - len(self.df_diurnal.index)):0, :pos_min.x]]
        else:
            df = self.df_diurnal.iloc[pos_max.
                                      x:min(pos_min.x + 1,
                                            len(self.df_diurnal.index) - 1)]

        class_start, class_end = self._bar_get_classes(df)

        if testing and _extensive_log:
            print('   ClassSpan', class_start, class_end, self.classed)
            print(df)

        if self.classed == 'crap':
            return

        pos_start = -1
        pos_end = -1

        # get the positions
        for i in range(0, len(df.index)):
            if (df.iloc[i][pos_classified] == class_start) and pos_start == -1:
                pos_start = i

            elif pos_start != -1 and df.iloc[i][pos_classified] == class_end:
                pos_end = i

        if pos_end == -1:
            pos_end = pos_start

        # get mean, add bar
        if testing and _extensive_log:
            print('   getBar', pos_start, df.index[pos_start], pos_end,
                  df.index[pos_end], _complex_bar_id,
                  df.index[pos_start] > df.index[pos_end])

        if df.index[pos_start] > df.index[pos_end]:

            assert (
                0 in df.index and 23 in df.index
            ), f'First and last index should be 0 & 23 resp, not {df.index[0]} and {df.index[-1]}'

            val = df.iloc[0:pos_end][df.columns[0]].sum()
            val += df.iloc[pos_start:len(self.df_diurnal.index)][
                df.columns[0]].sum()
            length = pos_end - 0 + len(self.df_diurnal.index) - pos_start
            mean_val = val / length

            self.hlines += [
                _hline(df.index[pos_start],
                       len(self.df_diurnal.index) - 1, mean_val),
                _hline(0, max(df.index[pos_end] - 1, 0), mean_val)
            ]
        else:
            mean_val = df.iloc[pos_start:pos_end + 1][df.columns[0]].mean()

            self.hlines += [
                _hline(df.index[pos_start], df.index[pos_end], mean_val)
            ]

        if testing and _extensive_log:
            for hl in self.hlines:
                print('   HLine', hl.x_min, hl.x_max, hl.y)

        if _complex_bar_id:
            # determine evening point
            for i in range(min(pos_end + 1,
                               len(df.index) - 1),
                           max(pos_start, pos_end - _max_search_range), -1):
                if df.iloc[i, pos_point_info] == "local_max" and df.iloc[
                        i, 0] >= mean_val:
                    self.evening_max_x = df.index[i]
                    self.evening_max_y = df.iloc[i, 0]
                    break

            # if not found, search for first point where value > mean
            if self.evening_max_x is None:
                for i in range(min(pos_end + 1,
                                   len(df.index) - 1),
                               max(pos_start, pos_end - _max_search_range),
                               -1):
                    if df.iloc[i, 0] >= mean_val:
                        self.evening_max_x = df.index[i]
                        self.evening_max_y = df.iloc[i, 0]
                        break

        # if still not, use safety and go to first / second
        if self.evening_max_x is None:
            if pos_end - 1 < 0:
                self.evening_max_x = df.index[pos_end]
                self.evening_max_y = df.iloc[pos_end, 0]
            elif df.iloc[pos_end, 0] > df.iloc[pos_end - 1, 0]:
                self.evening_max_x = df.index[pos_end]
                self.evening_max_y = df.iloc[pos_end, 0]
            else:
                self.evening_max_x = df.index[pos_end - 1]
                self.evening_max_y = df.iloc[pos_end - 1, 0]

        # daytime values
        # TODO instead get the position of the minimum value in the range
        self.day_min_x = df.index[min(
            len(df.index) - 1, self.evening_max_x,
            int(round(abs(pos_start - pos_end) * 0.5)))]
        self.day_min_y = mean_val

        _asserting_types(self.evening_max_x, 'evening_max_x', int, 'int',
                         np.int64, 'np.int64')
        _asserting_types(self.day_min_x, 'day_min_x', int, 'int', np.int64,
                         'np.int64')
        _asserting_types(self.night_min_x, 'night_min_x', int, 'int', np.int64,
                         'np.int64')
        _asserting_types(self.evening_max_y, 'evening_max_y', float, 'float',
                         np.float64, 'np.float64')
        _asserting_types(self.day_min_y, 'day_min_y', float, 'float',
                         np.float64, 'np.float64')
        _asserting_types(self.night_min_y, 'night_min_y', float, 'float',
                         np.float64, 'np.float64')

        return

    def _bar_get_classes(self, df: pd.DataFrame):

        class_start = -1
        class_end = -1

        # get respective classes
        vals = df.value_counts(["classified"]).sort_values(ascending=False)

        # indices, but check for max class
        # TODO make safe (can fail on insufficient classes present)
        pos = -1
        for i in range(0, len(vals.index)):
            if vals.index[i][0] != 4 and vals.index[i][0] > 1:
                pos = i
                break

        if pos == -1:
            self.classed = 'crap'
            self.sub_classed = 'no_morning_bar'
            return class_start, class_end

        vals_red = vals.where(lambda x: x == vals[vals.index[pos]]).dropna(
        ).sort_index(ascending=False)

        class_start = vals_red.index[0][0]
        class_end = class_start

        # check if several ones should be combined
        if len(vals_red) > 1:
            for i in range(len(vals_red.index) - 1, 0, -1):
                if vals_red.index[i][0] > 1 and class_end > vals_red.index[i][
                        0]:
                    class_end = vals_red.index[i][0]

        assert class_start >= class_end, 'class_start needs to be >= class_end'
        assert class_start != -1 and class_end != -1, 'class_start or class_end are not yet set'

        return class_start, class_end

    def _get_evening_peak_bar(self, testing: bool):

        # prep
        try:
            pos_classified = self.df_diurnal.columns.get_loc("classified")
            pos_point_info = self.df_diurnal.columns.get_loc("point_info")
        except KeyError:
            raise KeyError('EveningBar: Column does not exist')

        # get maxima & minima pos
        pos_min, pos_max = _get_maxima_pos(self.df_diurnal, testing)

        self.night_min_x = pos_min.x
        self.night_min_y = pos_min.y

        # get as separate dataframe
        if pos_min.x > pos_max.x:
            df = self.df_diurnal.iloc[np.r_[(
                pos_min.x - len(self.df_diurnal.index)):0, :pos_max.x]]
        else:
            df = self.df_diurnal.iloc[pos_min.
                                      x:min(pos_max.x +
                                            1, len(self.df_diurnal.index))]

        class_start, class_end = self._bar_get_classes(df)

        if self.classed == 'crap':
            return

        pos_start = -1
        pos_end = -1

        # get the positions
        for i in range(len(df.index) - 1, 0, -1):
            if (df.iloc[i][pos_classified] == class_start) and pos_end == -1:
                pos_end = i

            elif pos_end != -1 and df.iloc[i][pos_classified] == class_end:
                pos_start = i

        if pos_start == -1:
            pos_start = pos_end

        # get mean, add bar
        if df.index[pos_start] > df.index[pos_end]:

            assert (
                0 in df.index and 23 in df.index
            ), f'First and last index should be 0 & 23 resp, not {df.index[0]} and {df.index[-1]}'

            # if not (0 in df.index and 23 in df.index):
            #     print(f'\nFirst and last index should be 0 & 23 resp, not {df.index[0]} and {df.index[-1]}\n')
            #     print(self.df_diurnal)
            #     print('\n')
            #     print(df)
            #     print('\n',pos_start, pos_end, class_start, class_end)
            #     exit()

            val = df.iloc[0:pos_end][df.columns[0]].sum()
            val += df.iloc[pos_start:len(self.df_diurnal.index)][
                df.columns[0]].sum()
            length = pos_end - 0 + len(self.df_diurnal.index) - pos_start
            mean_val = val / length

            self.hlines += [
                _hline(df.index[pos_start],
                       len(self.df_diurnal.index) - 1, mean_val),
                _hline(0, max(df.index[pos_end] - 1, 0), mean_val)
            ]
        else:
            mean_val = df.iloc[pos_start:pos_end + 1][df.columns[0]].mean()
            self.hlines += [
                _hline(df.index[pos_start], df.index[pos_end], mean_val)
            ]

        if _complex_bar_id:
            # determine evening point
            for i in range(pos_start,
                           min(pos_start + _max_search_range, pos_end), 1):
                if df.iloc[i, pos_point_info] == "local_max" and df.iloc[
                        i, 0] >= mean_val:
                    self.morning_max_x = df.index[i]
                    self.morning_max_y = df.iloc[i, 0]
                    break

            # if not found, get first place where val > mean
            if self.morning_max_x is None:
                for i in range(pos_start,
                               min(pos_start + _max_search_range, pos_end), 1):
                    if df.iloc[i, 0] > mean_val:
                        self.morning_max_x = df.index[i]
                        self.morning_max_y = df.iloc[i, 0]
                        break

        # if still not, use safety and go to first / second
        if self.morning_max_x is None:
            if pos_start + 1 == len(df):
                self.morning_max_x = df.index[pos_start]
                self.morning_max_y = df.iloc[pos_start, 0]
            elif df.iloc[pos_start, 0] > df.iloc[pos_start + 1, 0]:
                self.morning_max_x = df.index[pos_start]
                self.morning_max_y = df.iloc[pos_start, 0]
            else:
                self.morning_max_x = df.index[pos_start + 1]
                self.morning_max_y = df.iloc[pos_start + 1, 0]

        # daytime values
        # TODO instead get the position of the minimum value in the range
        self.day_min_x = df.index[min(
            len(df.index) - 1,
            max(self.morning_max_x,
                int(round(abs(pos_start - pos_end) * 0.5))))]
        self.day_min_y = mean_val

        _asserting_types(self.morning_max_x, 'morning_max_x', int, 'int',
                         np.int64, 'np.int64')
        _asserting_types(self.day_min_x, 'day_min_x', int, 'int', np.int64,
                         'np.int64')
        _asserting_types(self.night_min_x, 'night_min_x', int, 'int', np.int64,
                         'np.int64')
        _asserting_types(self.morning_max_y, 'morning_max_y', float, 'float',
                         np.float64, 'np.float64')
        _asserting_types(self.day_min_y, 'day_min_y', float, 'float',
                         np.float64, 'np.float64')
        _asserting_types(self.night_min_y, 'night_min_y', float, 'float',
                         np.float64, 'np.float64')

        return

    def force_y_update(self, what: str, sub_what: str, state_info):
        # TODO: needs safeties

        if what == 'one_peak' and sub_what == 'morning_peak':
            try:
                self._do_ybar_update(state_info)
                self.morning_max_y = self._do_y_point_update(
                    state_info.morning_max_x)
            except Exception as e:
                raise Exception('Error while forcing y_update morning peak')

        elif what == 'one_peak' and sub_what == 'evening_peak':
            try:
                self._do_ybar_update(state_info)
                self.evening_max_y = self._do_y_point_update(
                    state_info.evening_max_x)
            except Exception as e:
                raise Exception('Error while forcing y_update evening peak')

        elif what == 'two_peaks':

            self.morning_max_y = self._do_y_point_update(
                state_info.morning_max_x)
            self.evening_max_y = self._do_y_point_update(
                state_info.evening_max_x)
            self.day_min_y = self._do_y_point_update(state_info.day_min_x,
                                                     get_min=True)

        elif what == 'long_peak':

            self._do_ybar_update(state_info)

        else:
            raise ValueError(
                f'Error: ClassificationInfo.force: Unsupported Case {what}')

        return

    def _do_ybar_update(self, state_info):

        num_entries = 0
        sum_entries = 0.0
        for hline in self.hlines:

            # get df part for main bar
            if hline.x_min > hline.x_max:
                df = self.df_diurnal.iloc[np.r_[(
                    hline.x_min - len(self.df_diurnal.index)):0, :hline.x_max]]
            else:
                df = self.df_diurnal.iloc[
                    hline.x_min:min(hline.x_max + 1,
                                    len(self.df_diurnal.index) - 1)]
            num_entries += len(df.index)
            sum_entries += df[df.columns[0]].sum()

        mean_val = sum_entries / num_entries
        self.day_min_y = mean_val
        self.hlines = copy.deepcopy(state_info.hlines)
        self.hlines[0].y = mean_val

        # now extract min for evening-morning span
        # get df part for main bar
        if state_info.morning_max_x < state_info.evening_max_x:
            df = self.df_diurnal.iloc[np.r_[(
                state_info.evening_max_x -
                len(self.df_diurnal.index)):0, :state_info.morning_max_x]]
        else:
            df = self.df_diurnal.iloc[
                state_info.evening_max_x:min(state_info.morning_max_x + 1,
                                             len(self.df_diurnal.index) - 1)]

        self.night_min_y = df[self.df_diurnal.columns[0]].min()

        return

    def _do_y_point_update(self, x: int, get_min: bool = False):

        # TODO this is inefficient, improve it!

        start = x - _max_search_range_check
        if start < 0:
            start += 24

        end = x + _max_search_range_check + 1
        if end > 23:
            end -= 24

        if start > end:
            df = self.df_diurnal.iloc[np.r_[(
                start - len(self.df_diurnal.index)):0, :end]]
        else:
            df = self.df_diurnal.iloc[start:min(end + 1,
                                                len(self.df_diurnal.index) -
                                                1)]

        if get_min:
            return df[self.df_diurnal.columns[0]].min()
        else:
            return df[self.df_diurnal.columns[0]].max()

    def abstraction_comparison(self, testing=False):

        # safety
        assert self.df_diurnal is not None, "\nClassify needs to be run first!\n"

        # general stuff
        if self.classed not in valid_classes:
            return

        # assert self.classed in ['long_peak', 'two_peaks', 'one_peak'
        #                         ], f'\nUnsupported class: {self.classed}\n'

        # prep
        self.df_diurnal['abstract'] = np.nan
        pos_abstract = self.df_diurnal.columns.get_loc('abstract')
        pos_classified = self.df_diurnal.columns.get_loc("classified")
        self._enter_hlines()

        # do class specific stuff
        if self.classed == 'long_peak':
            for i in range(0, self.df_diurnal.shape[0]):
                if self.df_diurnal.iloc[i, pos_classified] <= 1 and np.isnan(
                        self.df_diurnal.iloc[i, pos_abstract]):
                    self.df_diurnal.iat[i, pos_abstract] = self.night_min_y

        elif self.classed == 'two_peaks':
            self.df_diurnal.iat[self.morning_max_x,
                                pos_abstract] = self.morning_max_y
            self.df_diurnal.iat[self.evening_max_x,
                                pos_abstract] = self.evening_max_y
            self.df_diurnal.iat[self.day_min_x, pos_abstract] = self.day_min_y

            # set min values
            if self.morning_max_x > self.evening_max_x:
                for i in range(self.evening_max_x + 1, self.morning_max_x):
                    if self.df_diurnal.iloc[i, pos_classified] <= 1:
                        self.df_diurnal.iat[i, pos_abstract] = self.night_min_y

            elif self.morning_max_x < self.evening_max_x:
                for i in range(0, self.morning_max_x):
                    if self.df_diurnal.iloc[i, pos_classified] <= 1:
                        self.df_diurnal.iat[i, pos_abstract] = self.night_min_y
                for i in range(self.evening_max_x, len(self.df_diurnal.index)):
                    if self.df_diurnal.iloc[i, pos_classified] <= 1:
                        self.df_diurnal.iat[i, pos_abstract] = self.night_min_y
            else:
                raise Exception('\nUnsupported morning/evening max x case.\n')

        elif self.classed == 'one_peak':
            assert self.sub_classed in [
                'morning_peak', 'evening_peak'
            ], f'\nUnsupported sub-class: {self.classed}\n'

            if self.sub_classed == 'morning_peak':
                self.df_diurnal.iat[self.morning_max_x,
                                    pos_abstract] = self.morning_max_y

                # get first non-nan pos
                pos_start = 0
                for i in range(self.morning_max_x - 1, -1, -1):
                    if not np.isnan(self.df_diurnal.iloc[i, pos_abstract]):
                        pos_start = i
                        break

                # fill in the min values
                for i in range(pos_start, self.morning_max_x):
                    if self.df_diurnal.iloc[i, pos_classified] <= 1:
                        self.df_diurnal.iat[i, pos_abstract] = self.night_min_y

                # do the other side
                for i in range(
                        len(self.df_diurnal.index) - 1, self.morning_max_x,
                        -1):
                    if not np.isnan(self.df_diurnal.iloc[i, pos_abstract]):
                        break
                    if self.df_diurnal.iloc[i, pos_classified] <= 1:
                        self.df_diurnal.iat[i, pos_abstract] = self.night_min_y

            elif self.sub_classed == 'evening_peak':

                self.df_diurnal.iat[self.evening_max_x,
                                    pos_abstract] = self.evening_max_y

                for i in range(self.evening_max_x + 1,
                               len(self.df_diurnal.index)):
                    if not np.isnan(self.df_diurnal.iloc[i, pos_abstract]):
                        break
                    if self.df_diurnal.iloc[i, pos_classified] <= 1:
                        self.df_diurnal.iat[i, pos_abstract] = self.night_min_y

                # check for case that the first non-nan is evening max x
                pos_start = 0
                flag = False
                for i in range(0, self.evening_max_x + 1):
                    if not np.isnan(self.df_diurnal.iloc[
                            i, pos_abstract]) and i == self.evening_max_x:
                        flag = True
                        break
                    elif not np.isnan(self.df_diurnal.iloc[i, pos_abstract]):
                        break

                if not flag:
                    for i in range(0, self.evening_max_x):
                        if not np.isnan(self.df_diurnal.iloc[i, pos_abstract]):
                            break
                        if self.df_diurnal.iloc[i, pos_classified] <= 1:
                            self.df_diurnal.iat[
                                i, pos_abstract] = self.night_min_y
            else:
                raise Exception(
                    f'\nUnsupported sub-class: {self.sub_classed}\n')

        # linear interpolate
        self._interpolate_abstraction()

        return

    def _enter_hlines(self):

        # safety
        assert 'abstract' in self.df_diurnal.columns, '\nColumn _abstract_ needs to exist!\n'

        pos_abstract = self.df_diurnal.columns.get_loc('abstract')
        for hline in self.hlines:
            for i in range(hline.x_min, hline.x_max + 1):
                self.df_diurnal.iat[i, pos_abstract] = hline.y

        return

    def _interpolate_abstraction(self):

        # prep
        pos_abstract = self.df_diurnal.columns.get_loc('abstract')

        # deal with the wrap-around case
        if not (np.isnan(self.df_diurnal.iloc[0, pos_abstract])
                or np.isnan(self.df_diurnal.iloc[-1, pos_abstract])):
            self.df_diurnal["abstract"] = self.df_diurnal[
                "abstract"].interpolate()
            return

        pos_last = -1  # last position before the end which isn't NaN
        pos_first = -1  # first position from the start which isn't NaN

        # get last non-nan
        if np.isnan(self.df_diurnal.iloc[-1, pos_abstract]):
            for i in range(len(self.df_diurnal.index) - 1, -1, -1):
                if not (np.isnan(self.df_diurnal.iloc[i, pos_abstract])):
                    pos_last = i
                    val_last = self.df_diurnal.iloc[i, pos_abstract]
                    break
        else:
            pos_last = len(self.df_diurnal.index) - 1

        # first non-nan
        if np.isnan(self.df_diurnal.iloc[0, pos_abstract]):
            for i in range(0, len(self.df_diurnal.index)):
                if not (np.isnan(self.df_diurnal.iloc[i, pos_abstract])):
                    pos_first = i
                    val_first = self.df_diurnal.iloc[i, pos_abstract]
                    break
        else:
            pos_first = 0

        # checks
        assert pos_first >= 0, '\npos_first should be set!\n'
        assert pos_last >= 0, '\npos_last should be set!\n'

        val_first = self.df_diurnal.iloc[pos_first, pos_abstract]
        val_last = self.df_diurnal.iloc[pos_last, pos_abstract]

        diff = pos_first + (len(self.df_diurnal.index) - pos_last)
        interval = (val_first - val_last) / diff

        # from before midnight
        if np.isnan(self.df_diurnal.iloc[-1, pos_abstract]):
            for i in range(pos_last + 1, len(self.df_diurnal.index)):
                self.df_diurnal.iat[
                    i, pos_abstract] = val_last + (i - pos_last) * interval

        # after midnight
        if np.isnan(self.df_diurnal.iloc[0, pos_abstract]):
            for i in range(0, pos_first):
                self.df_diurnal.iat[i, pos_abstract] = val_last + (
                    len(self.df_diurnal.index) - pos_last + i) * interval

        # deal with 'normal' interpolation
        self.df_diurnal["abstract"] = self.df_diurnal["abstract"].interpolate()

        return

    def plot_abstraction(self,
                         dir_out: str,
                         postfix: str = '',
                         into_dirs: bool = True):

        # general stuff
        if self.classed not in ['long_peak', 'two_peaks', 'one_peak']:
            return

        # fig = plt.figure(figsize=(11, 6))
        # sns.lineplot(x=dfm['hour'].unique(), y=dfm['val'], hue=dfm['cat'])

        self.df_diurnal.plot(y=[self.name, "abstract"],
                             kind="line",
                             figsize=(11, 6))

        plt.xticks([0, 5, 11, 17, 23], ["1", "6", "12", "18", "24"])
        plt.minorticks_on()

        plt.title(
            f'Comparison Abstraction to Measured for Diurnal Demand Pattern for station {self.name}'
        )
        plt.xlabel('Hour')
        plt.ylabel('Water Demand [l/hour]')

        if into_dirs:
            output_dir = self._get_output_dir(dir_out)
        else:
            output_dir = dir_out

        if (len(postfix) == 0):
            fn = f'{output_dir}{self.name}_comparison.png'
        else:
            fn = f'{output_dir}{self.name}_comparison_{postfix}.png'

        plt.savefig(fn)
        self.df_diurnal.to_csv(fn[:-4] + '.csv')

        return


# 2. Functions =================================================================


def _asserting_types(var, str_var: str, type_01, str_type_01: str, type_02,
                     str_type_02: str):
    assert type(var) is type_01 or type(
        var
    ) is type_02, f'{str_var} is neither {str_type_01} nor {str_type_02}, but {type(var)}'


def _check_concentration(df: pd.DataFrame):

    sum_demand = df[df.columns[0]].sum()
    df['concentration'] = 100 * df[df.columns[0]] / sum_demand

    return df['concentration'].max()


def _get_maxima_pos(df: pd.DataFrame, testing: bool):

    val_min = 100000000.0
    pos_min = -1
    val_max = -100.0
    pos_max = -1

    for i in range(0, len(df.index)):

        if (df.iloc[i][df.columns[0]] < val_min):
            val_min = df.iloc[i][df.columns[0]]
            pos_min = i

        if (df.iloc[i][df.columns[0]] > val_max):
            val_max = df.iloc[i][df.columns[0]]
            pos_max = i

    return _position(pos_min, val_min), _position(pos_max, val_max)


def _get_peak_position(df: pd.DataFrame, testing: bool):

    # find longest minimum span
    pos_classified = df.columns.get_loc("classified")
    min_classified = df["classified"].min()
    pos_point_info = df.columns.get_loc("point_info")
    list_spans = []
    range_start = -1

    # testing stuff
    if testing:
        print('\n', df, '\n')
        print('min_classified', min_classified)

    # go through once
    for i in range(0, len(df.index)):
        if (df.iloc[i][pos_classified] == min_classified
                and range_start == -1):
            range_start = i

        elif (df.iloc[i][pos_classified] != min_classified
              and range_start != -1):
            list_spans.append([range_start, i - 1, i - range_start - 1])
            range_start = -1

    # deal with end-of-series
    if range_start != -1:
        list_spans.append(
            [range_start,
             len(df.index) - 1,
             len(df.index) - range_start])

    if testing:
        print('list_spans:', list_spans)

    # deal with case: start and end are both lowest class
    if df.iloc[0][pos_classified] == min_classified and df.iloc[-1][
            pos_classified] == min_classified:
        list_spans[0][0] = list_spans[0][1]
        list_spans[0][1] = range_start
        list_spans[0][2] += len(df.index) - range_start
        del list_spans[-1]

    if testing:
        print('list_spans, start/end check:', list_spans)

    # get pos of maximum (accounts for multiple, non-unique peaks)
    pos_max = -1
    val_max = -1
    pos_min = -1
    val_min = 10000000
    for i in df.index:
        if (df.iloc[i][pos_point_info] == 'max' and val_max < df.iloc[i][0]):
            pos_max = i
            val_max = df.iloc[i][0]
        if (df.iloc[i][pos_point_info] == 'min' and val_min > df.iloc[i][0]):
            pos_min = i
            val_min = df.iloc[i][0]

    # get pos of largest extend
    pos_item = -1
    for i, item in enumerate(list_spans):
        if (pos_item == -1):
            pos_item = i
        elif (item[2] > list_spans[pos_item][2]):
            pos_item = i

    # deal with multiple "longest" ones
    num = 0
    pos_list = []
    for i, item in enumerate(list_spans):
        if (list_spans[pos_item][2] == item[2]):
            num += 1
            pos_list += [i]

    # if more than one occurance
    if num > 1:

        print(
            f'Warning: get_peak_position: For station {df.columns[0]}: '
            f'Duplicate length max spans: {len(pos_list)} - combining where possible'
        )

        # determine whether the max is between them or not and act appropiately
        # TODO: fix for wrap-arounds
        # for i in range(num - 2, -1, -1):
        #     if not (list_spans[pos_list[i]][1] < pos_max <
        #             list_spans[pos_list[i + 1]][1]):
        #         list_spans[pos_list[i]][1] = list_spans[pos_list[i + 1]][1]
        #         del list_spans[pos_list[i + 1]]

    if testing:
        print('pos_max, pos_item', pos_max, pos_item)
        print('selected_item', list_spans[pos_item])

    # then decide from the max, where to is the longest distance
    dist_before = -1
    dist_after = -1

    if not (-len(list_spans) <= pos_item < len(list_spans)):
        print('\nError: get_peak_position: List index out of range')
        print('\n', df)
        print('\npos_max, pos_item', pos_max, pos_item)
        print(list_spans)
        exit(255)

    # position is at the start of a span
    if (pos_max == list_spans[pos_item][0]):
        if testing:
            print('at start')
        dist_before = 0
        dist_after = min(len(df.index), list_spans[pos_item][1]) - pos_max

    # position is at the end of a span
    elif (pos_max == list_spans[pos_item][1]):
        if testing:
            print('at end')
        dist_before = 0
        dist_after = max(len(df.index), list_spans[pos_item][1]) - pos_max

    # after the end of the biggest span
    elif (pos_max > list_spans[pos_item][1]):
        if testing:
            print('after end')
        dist_before = pos_max - list_spans[pos_item][1]
        dist_after = len(df.index) - pos_max + list_spans[pos_item][0]

    # before the start of the biggest part
    elif (pos_max < list_spans[pos_item][0]):
        if testing:
            print('before start')
        dist_before = len(df.index) - list_spans[pos_item][1] + pos_max
        dist_after = list_spans[pos_item][0] - pos_max

    # between the start and end == wrap-around
    elif (pos_max >= list_spans[pos_item][0]
          and pos_max <= list_spans[pos_item][1]):
        if testing:
            print('in between')
        # print(list_spans[pos_item][0], list_spans[pos_item][1])
        dist_before = pos_max - list_spans[pos_item][0]
        dist_after = list_spans[pos_item][1] - pos_max

    else:
        print(
            'Error: get_peak_position: uncaught case of longest distance determination'
        )
        print(
            f'{df.columns[0]} {list_spans[pos_item][0]} {list_spans[pos_item][1]} {pos_max}'
        )
        exit(255)

    if testing:
        print('dist_before, dist_after', dist_before, dist_after, pos_max,
              pos_min)

    # if distances ares equal, try to adjust it for distance from lowest point
    if (dist_before - dist_after == 0):
        if pos_max > pos_min:
            dist_before = pos_max - pos_min
            dist_after = pos_min + len(df.index) - pos_max
        elif pos_max < pos_min:
            dist_before = pos_max + len(df.index) - pos_min
            dist_after = pos_min - pos_max

        # if still equal after this, je ne sais pas!

    if (dist_before > dist_after):
        return 'evening'

    elif (dist_before < dist_after):
        return 'morning'

    elif (dist_before == dist_after):
        return 'equal'

    else:
        print(
            'Error: get_peak_position: uncaught case of position determination'
        )
        print(f'{dist_before} {dist_after}')
        exit(255)


def _get_num_distinct_peaks(df: pd.DataFrame):

    # prep
    num = 0
    last_peak = -1
    last_max = -1
    pos_point_info = df.columns.get_loc("point_info")

    unique_peaks = []
    current_pos = -1

    for i in range(0, len(df.index)):
        if df.iloc[i][pos_point_info] == 'max':
            if last_peak == -1 or (i - last_max) >= _min_peak_distance:
                # record last peak
                unique_peaks += [_position(i, df.iloc[i][0])]
                current_pos += 1

                # increment
                num += 1
                last_peak = i

            last_max = i

            # for other max check for maximum height
            if (unique_peaks[current_pos].y < df.iloc[i][0]):
                unique_peaks[current_pos] = _position(i, df.iloc[i][0])

    # check for wrap-around
    if df.iloc[0][pos_point_info] == 'max' and df.iloc[
            len(df.index) - 1][pos_point_info] == 'max':
        num -= 1

        # deal with recording
        if (unique_peaks[0].y > unique_peaks[-1].y):
            del unique_peaks[-1]

        elif (unique_peaks[0].y <= unique_peaks[-1].y):
            del unique_peaks[0]

    assert len(
        unique_peaks
    ) == num, f'unqual num peaks {num} & remembered positions {len(unique_peaks)}'

    return num, unique_peaks


def _get_troughs(df: pd.DataFrame):

    troughs = []
    pos_point_info = df.columns.get_loc("point_info")

    for i in range(0, len(df.index)):
        if df.iloc[i][pos_point_info] == 'min' or df.iloc[i][
                pos_point_info] == 'local_min':
            troughs += [_position(i, df.iloc[i][0])]

    return troughs


def _get_level_spread(df: pd.DataFrame, level: float):

    # TODO Take time jump into account

    # prep
    pos_classified = df.columns.get_loc("classified")
    pos_start = -1
    pos_last = -1
    num = 0

    for i in range(0, len(df.index)):
        if (df.iloc[i][pos_classified] >= level and pos_start == -1):
            pos_start = i
            pos_last = i
            num += 1
        elif (df.iloc[i][pos_classified] >= level):
            pos_last = i
            num += 1

    spread = num / (pos_last - pos_start + 1)

    return spread, pos_start, pos_last


def _get_point_info(df: pd.DataFrame):

    # prep
    df["point_info"] = 'point'

    pos_classified = df.columns.get_loc("classified")
    pos_point_info = df.columns.get_loc("point_info")

    # walk through and colour according to group
    val = int(df.iloc[0][pos_classified])
    val_before = int(df["classified"].max()) + 1
    max_val = -1000.0
    max_val_pos = -1
    min_val = 100000.0
    min_val_pos = -1

    # now onto general
    for i in range(0, len(df.index)):

        # max val stuff
        if (df.iloc[i][0] > max_val):
            max_val = df.iloc[i][0]
            max_val_pos = i

        # min val stuff
        if (df.iloc[i][0] < min_val):
            min_val = df.iloc[i][0]
            min_val_pos = i

        # if changes
        if (val != int(df.iloc[i][pos_classified])):

            # do max value
            if (val == 4):
                df.iat[max_val_pos, pos_point_info] = 'max'

            # go for min value
            if (val < int(df.iloc[i][pos_classified]) and val < val_before):
                df.iat[min_val_pos, pos_point_info] = 'min'

            # setups for next instance
            val_before = val
            val = int(df.iloc[i][pos_classified])
            max_val = df.iloc[i][0]
            max_val_pos = i
            min_val = df.iloc[i][0]
            min_val_pos = i

    # fix for last entry max
    # FIXME put into the above loop
    if int(df.iloc[i]
           [pos_classified]) == 4 and df.iloc[-1][0] > df.iloc[-2][0]:
        df.iat[len(df.index) - 1, pos_point_info] = 'max'

    # check first position for wrap around
    if df.iloc[0][pos_point_info] == 'max' and df.iloc[0][0] < df.iloc[-1][0]:
        df.iat[0, pos_point_info] = 'point'

    # check last position for wrap around
    if df.iloc[-1][pos_point_info] == 'max' and df.iloc[0][0] > df.iloc[-1][0]:
        df.iat[0, pos_point_info] = 'point'

    # finish last entries
    if (val < int(df.iloc[i][pos_classified])):
        df.iat[min_val_pos, pos_point_info] = 'min'

    if (val == 4):
        df.iat[max_val_pos, pos_point_info] = 'max'

    # check for wrap-around
    if (df.iloc[0][pos_classified] == 4 and df.iloc[-1][pos_classified] == 4):

        flag = True

        # find first max from the start
        pos_start = -1
        for i in range(0, len(df.index)):
            if (df.iloc[i][pos_point_info] == 'max'):
                pos_start = i
                break
            if df.iloc[i][pos_classified] != 4:
                flag = False
                break

        # find first max from the end
        pos_end = -1
        for i in range(len(df.index) - 1, -1, -1):
            if (df.iloc[i][pos_point_info] == 'max'):
                pos_end = i
                break
            if df.iloc[i][pos_classified] != 4:
                flag = False
                break

        # remove the max, which ever is lower
        if flag:
            if not (pos_start == -1 or pos_end == -1):
                if (df.iloc[pos_start][0] > df.iloc[pos_end][0]):
                    df.iat[pos_end, pos_point_info] = 'point'

                elif (df.iloc[pos_start][0] < df.iloc[pos_end][0]):
                    df.iat[pos_start, pos_point_info] = 'point'

                else:
                    df.iat[pos_end, pos_point_info] = 'point'

    # deal with wrap around local minima

    # first point
    if df.iloc[0][0] < df.iloc[1][0] and df.iloc[0][0] < df.iloc[-1][
            0] and df.iloc[0, pos_point_info] != 'min':
        df.iat[0, pos_point_info] = 'local_min'

    # last point
    if df.iloc[-1][0] < df.iloc[-2][0] and df.iloc[-1][0] < df.iloc[0][
            0] and df.iloc[-1, pos_point_info] != 'min':
        df.iat[-1, pos_point_info] = 'local_min'

    # now onto general local minima
    for i in range(0, len(df.index)):

        # safeties
        if (i == len(df.index) - 1):
            next_index = 0
        else:
            next_index = i + 1

        # local minima
        if i > 0 and i < len(df.index):
            if df.iloc[i][0] < df.iloc[i - 1][0] and df.iloc[i][0] < df.iloc[
                    next_index][0] and df.iloc[i, pos_point_info] != 'min':
                df.iat[i, pos_point_info] = 'local_min'

    # deal with wrap-around local maxima

    # first point
    if df.iloc[0][0] > df.iloc[1][0] and df.iloc[0][0] > df.iloc[-1][
            0] and df.iloc[0, pos_point_info] != 'max':
        df.iat[0, pos_point_info] = 'local_max'

    # last point
    if df.iloc[-1][0] > df.iloc[-2][0] and df.iloc[-1][0] > df.iloc[0][
            0] and df.iloc[-1, pos_point_info] != 'max':
        df.iat[-1, pos_point_info] = 'local_max'

    # now onto general local maxima
    for i in range(0, len(df.index)):

        # safeties
        if (i == len(df.index) - 1):
            next_index = 0
        else:
            next_index = i + 1

        # local maxima
        if i > 0 and i < len(df.index):
            if df.iloc[i][0] > df.iloc[i - 1][0] and df.iloc[i][0] > df.iloc[
                    next_index][0] and df.iloc[i, pos_point_info] != 'max':
                df.iat[i, pos_point_info] = 'local_max'

    # a sanity check...

    return df


def _if_above(val: float, val_center: float, val_above: float, return_val: int,
              default_return: int):

    if (val > val_center) and (val <= val_center +
                               (val_above - val_center) * _limit_nearly):
        return return_val

    return default_return


def _if_below(val: float, val_center: float, val_below: float, return_val: int,
              default_return: int):

    if (val < val_center) and (val >= val_center -
                               (val_center - val_below) * _limit_nearly):
        return return_val

    return default_return


def _classify_nearly_four_parts(val: float, lower_spread: float,
                                series_mean: float, upper_spread: float,
                                series_max: float):

    # around min_hour_viable
    val_return = _if_below(val, _min_hour_viable, 0.0, 1, -1)
    if val_return > -1:
        return val_return

    val_return = _if_above(val, _min_hour_viable, lower_spread, 0, -1)
    if val_return > -1:
        return val_return

    # around lower_spread
    val_return = _if_below(val, lower_spread, _min_hour_viable, 2, -1)
    if val_return > -1:
        return val_return

    val_return = _if_above(val, lower_spread, series_mean, 1, -1)
    if val_return > -1:
        return val_return

    # around mean
    val_return = _if_below(val, series_mean, lower_spread, 3, -1)
    if val_return > -1:
        return val_return

    val_return = _if_above(val, series_mean, upper_spread, 2, -1)
    if val_return > -1:
        return val_return

    # around upper spread
    val_return = _if_below(val, upper_spread, series_mean, 4, -1)
    if val_return > -1:
        return val_return

    val_return = _if_above(val, upper_spread, series_max, 3, -1)
    if val_return > -1:
        return val_return

    return -1


def _classify_four_parts(val: float, lower_spread: float, series_mean: float,
                         upper_spread: float, series_max: float):

    if val == _min_hour_viable:
        return 0

    elif val <= lower_spread:
        return 1

    elif val <= series_mean:
        return 2

    elif val <= upper_spread:
        return 3

    elif val <= series_max:
        return 4

    else:
        return -1


def _get_colour(val: int):

    if (val == 0):
        return 'brown'

    elif (val == 1):
        return 'blue'

    elif (val == 2):
        return 'green'

    elif (val == 3):
        return 'orange'

    elif (val == 4):
        return 'red'

    else:
        return 'black'


# 3. Main Exec =================================================================
if __name__ == '__main__':

    raise NotImplementedError
