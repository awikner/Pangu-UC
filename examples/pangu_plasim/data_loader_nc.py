import os
from os import listdir
from os.path import join
import pickle
import cftime
from typing import Literal

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose
import torch

import xarray as xr
import numpy as np

from datetime import datetime
from typing import Literal

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose
from dateutil.relativedelta import relativedelta

def load_mean_std(mean_file, std_file):
   with xr.open_dataset(mean_file) as ds:
       mean_dict = {var: ds[var].values for var in ds.data_vars}
   with xr.open_dataset(std_file) as ds:
       std_dict = {var: ds[var].values for var in ds.data_vars}
   return mean_dict, std_dict

def datetime_class_from_calendar(calendar):
    datetime_class_dict = {'standard': cftime.DatetimeGregorian,
                           'Gregorian:': cftime.DatetimeGregorian,
                           'noleap': cftime.DatetimeNoLeap,
                           '365_day': cftime.DatetimeNoLeap,
                           'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
                           'all_leap': cftime.DatetimeAllLeap,
                           '366_day': cftime.DatetimeAllLeap,
                           '360_day': cftime.Datetime360Day,
                           'julian': cftime.DatetimeJulian}
    return datetime_class_dict[calendar]

class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, year_start, year_end, flag, surface_variables=None,
                 upper_air_variables=None, boundary_variables=None, boundary_dir="boundary_variables",
                 surface_mean_file = "surface_mean.nc", surface_std_file = "surface_std.nc",
                 upper_air_mean_file = "upper_air_mean.nc", upper_air_std_file = "upper_air_std.nc",
                 calendar = 'proleptic_gregorian', timedelta_hours = 6):
        super().__init__()
        self.data_dir = data_dir
        self.year_start = year_start
        self.year_end = year_end
        self.flag = flag
        self.calendar = calendar
        self.timedelta_hours = timedelta_hours
        self.datetime_class = datetime_class_from_calendar(self.calendar)
        self.timedelta = self.datetime_class(1, 1, 1, hour = self.timedelta_hours) - \
                         self.datetime_class(1, 1, 1, hour = 0)
        self.surface_variables = surface_variables or []
        self.upper_air_variables = upper_air_variables or []
        self.boundary_variables = boundary_variables or []
        self.boundary_dir = boundary_dir

        self.surface_mean, self.surface_std = load_mean_std(join(data_dir, surface_mean_file), join(data_dir, surface_std_file))
        self.upper_air_mean, self.upper_air_std = load_mean_std(join(data_dir, upper_air_mean_file), join(data_dir, upper_air_std_file))

        self.surface_transform = self._create_surface_transform()
        self.upper_air_transform = self._create_upper_air_transform()
        self.channel_seq = self.surface_variables + self.upper_air_variables

        self.boundary_ds = self._load_boundary_data()
        self.dates = self._get_dates()

    def __getitem__(self, index):
        surface_t, upper_air_t = self._get_data(self.dates[index])
        surface_t_1, upper_air_t_1 = self._get_data(self.dates[index + 1])

        start_time = self.dates[index].astype(int)
        end_time = self.dates[index + 1].astype(int)
        boundary_data = self._get_boundary_data(start_time, end_time)

        if self.flag == "train":
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, boundary_data
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, boundary_data, torch.tensor([start_time, end_time])

    def _get_data(self, date):
        #year = date.astype("datetime64[Y]").astype(int) + 1970
        file_name = join(self.data_dir, f"data_{date.year}.nc")
        ds = xr.open_dataset(file_name)

        time_index = ds['time'].values.astype('datetime64[D]').astype(int) == date.astype(int)
        time_value = ds['time'].values[time_index][0]

        surface_data = torch.stack([torch.from_numpy(ds[var].sel(time=time_value).values.astype(np.float32)) for var in self.surface_variables], dim=0)
        surface_data = self.surface_transform(surface_data)

        upper_air_data = torch.stack([self.upper_air_transform[pl](
            torch.from_numpy(np.stack([ds[var].sel(time=time_value, plev=pl).values for var in self.upper_air_variables], axis=0).astype(np.float32))
        ) for pl in self.upper_air_std], dim=1)

        return surface_data, upper_air_data

    def _get_boundary_data(self, start_time, end_time):
        batch_boundary_ds = self.boundary_ds.sel(time=slice(cftime.DatetimeNoLeap(start_time, 'seconds since 1970-01-01'), cftime.DatetimeNoLeap(end_time, 'seconds since 1970-01-01')))
        boundary_data = torch.tensor(np.stack([batch_boundary_ds[var].values for var in self.boundary_variables], axis=0), dtype=torch.float32)
        return boundary_data

    def __len__(self):
        return len(self.dates) - 1

    def _create_surface_transform(self):
        mean_seq = [self.surface_mean[var] for var in self.surface_variables]
        std_seq = [self.surface_std[var] for var in self.surface_variables]
        return Normalize(mean_seq, std_seq)

    def _create_upper_air_transform(self):
        normalize = {}
        for pl in self.upper_air_std:
            mean_seq = [self.upper_air_mean[var] for var in self.upper_air_variables]
            std_seq = [self.upper_air_std[pl] for _ in self.upper_air_variables]
            normalize[pl] = Normalize(mean_seq, std_seq)
        return normalize

# If the order of the variables in the mean and standard deviation dictionaries is assumed to match the order in the dataset follow above or else
# below implementation :
    
    # def _create_surface_transform(self):
    #     mean_seq = [self.surface_mean[var] for var in self.channel_seq if var in self.surface_variables]
    #     std_seq = [self.surface_std[var] for var in self.channel_seq if var in self.surface_variables]
    #     return Normalize(mean_seq, std_seq)

    # def _create_upper_air_transform(self):
    #     normalize = {}
    #     for pl in self.upper_air_std:
    #         mean_seq = [self.upper_air_mean[var] for var in self.channel_seq if var in self.upper_air_variables]
    #         std_seq = [self.upper_air_std[pl] for _ in range(len(mean_seq))]
    #         normalize[pl] = Normalize(mean_seq, std_seq)
    #     return normalize

    def surface_inv_transform(self):
        mean_seq = [self.surface_mean[var] for var in self.surface_variables]
        std_seq = [self.surface_std[var] for var in self.surface_variables]
        invTrans = Compose([
            Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
            Normalize([-x for x in mean_seq], [1.] * len(std_seq))
        ])
        return invTrans

    def upper_air_inv_transform(self):
        normalize = {}
        for pl in self.upper_air_std:
            mean_seq = [self.upper_air_mean[var] for var in self.upper_air_variables]
            std_seq = [self.upper_air_std[pl] for _ in self.upper_air_variables]
            invTrans = Compose([
                Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
                Normalize([-x for x in mean_seq], [1.] * len(std_seq))
            ])
            normalize[pl] = invTrans

        return normalize

    def _load_boundary_data(self):
        boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in os.listdir(join(self.data_dir, self.boundary_dir))]
        return xr.open_mfdataset(boundary_files, combine='nested', concat_dim='boundary')

    def _get_dates(self):
        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1)
        return xr.cftime_range(start_date, end_date, freq = '%dh' % self.timedelta_hours, calendar=self.calendar)

    def get_lat_lon(self):
        example_file = join(self.data_dir, f"data_{self.year_start}.nc")
        ds = xr.open_dataset(example_file)
        return ds["lat"].values, ds["lon"].values

