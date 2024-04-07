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

class DatasetFromFolder(Dataset):
   def __init__(self, data_dir, year_start, year_end, flag, surface_variables=None, boundary_dir="boundary_variables"):
       super().__init__()
       self.data_dir = data_dir
       self.year_start = year_start
       self.year_end = year_end
       self.flag = flag
       self.surface_variables = surface_variables or []
       self.boundary_dir = boundary_dir

       self.surface_mean, self.surface_std = load_mean_std(join(data_dir, "surface_mean.nc"), 
                                                            join(data_dir, "surface_std.nc"))
       self.upper_air_mean, self.upper_air_std = load_mean_std(join(data_dir, "upper_air_mean.nc"),
                                                               join(data_dir, "upper_air_std.nc"))

       self.surface_transform = self._create_surface_transform()
       self.upper_air_transform = self._create_upper_air_transform()

       self.boundary_ds = self._load_boundary_data()
       self.dates = self._get_dates()

   def __getitem__(self, index):
       surface_t, upper_air_t = self._get_data(self.dates[index])
       surface_t_1, upper_air_t_1 = self._get_data(self.dates[index + 1])
       if self.flag == "train":
           return surface_t, upper_air_t, surface_t_1, upper_air_t_1
       return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
           self.dates[index].astype(int), self.dates[index + 1].astype(int)
       ])

   def _get_data(self, date):
       year = date.astype("datetime64[Y]").astype(int) + 1970
       file_name = join(self.data_dir, f"data_{year}.nc")
       ds = xr.open_dataset(file_name)

       time_index = ds['time'].values.astype('datetime64[D]').astype(int) == date.astype(int)
       time_value = ds['time'].values[time_index][0]

       surface_data = np.stack([ds[var].sel(time=time_value).values for var in self.surface_variables], axis=0)
       surface_data = torch.from_numpy(surface_data.astype(np.float32))
       surface_data = self.surface_transform(surface_data)

       upper_air_data = torch.stack([self.upper_air_transform[pl](
           torch.from_numpy(np.stack([ds[var].sel(time=time_value, plev=pl).values for var in self.upper_air_mean], axis=0).astype(np.float32))
       ) for pl in self.upper_air_std], dim=1)

       return surface_data, upper_air_data

   def __len__(self):
       return len(self.dates) - 1

   def _create_surface_transform(self):
       mean_seq, std_seq = [], []
       for var in self.surface_variables:
           mean_seq.append(self.surface_mean[var])
           std_seq.append(self.surface_std[var])
       return Normalize(mean_seq, std_seq)

   def _create_upper_air_transform(self):
       normalize = {}
       for pl in self.upper_air_std:
           mean_seq, std_seq = [], []
           for var in self.upper_air_mean:
               mean_seq.append(self.upper_air_mean[var])
               std_seq.append(self.upper_air_std[pl])
           normalize[pl] = Normalize(mean_seq, std_seq)
       return normalize

   def surface_inv_transform(self):
    mean_seq, std_seq = [], []
    for var in self.surface_variables:
        mean_seq.append(self.surface_mean[var])
        std_seq.append(self.surface_std[var])

    invTrans = Compose([
        Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
        Normalize([-x for x in mean_seq], [1.] * len(std_seq))
    ])
    return invTrans

   def upper_air_inv_transform(self):
        normalize = {}
        for pl in self.upper_air_std:
            mean_seq, std_seq = [], []
            for var in self.upper_air_mean:
                mean_seq.append(self.upper_air_mean[var])
                std_seq.append(self.upper_air_std[pl])

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
       start_date = np.datetime64(f"{self.year_start}-01-01")
       end_date = np.datetime64(f"{self.year_end + 1}-01-01")
       return np.arange(start_date, end_date, dtype='datetime64[D]')

   def get_lat_lon(self):
       example_file = join(self.data_dir, f"data_{self.year_start}.nc")
       ds = xr.open_dataset(example_file)
       return ds["lat"].values, ds["lon"].values


# # Add these imports
# import cf_xarray
# import cftime

# def surface_transform(mean_path, std_path):
#     # ... (same as before)

# def upper_air_transform(mean_path, std_path):
#     # ... (same as before)

# def surface_inv_transform(mean_path, std_path):
#     # ... (same as before)

# def upper_air_inv_transform(mean_path, std_path):
#     # ... (same as before)

# class DatasetFromFolder(Dataset):
#     def __init__(self, dataset_file, flag: Literal["train", "test", "valid"], surface_variables=None):
#         super().__init__()
#         self.dataset_file = dataset_file
#         self.flag = flag
#         self.surface_variables = surface_variables or []  # Default to empty list if not provided

#         self.surface_transform, self.surface_variables_full = surface_transform(join("data", "surface_mean.pkl"), 
#                                                                                 join("data", "surface_std.pkl"))
#         self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join("data", "upper_air_mean.pkl"), 
#                                                                                                          join("data", "upper_air_std.pkl"))
        
#         self.ds = xr.open_dataset(self.dataset_file, decode_times=False)
#         self.date = self.ds['time'].values.astype('datetime64[D]')

#         self.land_mask, self.soil_type, self.topography = self._load_constant_mask()

#     def __getitem__(self, index):
#         surface_t, upper_air_t = self._get_data(index)
#         surface_t_1, upper_air_t_1 = self._get_data(index + 1)
#         if self.flag == "train":
#             return surface_t, upper_air_t, surface_t_1, upper_air_t_1
#         return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
#             self.date[index].astype(int), self.date[index + 1].astype(int)
#         ])

#     def _get_data(self, index):
#         date = self.date[index]
#         time = self.ds['time'].values[index]

#         surface_data = np.stack([self.ds[x].sel(time=time).data for x in self.surface_variables], axis=0)  # C Lat Lon
#         surface_data = torch.from_numpy(surface_data.astype(np.float32))
#         surface_data = self.surface_transform(surface_data)

#         upper_air_data = torch.stack([self.upper_air_transform[pl](
#             torch.from_numpy(np.stack([self.ds.sel(time=time, level=pl)[x].data for x in self.upper_air_variables], axis=0).astype(np.float32))
#         ) for pl in self.upper_air_pLevels], dim=1)  # C Pl Lat Lon
#         return surface_data, upper_air_data

#     def __len__(self):
#         return len(self.date) - 1

#     def _load_constant_mask(self):
#         mask_dir = "constant_mask"
#         land_mask = torch.from_numpy(np.load(join("data", mask_dir, "land_mask.npy")).astype(np.float32))
#         soil_type = torch.from_numpy(np.load(join("data", mask_dir, "soil_type.npy")).astype(np.float32))
#         topography = torch.from_numpy(np.load(join("data", mask_dir, "topography.npy")).astype(np.float32))

#         return land_mask, soil_type, topography
    
#     def get_constant_mask(self):
#         return self.land_mask, self.soil_type, self.topography

#     def get_lat_lon(self):
#         return self.ds["latitude"].data, self.ds["longitude"].data

