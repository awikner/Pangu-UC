import argparse
import os
from datetime import datetime
import cftime
import numpy as np


from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
import pandas as pd

# vscode Relative path
import sys
sys.path.append("../../")

from weatherlearn.models import Pangu_lite
from data_utils import DatasetFromFolder

parser = argparse.ArgumentParser(description="Train Pangu_lite Models")
parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--year_start", type=int, required=True, help="Start year for the data")
parser.add_argument("--year_end", type=int, required=True, help="End year for the data")
parser.add_argument("--surface_variables", nargs="+", required=True, help="List of surface variables to include")
parser.add_argument("--boundary_dir", type=str, default="boundary_variables", help="Directory containing boundary variable files")

if __name__ == "__main__":
    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs
    DATA_DIR = opt.data_dir
    YEAR_START = opt.year_start
    YEAR_END = opt.year_end
    SURFACE_VARIABLES = opt.surface_variables
    BOUNDARY_DIR = opt.boundary_dir

    train_set = DatasetFromFolder(DATA_DIR, YEAR_START, YEAR_END, "train", SURFACE_VARIABLES, BOUNDARY_DIR)
    val_set = DatasetFromFolder(DATA_DIR, YEAR_START, YEAR_END, "valid", SURFACE_VARIABLES, BOUNDARY_DIR)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    lat, lon = train_set.get_lat_lon()
    boundary_ds = train_set.boundary_ds

    pangu_lite = Pangu_lite()
    print("# parameters: ", sum(param.numel() for param in pangu_lite.parameters()))

    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    if torch.cuda.is_available():
        pangu_lite.cuda()
        surface_criterion.cuda()
        upper_air_criterion.cuda()

    surface_invTrans = train_set.surface_inv_transform()
    upper_air_invTrans = train_set.upper_air_inv_transform()

    optimizer = torch.optim.Adam(pangu_lite.parameters(), lr=5e-4, weight_decay=3e-6)

    results = {'loss': [], 'surface_mse': [], 'upper_air_mse': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}

        pangu_lite.train()
        for input_surface, input_upper_air, target_surface, target_upper_air in train_bar:
            batch_size = input_surface.size(0)
            if torch.cuda.is_available():
                input_surface = input_surface.cuda()
                input_upper_air = input_upper_air.cuda()
                target_surface = target_surface.cuda()
                target_upper_air = target_upper_air.cuda()

            # Select the boundary variables for this batch
            start_time = input_surface[0, 0, 0, 0].item()
            end_time = target_surface[0, 0, 0, 0].item()
            batch_boundary_ds = boundary_ds.sel(time=slice(cftime.DatetimeNoLeap(start_time, 'seconds since 1970-01-01'),
                                                            cftime.DatetimeNoLeap(end_time, 'seconds since 1970-01-01')))
            boundary_vars = [var for var in batch_boundary_ds.data_vars]
            boundary_data = torch.tensor(np.stack([batch_boundary_ds[var].values for var in boundary_vars], axis=0), dtype=torch.float32)
            if torch.cuda.is_available():
                boundary_data = boundary_data.cuda()

            output_surface, output_upper_air = pangu_lite(input_surface, boundary_data, input_upper_air)

            optimizer.zero_grad()
            surface_loss = surface_criterion(output_surface, target_surface)
            upper_air_loss = upper_air_criterion(output_upper_air, target_upper_air)
            loss = upper_air_loss + surface_loss * 0.25
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            running_results["batch_sizes"] += batch_size

            train_bar.set_description(desc="[%d/%d] Loss: %.4f" %
                                      (epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]))

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {"batch_sizes": 0, "surface_mse": 0, "upper_air_mse": 0}
            for val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, times in val_bar:
                batch_size = val_input_surface.size(0)
                if torch.cuda.is_available():
                    val_input_surface = val_input_surface.cuda()
                    val_input_upper_air = val_input_upper_air.cuda()
                    val_target_surface = val_target_surface.cuda()
                    val_target_upper_air = val_target_upper_air.cuda()

                start_time = val_input_surface[0, 0, 0, 0].item()
                end_time = val_target_surface[0, 0, 0, 0].item()
                batch_boundary_ds = boundary_ds.sel(time=slice(cftime.DatetimeNoLeap(start_time, 'seconds since 1970-01-01'),
                                                                cftime.DatetimeNoLeap(end_time, 'seconds since 1970-01-01')))
                boundary_vars = [var for var in batch_boundary_ds.data_vars]
                boundary_data = torch.tensor(np.stack([batch_boundary_ds[var].values for var in boundary_vars], axis=0), dtype=torch.float32)
                if torch.cuda.is_available():
                    boundary_data = boundary_data.cuda()

                val_output_surface, val_output_upper_air = pangu_lite(val_input_surface, boundary_data, val_input_upper_air)

                val_output_surface = val_output_surface.squeeze(0)
                val_output_upper_air = val_output_upper_air.squeeze(0)

                val_target_surface = val_target_surface.squeeze(0)
                val_target_upper_air = val_target_upper_air.squeeze(0)

                valing_results["batch_sizes"] += batch_size

                surface_mse = ((val_output_surface - val_target_surface) ** 2).data.mean().cpu().item()
                upper_air_mse = ((val_output_upper_air - val_target_upper_air) ** 2).data.mean().cpu().item()

                valing_results["surface_mse"] += surface_mse * batch_size
                valing_results["upper_air_mse"] += upper_air_mse * batch_size

                val_bar.set_description(desc="[validating] Surface MSE: %.4f Upper Air MSE: %.4f" %
                                        (valing_results["surface_mse"] / valing_results["batch_sizes"], valing_results["upper_air_mse"] / valing_results["batch_sizes"]))

        os.makedirs("epochs", exist_ok=True)
        torch.save(pangu_lite.state_dict(), "epochs/pangu_lite_epoch_%d.pth" % (epoch))

        results["loss"].append(running_results["loss"] / running_results["batch_sizes"])
        results["surface_mse"].append(valing_results["surface_mse"] / valing_results["batch_sizes"])
        results["upper_air_mse"].append(valing_results["upper_air_mse"] / valing_results["batch_sizes"])

        data_frame = pd.DataFrame(
            data=results,
            index=range(1, epoch + 1)
        )
        save_root = "train_logs"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        data_frame.to_csv(os.path.join(save_root, "logs.csv"), index_label="Epoch")

# python train.py --data_dir /path/to/data/dir --year_start 2000 --year_end 2010 --surface_variables var1 var2 var3 --boundary_dir boundary_variables