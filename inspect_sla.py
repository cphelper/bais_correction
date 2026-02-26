
import xarray as xr
import glob
import os

sla_dir = "/Users/shyam/Downloads/isro/altimeter_sla"
files = sorted(glob.glob(os.path.join(sla_dir, "*.nc")))
print(f"Found {len(files)} files.")

ds = xr.open_dataset(files[0])
print("Dims:", ds.dims)
print("Coords:", list(ds.coords))
print("Data Vars:", list(ds.data_vars))

print("\nChecking latitude/longitude names...")
if 'latitude' in ds.coords: print("Found 'latitude'")
if 'lat' in ds.coords: print("Found 'lat'")
if 'longitude' in ds.coords: print("Found 'longitude'")
if 'lon' in ds.coords: print("Found 'lon'")

print("\nModel file check:")
model_path = "/Users/shyam/Downloads/isro/sla_2/zos_Omon_MPI-ESM1-2-LR_ssp245_r1i1p1f1_gn_20150116-20251216.nc"
ds2 = xr.open_dataset(model_path)
print("Model Dims:", ds2.dims)
print("Model Coords:", list(ds2.coords))

print("Model Lat shape:", ds2['latitude'].shape)
print("Model Lon shape:", ds2['longitude'].shape)

# Check if 1D effectively
import numpy as np
lat = ds2['latitude'].values
lon = ds2['longitude'].values
print(f"Model Lat range: {lat.min()} to {lat.max()}")
print(f"Model Lon range: {lon.min()} to {lon.max()}")

