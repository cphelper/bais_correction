
import xarray as xr

def inspect_nc(file_path):
    print(f"--- Inspecting {file_path} ---")
    try:
        ds = xr.open_dataset(file_path)
        print(ds)
        print("\nVariables:")
        for var in ds.variables:
            print(f"  {var}: {ds[var].dims}, {ds[var].attrs}")
        ds.close()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    inspect_nc("cowcliphs.nc")
    inspect_nc("io-altimeter.nc")

