import xarray as xr
import os

output_dir = 'cmems_output'
for f in os.listdir(output_dir):
    if f.endswith('.nc'):
        print(f'파일: {f}')
        ds = xr.open_dataset(os.path.join(output_dir, f))
        for v in ds.data_vars:
            print(f'변수: {v}, 값 shape: {ds[v].values.shape}')
            print(ds[v].values)
        ds.close()
