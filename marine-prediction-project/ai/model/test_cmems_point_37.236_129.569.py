import copernicusmarine
import xarray as xr
import os

output_dir = "cmems_output"
os.makedirs(output_dir, exist_ok=True)

lat = 37.236190413746804
lon = 129.5697280457114
date = "2025-08-28"
phy_id = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
phy_vars = ["tob", "sob", "zos"]
bgc_id = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
bgc_vars = ["o2"]

print(f"테스트 좌표: lat={lat}, lon={lon}")
# 물리 데이터 다운로드
copernicusmarine.subset(
    dataset_id=phy_id,
    variables=phy_vars,
    start_datetime=date + " 00:00:00",
    end_datetime=date + " 23:59:59",
    minimum_longitude=lon,
    maximum_longitude=lon,
    minimum_latitude=lat,
    maximum_latitude=lat,
    output_directory=output_dir
)
# BGC 데이터 다운로드
copernicusmarine.subset(
    dataset_id=bgc_id,
    variables=bgc_vars,
    start_datetime=date + " 00:00:00",
    end_datetime=date + " 23:59:59",
    minimum_longitude=lon,
    maximum_longitude=lon,
    minimum_latitude=lat,
    maximum_latitude=lat,
    output_directory=output_dir
)

# 가장 최근 파일 확인 및 값 출력
files = [f for f in os.listdir(output_dir) if f.endswith('.nc')]
files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
if files:
    for f in files[:4]:  # 최신 4개만 확인
        print(f'파일: {f}')
        ds = xr.open_dataset(os.path.join(output_dir, f))
        for v in ds.data_vars:
            arr = ds[v].values
            print(f'  변수: {v}, 값 shape: {arr.shape}')
            print(arr)
        ds.close()
