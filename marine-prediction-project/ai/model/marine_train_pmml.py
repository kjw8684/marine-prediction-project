import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

# 환경 데이터 캐시(DB/실시간 환경 데이터 없음) 기능 비활성화
# def load_env_cache(csv_path="../ai/data/env_cache.csv"):
#     ...

# OBIS API에서 해당 위치/날짜의 생물 출현 데이터 수집
def get_obis_data(lat, lon, date, radius_km=10):
    url = f"https://api.obis.org/v3/occurrence?geometry=POINT({lon} {lat})&radius={radius_km}&date={date}"
    resp = requests.get(url)
    data = resp.json()
    species = set()
    for rec in data.get("results", []):
        species.add(rec.get("scientificName"))
    return list(species)

# 학습 데이터셋 구축
# 실시간 환경 데이터를 CMEMS에서 직접 받아와서 바로 사용 (DB/파일 저장 없이)
def get_cmems_data(lat, lon, date):
    """
    Copernicus Marine Toolbox(copernicusmarine)로 지정 위치/날짜/변수의 데이터를 다운로드하고 파싱
    환경 변수로 인증 정보를 받아야 하며, Python 3.9+ 필요
    """
    import copernicusmarine
    import os
    import xarray as xr
    output_dir = "cmems_output"
    os.makedirs(output_dir, exist_ok=True)
    start_datetime = date + " 00:00:00"
    end_datetime = date + " 23:59:59"
    # 실제 변수명 describe로 확인 후 사용 (2024년 이후 표준)
    # 실제 describe 결과에 있는 변수명만 사용 (표층 해류 등은 별도 데이터셋 필요)
    phy_id = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    phy_vars = ["tob", "sob", "zos"]  # 바닥 온도, 바닥 염분, 해수면 높이
    copernicusmarine.subset(
        dataset_id=phy_id,
        variables=phy_vars,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_longitude=lon,
        maximum_longitude=lon,
        minimum_latitude=lat,
        maximum_latitude=lat,
        output_directory=output_dir
    )
    # 용존산소
    bgc_id = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
    bgc_vars = ["o2"]
    copernicusmarine.subset(
        dataset_id=bgc_id,
        variables=bgc_vars,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_longitude=lon,
        maximum_longitude=lon,
        minimum_latitude=lat,
        maximum_latitude=lat,
        output_directory=output_dir
    )
    # 엽록소 (chl 변수는 describe 결과에 없음, 필요시 별도 확인)
    # chl_id = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"
    # chl_vars = ["chl"]
    # copernicusmarine.subset(
    #     dataset_id=chl_id,
    #     variables=chl_vars,
    #     start_datetime=start_datetime,
    #     end_datetime=end_datetime,
    #     minimum_longitude=lon,
    #     maximum_longitude=lon,
    #     minimum_latitude=lat,
    #     maximum_latitude=lat,
    #     output_directory=output_dir
    # )
    # NetCDF 파일 파싱 (가장 최근 파일)
    files = [f for f in os.listdir(output_dir) if f.endswith('.nc')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    result = {"bottom_temp": np.nan, "bottom_salinity": np.nan, "sea_surface_height": np.nan, "oxygen": np.nan}
    for f in files:
        try:
            ds = xr.open_dataset(os.path.join(output_dir, f))
            # 바닥 온도
            if "tob" in ds:
                result["bottom_temp"] = float(ds["tob"].isel(time=0, depth=0).values)
            # 바닥 염분
            if "sob" in ds:
                result["bottom_salinity"] = float(ds["sob"].isel(time=0, depth=0).values)
            # 해수면 높이
            if "zos" in ds:
                result["sea_surface_height"] = float(ds["zos"].isel(time=0).values)
            # 용존산소
            if "o2" in ds:
                result["oxygen"] = float(ds["o2"].isel(time=0, depth=0).values)
            ds.close()
        except Exception as e:
            print(f"NetCDF 파싱 오류: {f}, {e}")
    return result

def build_training_data(lat_range, lon_range, date_range, target_species_list):
    rows = []
    for lat in lat_range:
        for lon in lon_range:
            for date in date_range:
                env = get_cmems_data(lat, lon, date)
                observed_species = get_obis_data(lat, lon, date)
                for species in target_species_list:
                    label = 1 if species in observed_species else 0
                    rows.append({
                        "lat": lat, "lon": lon, "date": date,
                        **env, "species": species, "present": label
                    })
    return pd.DataFrame(rows)

# Random Forest 학습 및 PMML 변환
def train_and_export_pmml(df, pmml_path="marine_model.pmml"):
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"]).astype(int) // 10**9  # timestamp
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])
    X = df[["lat", "lon", "date", "temp", "current_speed", "current_dir", "oxygen", "chlorophyll", "salinity", "species"]]
    y = df["present"]
    pipeline = PMMLPipeline([
        ("classifier", RandomForestClassifier(n_estimators=100))
    ])
    pipeline.fit(X, y)
    sklearn2pmml(pipeline, pmml_path, with_repr=True)
    print(f"PMML 모델이 {pmml_path}에 저장되었습니다.")

# 주기적 재학습/예측 기능은 현재 테스트 용도로 주석 처리
# def batch_retrain():
#     while True:
#         lat_range = np.arange(33.0, 39.0, 1.0)
#         lon_range = np.arange(124.0, 132.0, 1.0)
#         date_range = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
#         target_species_list = ["Engraulis japonicus", "Todarodes pacificus"]
#         df = build_training_data(lat_range, lon_range, date_range, target_species_list)
#         train_and_export_pmml(df)
#         print("모델 재학습 및 PMML 저장 완료")
#         time.sleep(60*60*24)

if __name__ == "__main__":
    # 환경 변수로 인증 정보가 등록되어 있어야 함
    # 예시: set COPERNICUSMARINE_SERVICE_USERNAME=your_username
    #       set COPERNICUSMARINE_SERVICE_PASSWORD=your_password
    test_lat, test_lon = 36.0, 129.0
    test_date = datetime.now().strftime("%Y-%m-%d")
    import copernicusmarine

    print("[CMEMS describe로 실제 변수명 확인]")
    phy_id = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    bgc_id = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
    chl_id = "cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m"
    for name, dsid in [
        ("물리 데이터셋", phy_id),
        ("BGC(용존산소) 데이터셋", bgc_id),
        ("엽록소 데이터셋", chl_id)
    ]:
        desc = copernicusmarine.describe(dataset_id=dsid)
        # CopernicusMarineCatalogue 객체에서 변수명 안전하게 추출
        variables = getattr(desc, "variables", None)
        if variables is None and hasattr(desc, "to_dict"):
            variables = desc.to_dict().get("variables")
        print(f"{name} 변수: {variables if variables is not None else desc}")

    print("[CMEMS 환경 데이터 다운로드 및 파싱 테스트]")
    try:
        env = get_cmems_data(test_lat, test_lon, test_date)
        print("CMEMS 환경 데이터:", env)
    except Exception as e:
        print("CMEMS API 호출 실패:", e)

    print("[OBIS API 테스트]")
    try:
        species = get_obis_data(test_lat, test_lon, test_date)
        print("OBIS 생물종:", species)
    except Exception as e:
        print("OBIS API 호출 실패:", e)

    # 아래는 기존 학습 예시 (주석처리)
    # lat_range = np.arange(33.0, 39.0, 1.0)
    # lon_range = np.arange(124.0, 132.0, 1.0)
    # date_range = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    # target_species_list = ["Engraulis japonicus", "Todarodes pacificus"]
    # df = build_training_data(lat_range, lon_range, date_range, target_species_list)
    # train_and_export_pmml(df)
    # batch_retrain()  # 주기적 재학습을 원할 경우 주석 해제
