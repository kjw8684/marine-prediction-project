# 환경 변수 추출 함수 (NetCDF)
def extract_var(nc_path, var_name, lat, lon, date):
    # 파일 경로 및 존재 여부 강제 진단
    import os
    log_path = os.path.abspath("extract_var_debug.log")
    try:
        msg = f"[DIAG][extract_var] call | PID={os.getpid()} | nc_path={nc_path} | exists={os.path.exists(nc_path)}"
        print(msg)
        with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
            logf.write(msg + "\n")
    except Exception as e:
        pass
    import os
    import xarray as xr
    import numpy as np
    from datetime import datetime
    import os
    log_path = os.path.abspath("extract_var_debug.log")
    import traceback
    try:
        call_msg = f"[CALL] {datetime.now()} | PID={os.getpid()} | {nc_path} | {var_name} | lat={lat}, lon={lon}, date={date}"
        print(call_msg)
        with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
            logf.write(call_msg + "\n")
        if not os.path.exists(nc_path):
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[ERR] {datetime.now()} | PID={os.getpid()} | {nc_path} | 파일 없음\n")
            return float('nan')
        ds = None
        try:
            ds = xr.open_dataset(nc_path)
        except Exception as e:
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[ERR] {datetime.now()} | PID={os.getpid()} | {nc_path} | open_dataset 실패 | {e}\n")
            return float('nan')
        # NetCDF 구조 진단 로그 추가
        try:
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[NCINFO] {datetime.now()} | {nc_path} | variables: {list(ds.variables.keys())} | dims: {ds.dims} | coords: {list(ds.coords.keys())}\n")
                for vname in ds.variables:
                    v = ds[vname]
                    logf.write(f"[NCVAR] {vname} | dims: {v.dims} | shape: {v.shape}\n")
        except Exception as e:
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[NCINFO_ERR] {datetime.now()} | {nc_path} | 구조 진단 실패 | {e}\n")
        if var_name not in ds:
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[ERR] {datetime.now()} | PID={os.getpid()} | {nc_path} | {var_name} not in dataset | ds vars: {list(ds.variables.keys())}\n")
            return float('nan')
        v = ds[var_name]
        # 좌표 인덱스 찾기
        try:
            lat_vals = v['lat'].values
            lon_vals = v['lon'].values
            lat_idx = np.abs(lat_vals - lat).argmin()
            lon_idx = np.abs(lon_vals - lon).argmin()
        except Exception as e:
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[ERR] {datetime.now()} | PID={os.getpid()} | {nc_path} | {var_name} | 좌표 인덱스 실패 | {e}\n")
            return float('nan')
        # 시간 인덱스 찾기
        try:
            if 'time' in v.dims:
                time_vals = v['time'].values
                # date가 str이면 np.datetime64로 변환
                if isinstance(date, str):
                    date64 = np.datetime64(date)
                else:
                    date64 = date
                time_idx = np.abs(time_vals - date64).argmin()
                value = v.isel(lat=lat_idx, lon=lon_idx, time=time_idx).values.item()
            else:
                value = v.isel(lat=lat_idx, lon=lon_idx).values.item()
        except Exception as e:
            with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                logf.write(f"[ERR] {datetime.now()} | PID={os.getpid()} | {nc_path} | {var_name} | 인덱싱/값 추출 실패 | {e}\n")
            return float('nan')
        with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
            logf.write(f"[OK] {datetime.now()} | PID={os.getpid()} | {nc_path} | {var_name} | lat={lat}, lon={lon}, date={date} | lat_idx={lat_idx}, lon_idx={lon_idx} | value={value}\n")
        return value
    except Exception as e:
        with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
            logf.write(f"[FATAL] {datetime.now()} | PID={os.getpid()} | {nc_path} | {var_name} | lat={lat}, lon={lon}, date={date} | {e}\n{traceback.format_exc()}\n")
        return float('nan')
from datetime import datetime, timedelta
today = datetime.now().date()
date_range = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
date_range = sorted(date_range)
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
    # (예시) 하루 단위 NetCDF 파일 경로 생성 및 다운로드 (중복 방지)
    # 실제 구현에서는 phy_nc, bgc_nc 등 파일 경로와 download_with_lock 함수가 필요
    # 아래는 예시이며, 실제 변수명과 로직은 기존 정상 동작하던 코드로 복구 필요
    import os
    from datetime import datetime, timedelta
    # 파일명 및 날짜 포맷
    date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
    phy_nc = os.path.abspath(f"cmems_output/cmems_mod_glo_phy_anfc_0.083deg_P1D-m_{date_str}.nc")
    bgc_nc = os.path.abspath(f"cmems_output/cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m_{date_str}.nc")
    # 다운로드 함수 정의 (중복 방지)
    def download_with_lock(nc_path, dataset_id, variables, start_datetime, end_datetime):
        import copernicusmarine
        lock_path = nc_path + ".lock"
        if os.path.exists(nc_path):
            return
        if os.path.exists(lock_path):
            # 다른 프로세스가 다운로드 중이면 대기
            import time
            for _ in range(60):
                if not os.path.exists(lock_path):
                    break
                time.sleep(1)
            if os.path.exists(nc_path):
                return
        # lock 파일 생성
        with open(lock_path, "w") as f:
            f.write("downloading")
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                minimum_longitude=124.0,
                maximum_longitude=132.0,
                minimum_latitude=33.0,
                maximum_latitude=39.0,
                output_filename=nc_path,
                overwrite=False
            )
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
    # CMEMS product 정보 (예시)
    phy_id = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
    bgc_id = "cmems_mod_glo_bgc-bio_anfc_0.083deg_P1D-m"
    phy_vars = ["tob", "sob", "zos"]
    bgc_vars = ["o2"]
    start_datetime = date_str + "T00:00:00"
    end_datetime = date_str + "T23:59:59"
    # 파일 다운로드 (중복 방지)
    download_with_lock(phy_nc, phy_id, phy_vars, start_datetime, end_datetime)
    download_with_lock(bgc_nc, bgc_id, bgc_vars, start_datetime, end_datetime)
    # 환경 변수 추출
    result = {}
    result["bottom_temp"] = extract_var(phy_nc, "tob", lat, lon, date_str)
    result["bottom_salinity"] = extract_var(phy_nc, "sob", lat, lon, date_str)
    result["sea_surface_height"] = extract_var(phy_nc, "zos", lat, lon, date_str)
    result["oxygen"] = extract_var(bgc_nc, "o2", lat, lon, date_str)
    return result


# 환경데이터 캐싱: 이미 받은 좌표/날짜는 재요청하지 않음
import csv
import threading
import os
env_cache_lock = threading.Lock()
def get_env_cached(lat, lon, date, env_cache_path="ai/data/env_cache.csv"):
    # 절대경로 강제
    env_cache_path = os.path.abspath(env_cache_path)
    key = f"{lat:.3f}_{lon:.3f}_{date}"
    # 캐시 파일이 없으면 헤더 생성
    if not os.path.exists(env_cache_path):
        with env_cache_lock:
            if not os.path.exists(env_cache_path):
                with open(env_cache_path, "w", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["key", "bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"])
                    writer.writeheader()
    try:
        with env_cache_lock:
            with open(env_cache_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["key"] == key:
                        # 캐시된 값이 모두 nan이면 무시하고 재추출
                        vals = {k: float(row[k]) for k in ["bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]}
                        if all(pd.isna(vals[k]) for k in vals):
                            print(f"[DIAG][get_env_cached] nan 캐시 무시: {key}")
                            break
                        return vals
    except Exception as e:
        print(f"[DIAG][get_env_cached] cache read error: {e}")
    env = get_cmems_data(lat, lon, date)
    # None 또는 dict가 아니거나 키가 부족하면 nan dict 반환
    if not isinstance(env, dict) or any(k not in env for k in ["bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]):
        env = {k: float('nan') for k in ["bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]}
    with env_cache_lock:
        with open(env_cache_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"])
            writer.writerow({"key": key, **env})
    return env

# 병렬처리용 글로벌 함수로 분리
def process_point(args):
    # 병렬 프로세스 진단: 호출 여부 print/log
    import os, traceback
    log_path = os.path.abspath("extract_var_debug.log")
    try:
        msg = f"[DIAG][process_point] called | PID={os.getpid()} | args={args}"
        print(msg)
        with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
            logf.write(msg + "\n")
    except Exception as e:
        pass
    lat, lon, date, target_species_list, env_cache_path = args
    import pandas as pd
    import os
    # env_cache_path를 절대경로로 강제
    env_cache_path = os.path.abspath(env_cache_path)
    env = get_env_cached(lat, lon, date, env_cache_path)
    # env가 None이거나 dict가 아니거나 키가 부족하면 nan dict로 대체
    if not isinstance(env, dict) or any(k not in env for k in ["bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]):
        print(f"[DIAG][process_point] env is invalid for lat={lat}, lon={lon}, date={date}: {env}")
        env = {k: float('nan') for k in ["bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]}
    observed_species = get_obis_data(lat, lon, date)
    rows = []
    for species in target_species_list:
        label = 1 if species in observed_species else 0
        row = {"lat": lat, "lon": lon, "date": date, **env, "species": species, "present": label}
        # nan 여부 진단
        nan_count = sum([pd.isna(row[k]) for k in ["bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]])
        if nan_count > 0:
            print(f"[DIAG][process_point] nan in row: {row}")
        rows.append(row)
    if not rows:
        print(f"[DEBUG] No row generated for lat={lat}, lon={lon}, date={date}, env={env}, observed_species={observed_species}")
    return rows

def build_training_data(lat_range, lon_range, date_range, target_species_list, csv_path="ai/data/training_data_korea_full.csv", env_cache_path="ai/data/env_cache.csv", max_workers=4):
    import concurrent.futures
    import pandas as pd
    import os
    fieldnames = ["lat", "lon", "date", "bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen", "species", "present"]
    # csv_path, env_cache_path를 절대경로로 강제
    csv_path = os.path.abspath(csv_path)
    env_cache_path = os.path.abspath(env_cache_path)
    # 파일 헤더 보장
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 모든 (lat, lon, date) 조합 생성
    points = [(lat, lon, date, target_species_list, env_cache_path) for lat in lat_range for lon in lon_range for date in date_range]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_point, pt) for pt in points]
        for fut in concurrent.futures.as_completed(futures):
            try:
                rows = fut.result()
                if rows:
                    results.extend(rows)
            except Exception as e:
                import traceback, os
                log_path = os.path.abspath("extract_var_debug.log")
                print(f"[ERROR] 병렬 처리 중 오류: {e}")
                try:
                    with open(log_path, "a", encoding="utf-8", buffering=1) as logf:
                        logf.write(f"[ERROR][build_training_data] {e}\n{traceback.format_exc()}\n")
                except Exception as log_e:
                    pass
    # 메인 프로세스에서만 파일 기록
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in results:
            writer.writerow(row)
    print(f"[INFO] 학습 데이터가 {csv_path}에 저장되었습니다. row 수: {len(results)}")

# Random Forest 학습 및 PMML 변환
def train_and_export_pmml(df, pmml_path="marine_model.pmml"):
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"]).astype('int64') // 10**9  # timestamp
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])
    X = df[["lat", "lon", "date", "bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen", "species"]]
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

    try:
        print("[한국 주변 해역 전체 학습 데이터 생성 및 PMML 학습 - 0.5도 격자, 최근 7일, 병렬처리/캐싱]")
        lat_range = np.arange(33.0, 39.0, 0.5)
        lon_range = np.arange(124.0, 132.0, 0.5)
        target_species_list = ["Engraulis japonicus", "Todarodes pacificus"]
        # date_range는 파일 최상단에서 최근 7일로만 정의됨
        # --- 전체 학습 (최근 7일) ---
        build_training_data(lat_range, lon_range, date_range, target_species_list, csv_path="ai/data/training_data_korea_full.csv", env_cache_path="ai/data/env_cache.csv", max_workers=4)
        import pandas as pd
        df = pd.read_csv("ai/data/training_data_korea_full.csv")
        print(f"[INFO] 학습 row: {len(df)}건")
        train_and_export_pmml(df, pmml_path="marine_model_korea.pmml")
        print("[완료] PMML 모델 학습 및 저장 완료: marine_model_korea.pmml")
    except KeyboardInterrupt:
        print("[중단] 사용자가 키보드 인터럽트(Ctrl+C)로 작업을 중단했습니다.")
        import sys
        sys.exit(0)

    # 환경데이터만 저장하고 싶을 때는 아래 주석 해제
    # print("[한국 주변 해역 전체 격자 환경데이터만 저장]")
    # rows = []
    # for lat in lat_range:
    #     for lon in lon_range:
    #         for date in date_range:
    #             env = get_cmems_data(lat, lon, date)
    #             if not all(np.isnan(v) for v in env.values()):
    #                 rows.append({"lat": lat, "lon": lon, "date": date, **env})
    # df_env = pd.DataFrame(rows)
    # df_env.to_csv("training_data_korea.csv", index=False)
    # print(f"[완료] 유효 환경데이터 {len(df_env)}건을 training_data_korea.csv로 저장")
