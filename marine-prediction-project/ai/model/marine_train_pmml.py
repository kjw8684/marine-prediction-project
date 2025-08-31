import os
import csv
import threading
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import copernicusmarine

# 환경 변수 캐시 락
env_cache_lock = threading.Lock()

# 주요 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
CMEMS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "cmems_output"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CMEMS_DIR, exist_ok=True)

ENV_CACHE_PATH = os.path.join(DATA_DIR, "env_cache.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "training_data_korea_full.csv")
PMML_PATH = os.path.join(DATA_DIR, "marine_ai_model.pmml")
LOG_PATH = os.path.join(BASE_DIR, "extract_var_debug.log")

# 타겟 종
TARGET_SPECIES = ["Engraulis japonicus", "Todarodes pacificus"]

# 격자/영역/날짜
LATS = np.arange(33.0, 39.1, 0.5)
LONS = np.arange(124.0, 132.1, 0.5)
D_UTC = datetime.now().astimezone().tzinfo
TODAY = datetime.now(D_UTC).date()
DAYS = [TODAY - timedelta(days=i) for i in range(7)]

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8", buffering=1) as f:
            f.write(msg + "\n")
    except Exception:
        pass


def get_best_dataset_and_variables():
    """
    실제 사용할 CMEMS 데이터셋과 변수명을 반환
    """
    dataset_config = {
        "physics": {
            "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "variables": {
                "temperature": "tob",  # sea_water_potential_temperature_at_sea_floor
                "salinity": "sob",     # sea_water_salinity_at_sea_floor
                "sea_surface_height": "zos"  # sea_surface_height_above_geoid
            }
        },
        "biogeochemistry": {
            "dataset_id": "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
            "variables": {
                "primary_production": "nppv",  # net_primary_production_of_biomass_expressed_as_carbon_per_unit_volume_in_sea_water
                "oxygen": "o2"  # mole_concentration_of_dissolved_molecular_oxygen_in_sea_water
            }
        }
    }

    return dataset_config

def extract_var(nc_path, var_name, lat, lon, date):
    """
    개선된 NetCDF 변수 추출 함수
    """
    try:
        log(f"[extract_var] 시작: {nc_path} {var_name} {lat} {lon} {date}")

        if not os.path.exists(nc_path):
            log(f"[extract_var] 파일 없음: {nc_path}")
            return float('nan')

        with xr.open_dataset(nc_path) as ds:
            log(f"[extract_var] 파일 구조: vars={list(ds.variables.keys())}")

            if var_name not in ds:
                # 유사한 변수명 찾기
                similar_vars = [v for v in ds.variables.keys() if var_name.lower() in v.lower() or v.lower() in var_name.lower()]
                log(f"[extract_var] 변수 없음: {var_name}, 유사한 변수들: {similar_vars}")

                if similar_vars:
                    var_name = similar_vars[0]  # 첫 번째 유사한 변수 사용
                    log(f"[extract_var] 대체 변수 사용: {var_name}")
                else:
                    return float('nan')

            var_data = ds[var_name]
            log(f"[extract_var] 변수 정보: {var_name} dims={var_data.dims}, shape={var_data.shape}")

            # 좌표 차원명 자동 감지
            lat_dim = None
            lon_dim = None
            time_dim = None
            depth_dim = None

            for dim in var_data.dims:
                dim_lower = dim.lower()
                if dim_lower in ['lat', 'latitude', 'y', 'nav_lat']:
                    lat_dim = dim
                elif dim_lower in ['lon', 'longitude', 'x', 'nav_lon']:
                    lon_dim = dim
                elif dim_lower in ['time', 't', 'time_counter']:
                    time_dim = dim
                elif dim_lower in ['depth', 'lev', 'level', 'z', 'deptht']:
                    depth_dim = dim

            log(f"[extract_var] 감지된 차원들: lat={lat_dim}, lon={lon_dim}, time={time_dim}, depth={depth_dim}")

            if not lat_dim or not lon_dim:
                log(f"[extract_var] 필수 좌표 차원 없음: lat_dim={lat_dim}, lon_dim={lon_dim}")
                return float('nan')

            # 선택할 좌표 준비
            select_kwargs = {
                lat_dim: lat,
                lon_dim: lon
            }

            # 시간 차원 처리
            if time_dim and time_dim in var_data.dims:
                if isinstance(date, str):
                    date_obj = pd.to_datetime(date)
                else:
                    date_obj = pd.to_datetime(str(date))
                select_kwargs[time_dim] = date_obj


            # 깊이 차원 처리 (바닥층 또는 nan이 아닌 가장 깊은 값)
            if depth_dim and depth_dim in var_data.dims:
                depth_values = ds.coords[depth_dim].values
                if len(depth_values) > 0:
                    # 가장 깊은 층부터 nan이 아닌 값을 찾음
                    for depth in reversed(depth_values):
                        select_kwargs[depth_dim] = depth
                        try:
                            selected_data = var_data.sel(method='nearest', **select_kwargs)
                            value = float(selected_data.values)
                            if np.isfinite(value):
                                log(f"[extract_var] 바닥층(실제 유효) 선택: {depth}m, 값: {value}")
                                return value
                        except Exception as e:
                            continue
                    log(f"[extract_var] 모든 깊이에서 유효값 없음 (모두 nan)")
                    return float('nan')
            else:
                # 깊이 차원이 없으면 기존 방식
                try:
                    selected_data = var_data.sel(method='nearest', **select_kwargs)
                    value = float(selected_data.values)
                    if np.isfinite(value):
                        log(f"[extract_var] 성공: {var_name} at {lat},{lon},{date} = {value}")
                        return value
                    else:
                        log(f"[extract_var] 무효한 값: {value}")
                        return float('nan')
                except Exception as e:
                    log(f"[extract_var] 선택 오류: {e}")
                    return float('nan')

    except Exception as e:
        log(f"[extract_var] 치명적 오류: {nc_path} {var_name} - {e}")
        return float('nan')

def download_with_lock(nc_path, dataset_id, variables, start_datetime, end_datetime):
    """
    개선된 CMEMS 다운로드 함수
    """
    lock_path = nc_path + ".lock"

    # 이미 파일이 존재하면 스킵
    if os.path.exists(nc_path):
        log(f"[download_with_lock] 파일 이미 존재: {nc_path}")
        return True

    # 다른 프로세스가 다운로드 중이면 대기
    if os.path.exists(lock_path):
        import time
        log(f"[download_with_lock] 다른 프로세스 다운로드 중, 대기: {nc_path}")
        for _ in range(120):
            if not os.path.exists(lock_path):
                break
            time.sleep(1)
        if os.path.exists(nc_path):
            return True

    # 락 파일 생성
    try:
        with open(lock_path, "w") as f:
            f.write(f"downloading by PID {os.getpid()}")

        log(f"[download_with_lock] 다운로드 시작: {dataset_id} variables={variables}")
        log(f"[download_with_lock] 저장 경로: {nc_path}")

        # 디렉토리 생성
        os.makedirs(os.path.dirname(nc_path), exist_ok=True)

        # CMEMS 데이터 다운로드
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

        # 다운로드 성공 여부 확인
        if os.path.exists(nc_path):
            log(f"[download_with_lock] 다운로드 완료: {nc_path}")
            return True
        else:
            log(f"[download_with_lock] 다운로드 실패: 파일이 생성되지 않음")
            return False

    except Exception as e:
        log(f"[download_with_lock] 다운로드 실패: {nc_path} - {e}")
        # 실패한 경우 파일 삭제
        if os.path.exists(nc_path):
            os.remove(nc_path)
        return False
    finally:
        # 락 파일 삭제
        if os.path.exists(lock_path):
            os.remove(lock_path)

def get_cmems_data(lat, lon, date):
    """
    올바른 변수명을 사용한 CMEMS 데이터 추출
    """
    date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
    config = get_best_dataset_and_variables()
    phy_config = config["physics"]
    bgc_config = config["biogeochemistry"]
    phy_nc = os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}.nc")
    bgc_nc = os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}.nc")
    start_datetime = f"{date_str}T00:00:00"
    end_datetime = f"{date_str}T23:59:59"

    # 하루에 한 번만 파일 다운로드 (존재하지 않을 때만)
    if not os.path.exists(phy_nc):
        download_with_lock(
            phy_nc,
            phy_config["dataset_id"],
            list(phy_config["variables"].values()),
            start_datetime,
            end_datetime
        )
    if not os.path.exists(bgc_nc):
        download_with_lock(
            bgc_nc,
            bgc_config["dataset_id"],
            list(bgc_config["variables"].values()),
            start_datetime,
            end_datetime
        )

    result = {}
    if os.path.exists(phy_nc):
        result["bottom_temp"] = extract_var(phy_nc, phy_config["variables"]["temperature"], lat, lon, date_str)
        result["bottom_salinity"] = extract_var(phy_nc, phy_config["variables"]["salinity"], lat, lon, date_str)
        result["sea_surface_height"] = extract_var(phy_nc, phy_config["variables"]["sea_surface_height"], lat, lon, date_str)
    else:
        result["bottom_temp"] = float('nan')
        result["bottom_salinity"] = float('nan')
        result["sea_surface_height"] = float('nan')

    if os.path.exists(bgc_nc):
        result["primary_production"] = extract_var(bgc_nc, bgc_config["variables"]["primary_production"], lat, lon, date_str)
        result["oxygen"] = extract_var(bgc_nc, bgc_config["variables"]["oxygen"], lat, lon, date_str)
    else:
        result["primary_production"] = float('nan')
        result["oxygen"] = float('nan')

    log(f"[get_cmems_data] 결과: {lat},{lon},{date} -> {result}")
    return result

def get_obis_data(lat, lon, date):
    return []

def process_point(args):
    lat, lon, date, target_species_list = args
    env = get_cmems_data(lat, lon, date)
    observed_species = get_obis_data(lat, lon, date)

    rows = []
    for species in target_species_list:
        label = 1 if species in observed_species else 0
        row = {"lat": lat, "lon": lon, "date": date, **env, "species": species, "present": label}

        rows.append(row)

    return rows

def process_one_day(date):
    all_rows = []
    for lat in LATS:
        for lon in LONS:
            task = (float(lat), float(lon), str(date), TARGET_SPECIES)
            rows = process_point(task)
            all_rows.extend(rows)
    day_df = pd.DataFrame(all_rows)
    day_df = day_df.dropna()

    # 학습 데이터 누적 저장 (append)
    if not day_df.empty:
        if not os.path.exists(TRAIN_CSV_PATH):
            day_df.to_csv(TRAIN_CSV_PATH, index=False)
        else:
            day_df.to_csv(TRAIN_CSV_PATH, mode='a', header=False, index=False)

    # 해당 날짜 .nc 파일 삭제
    date_str = str(date)[:10]
    phy_nc = os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}.nc")
    bgc_nc = os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}.nc")
    for nc_path in [phy_nc, bgc_nc]:
        if os.path.exists(nc_path):
            try:
                os.remove(nc_path)
            except Exception as e:
                print(f"{nc_path} 삭제 실패: {e}")

def build_training_data():
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(process_one_day, DAYS))
    print("모든 날짜 데이터 처리 및 .nc 파일 정리 완료")

def train_and_export_pmml():
    df = pd.read_csv(TRAIN_CSV_PATH)
    log(f"전체 데이터: {len(df)}행")

    df_clean = df.dropna()
    log(f"NaN 제거 후: {len(df_clean)}행")

    if len(df_clean) == 0:
        log("[train_and_export_pmml] 학습할 유효 데이터가 없습니다.")
        return

    X = df_clean[["lat", "lon", "bottom_temp", "bottom_salinity", "sea_surface_height", "oxygen"]]
    y = df_clean["present"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = PMMLPipeline([("classifier", clf)])
    pipeline.fit(X, y)

    sklearn2pmml(pipeline, PMML_PATH, with_repr=True)
    log(f"[train_and_export_pmml] PMML 저장 완료: {PMML_PATH}")

if __name__ == "__main__":
    try:
        # 불필요한 테스트 및 정의되지 않은 함수 호출부 완전 삭제
        build_training_data()
        train_and_export_pmml()
        log("[main] 완료.")

    except KeyboardInterrupt:
        log("[main] 사용자 중단.")
    except Exception as e:
        log(f"[main] 오류: {e}")
        import traceback
        log(f"[main] 스택트레이스: {traceback.format_exc()}")