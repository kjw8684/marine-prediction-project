import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import copernicusmarine
import threading
import time
import glob

# 파일 다운로드 동기화를 위한 글로벌 락
download_locks = {}
lock_manager_lock = threading.Lock()

# 주요 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
CMEMS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "cmems_output"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CMEMS_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(DATA_DIR, "training_data_korea_full.csv")
PMML_PATH = os.path.join(DATA_DIR, "marine_ai_model.pmml")
LOG_PATH = os.path.join(BASE_DIR, "extract_var_debug.log")

# 타겟 종
TARGET_SPECIES = ["Engraulis japonicus", "Todarodes pacificus"]

# 실제 데이터가 있는 0.5도 격자 셀들 (중심점 기준)
GRID_CELLS = [
    (33.0, 124.0), (33.0, 124.5), (33.0, 125.0), (33.0, 125.5), (33.0, 126.0),
    (33.0, 126.5), (33.0, 127.0), (33.0, 127.5), (33.0, 128.0), (33.0, 128.5),
    (33.0, 129.5), (33.0, 132.0), (33.5, 124.0), (33.5, 124.5), (33.5, 125.0),
    (33.5, 125.5), (33.5, 126.0), (33.5, 127.0), (33.5, 127.5), (33.5, 128.0),
    (33.5, 128.5), (33.5, 129.0), (33.5, 129.5), (33.5, 130.0), (33.5, 132.0),
    (34.0, 124.0), (34.0, 124.5), (34.0, 125.0), (34.0, 125.5), (34.0, 126.0),
    (34.0, 126.5), (34.0, 127.0), (34.0, 127.5), (34.0, 128.0), (34.0, 128.5),
    (34.0, 129.0), (34.0, 129.5), (34.0, 130.0), (34.0, 130.5), (34.0, 131.5),
    (34.5, 124.0), (34.5, 124.5), (34.5, 125.0), (34.5, 125.5), (34.5, 126.0),
    (34.5, 127.5), (34.5, 128.0), (34.5, 128.5), (34.5, 129.0), (34.5, 130.0),
    (34.5, 130.5), (34.5, 131.0), (35.0, 124.0), (35.0, 124.5), (35.0, 125.0),
    (35.0, 125.5), (35.0, 126.0), (35.0, 129.0), (35.0, 129.5), (35.0, 130.0),
    (35.0, 130.5), (35.0, 131.0), (35.0, 131.5), (35.0, 132.0), (35.5, 124.0),
    (35.5, 124.5), (35.5, 125.0), (35.5, 125.5), (35.5, 126.0), (35.5, 126.5),
    (35.5, 130.0), (35.5, 130.5), (35.5, 131.0), (35.5, 131.5), (35.5, 132.0),
    (36.0, 124.0), (36.0, 124.5), (36.0, 125.0), (36.0, 125.5), (36.0, 126.0),
    (36.0, 126.5), (36.0, 130.0), (36.0, 130.5), (36.0, 131.0), (36.0, 131.5),
    (36.0, 132.0), (36.5, 124.0), (36.5, 124.5), (36.5, 125.0), (36.5, 125.5),
    (36.5, 126.0), (36.5, 129.5), (36.5, 130.0), (36.5, 130.5), (36.5, 131.0),
    (36.5, 131.5), (36.5, 132.0), (37.0, 124.0), (37.0, 124.5), (37.0, 125.0),
    (37.0, 125.5), (37.0, 126.0), (37.0, 129.5), (37.0, 130.0), (37.0, 130.5),
    (37.0, 131.0), (37.0, 131.5), (37.0, 132.0), (37.5, 124.0), (37.5, 124.5),
    (37.5, 125.0), (37.5, 125.5), (37.5, 126.0), (37.5, 129.5), (37.5, 130.0),
    (37.5, 130.5), (37.5, 131.0), (37.5, 131.5), (37.5, 132.0), (38.0, 124.0),
    (38.0, 124.5), (38.0, 125.0), (38.0, 129.0), (38.0, 129.5), (38.0, 130.0),
    (38.0, 130.5), (38.0, 131.0), (38.0, 131.5), (38.0, 132.0), (38.5, 124.0),
    (38.5, 124.5), (38.5, 128.5), (38.5, 129.0), (38.5, 129.5), (38.5, 130.0),
    (38.5, 130.5), (38.5, 131.0), (38.5, 131.5), (38.5, 132.0), (39.0, 124.0),
    (39.0, 124.5), (39.0, 125.0), (39.0, 128.0), (39.0, 128.5), (39.0, 129.0),
    (39.0, 129.5), (39.0, 130.0), (39.0, 130.5), (39.0, 131.0), (39.0, 131.5),
    (39.0, 132.0)
]

# 격자 셀 크기 (0.5도 x 0.5도)
GRID_RESOLUTION = 0.5
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


def get_dataset_config():
    """CMEMS 데이터셋과 변수명 설정"""
    return {
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

def extract_var_from_grid_area(nc_path, var_name, lat_min, lat_max, lon_min, lon_max, date):
    """
    0.5도 격자 셀 영역에서 환경변수의 평균값 추출 - 유효한 값들만 사용
    """
    try:
        log(f"[extract_grid_area] 시작: {var_name} 영역({lat_min}-{lat_max}, {lon_min}-{lon_max}) 날짜: {date}")

        if not os.path.exists(nc_path):
            log(f"[extract_grid_area] 파일 없음: {nc_path}")
            return float('nan')

        with xr.open_dataset(nc_path) as ds:
            if var_name not in ds.variables:
                log(f"[extract_grid_area] 변수 없음: {var_name}")
                return float('nan')

            var_data = ds[var_name]

            # 차원 이름 식별
            lat_dim = next((dim for dim in var_data.dims if 'lat' in dim.lower()), None)
            lon_dim = next((dim for dim in var_data.dims if 'lon' in dim.lower()), None)
            time_dim = next((dim for dim in var_data.dims if 'time' in dim.lower()), None)
            depth_dim = next((dim for dim in var_data.dims if 'depth' in dim.lower()), None)

            if not lat_dim or not lon_dim:
                log(f"[extract_grid_area] 위도/경도 차원 찾을 수 없음")
                return float('nan')

            # BGC 데이터는 더 넓은 영역에서 검색 (원래 영역에서 확장)
            search_lat_min, search_lat_max = lat_min, lat_max
            search_lon_min, search_lon_max = lon_min, lon_max

            # BGC 데이터인 경우 영역 확장 (0.25도씩 확장)
            if 'bgc' in nc_path.lower() or var_name in ['nppv', 'o2']:
                margin = 0.5  # 0.5도씩 확장
                search_lat_min = max(lat_min - margin, -90)
                search_lat_max = min(lat_max + margin, 90)
                search_lon_min = max(lon_min - margin, -180)
                search_lon_max = min(lon_max + margin, 180)
                log(f"[extract_grid_area] BGC 데이터 확장 검색: {var_name} 영역({search_lat_min}-{search_lat_max}, {search_lon_min}-{search_lon_max})")

            # 1단계: 공간 영역 선택 (slice만 사용, method 없이)
            spatial_select_kwargs = {
                lat_dim: slice(search_lat_min, search_lat_max),
                lon_dim: slice(search_lon_min, search_lon_max)
            }

            # 공간 영역 선택
            spatial_data = var_data.sel(**spatial_select_kwargs)

            # 2단계: 시간/깊이 차원 선택 (method='nearest' 사용)
            temporal_select_kwargs = {}

            # 시간 차원 처리
            if time_dim and time_dim in spatial_data.dims:
                date_obj = pd.to_datetime(date)
                temporal_select_kwargs[time_dim] = date_obj

            # 깊이 차원 처리 (BGC는 표층, 물리는 심층)
            if depth_dim and depth_dim in spatial_data.dims:
                depth_values = ds.coords[depth_dim].values
                if len(depth_values) > 0:
                    if var_name in ['nppv', 'o2']:  # BGC 변수는 표층 사용
                        target_depth = depth_values[0]  # 가장 얕은 층
                        log(f"[extract_grid_area] BGC 표층 선택: {target_depth}m")
                    else:  # 물리 변수는 심층 사용
                        target_depth = depth_values[-1]  # 가장 깊은 층
                    temporal_select_kwargs[depth_dim] = target_depth

            # 시간/깊이 선택 (method='nearest' 사용)
            if temporal_select_kwargs:
                selected_data = spatial_data.sel(method='nearest', **temporal_select_kwargs)
            else:
                selected_data = spatial_data

            if hasattr(selected_data, 'values'):
                values = selected_data.values
                if np.isscalar(values):
                    mean_value = float(values)
                else:
                    # 배열인 경우 NaN 제외하고 평균 계산
                    finite_values = values[np.isfinite(values)]
                    if len(finite_values) > 0:
                        mean_value = np.mean(finite_values)
                        log(f"[extract_grid_area] 유효한 값 {len(finite_values)}개에서 평균 계산: {var_name} = {mean_value}")
                    else:
                        mean_value = float('nan')

                if np.isfinite(mean_value):
                    log(f"[extract_grid_area] 성공: {var_name} 격자 평균 = {mean_value}")
                    return mean_value
                else:
                    # 확장된 영역에서도 값이 없는 경우
                    log(f"[extract_grid_area] 확장 영역에서도 유효한 값 없음: {var_name} 영역({search_lat_min}-{search_lat_max}, {search_lon_min}-{search_lon_max})")
                    return float('nan')
            else:
                log(f"[extract_grid_area] 데이터 값 없음")
                return float('nan')

    except Exception as e:
        log(f"[extract_grid_area] 치명적 오류: {nc_path} {var_name} - {e}")
        return float('nan')


def get_file_lock(file_path):
    """파일별 락 객체 반환"""
    with lock_manager_lock:
        if file_path not in download_locks:
            download_locks[file_path] = threading.Lock()
        return download_locks[file_path]


def cleanup_duplicate_nc_files(date_str):
    """중복 다운로드된 .nc 파일들 정리 (번호가 붙은 파일들 삭제)"""
    try:
        patterns = [
            os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}_*.nc"),
            os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}_*.nc"),
        ]
        
        cleaned_files = []
        for pattern in patterns:
            duplicate_files = glob.glob(pattern)
            for dup_file in duplicate_files:
                try:
                    os.remove(dup_file)
                    cleaned_files.append(os.path.basename(dup_file))
                    log(f"[cleanup_duplicates] 중복 파일 삭제: {os.path.basename(dup_file)}")
                except Exception as e:
                    log(f"[cleanup_duplicates] 삭제 실패: {os.path.basename(dup_file)} - {e}")
        
        if cleaned_files:
            log(f"[cleanup_duplicates] {date_str}: {len(cleaned_files)}개 중복 파일 정리 완료")
            
    except Exception as e:
        log(f"[cleanup_duplicates] 오류: {date_str} - {e}")


def cleanup_daily_nc_files(date_str):
    """하루치 데이터 처리 완료 후 .nc 파일들을 삭제하여 메모리 절약"""
    try:
        phy_nc = os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}.nc")
        bgc_nc = os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}.nc")
        
        deleted_files = []
        total_size_saved = 0
        
        for nc_file in [phy_nc, bgc_nc]:
            if os.path.exists(nc_file):
                try:
                    file_size = os.path.getsize(nc_file)
                    os.remove(nc_file)
                    deleted_files.append(os.path.basename(nc_file))
                    total_size_saved += file_size
                    log(f"[cleanup] 삭제 완료: {os.path.basename(nc_file)} ({file_size/1024/1024:.1f}MB)")
                except Exception as e:
                    log(f"[cleanup] 삭제 실패: {os.path.basename(nc_file)} - {e}")
        
        if deleted_files:
            log(f"[cleanup] {date_str} 정리 완료: {len(deleted_files)}개 파일, {total_size_saved/1024/1024:.1f}MB 절약")
        else:
            log(f"[cleanup] {date_str}: 삭제할 파일 없음")
            
    except Exception as e:
        log(f"[cleanup] 오류: {date_str} - {e}")


def download_with_lock(nc_path, dataset_id, variables, start_datetime, end_datetime):
    """CMEMS 데이터 다운로드 - 파일 잠금으로 중복 다운로드 방지"""
    
    # 파일별 락 획득
    file_lock = get_file_lock(nc_path)
    
    with file_lock:
        # 락 내에서 다시 파일 존재 여부 확인
        if os.path.exists(nc_path):
            log(f"[download_with_lock] 파일 이미 존재: {nc_path}")
            return True
            
        # 중복 파일들이 있다면 먼저 정리
        date_str = os.path.basename(nc_path).replace("cmems_phy_", "").replace("cmems_bgc_", "").replace(".nc", "")
        cleanup_duplicate_nc_files(date_str)

        try:
            log(f"[download_with_lock] 다운로드 시작: {dataset_id}")
            os.makedirs(os.path.dirname(nc_path), exist_ok=True)

            # 임시 파일명으로 다운로드
            temp_nc_path = nc_path + ".temp"
            
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                minimum_longitude=124.0,
                maximum_longitude=132.0,
                minimum_latitude=33.0,
                maximum_latitude=39.0,
                output_filename=temp_nc_path,
                overwrite=True  # 임시 파일은 덮어쓰기 허용
            )

            # 다운로드 완료 후 원래 이름으로 이동
            if os.path.exists(temp_nc_path):
                os.rename(temp_nc_path, nc_path)
                log(f"[download_with_lock] 다운로드 완료: {nc_path}")
                return True
            else:
                log(f"[download_with_lock] 다운로드 실패: 임시 파일이 생성되지 않음")
                return False

        except Exception as e:
            log(f"[download_with_lock] 다운로드 실패: {nc_path} - {e}")
            
            # 임시 파일 정리
            temp_nc_path = nc_path + ".temp"
            try:
                if os.path.exists(temp_nc_path):
                    os.remove(temp_nc_path)
            except Exception:
                pass
                
            return False

def get_cmems_data_for_grid_cell(center_lat, center_lon, date):
    """0.5도 격자 셀에서 CMEMS 데이터 추출 (격자 평균값)"""
    date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
    config = get_dataset_config()
    phy_config = config["physics"]
    bgc_config = config["biogeochemistry"]
    phy_nc = os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}.nc")
    bgc_nc = os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}.nc")
    start_datetime = f"{date_str}T00:00:00"
    end_datetime = f"{date_str}T23:59:59"

    # 하루에 한 번만 파일 다운로드
    if not os.path.exists(phy_nc):
        download_with_lock(phy_nc, phy_config["dataset_id"], list(phy_config["variables"].values()), start_datetime, end_datetime)
    if not os.path.exists(bgc_nc):
        download_with_lock(bgc_nc, bgc_config["dataset_id"], list(bgc_config["variables"].values()), start_datetime, end_datetime)

    # 격자 셀 경계 계산 (중심점 ±0.25도)
    lat_min = center_lat - GRID_RESOLUTION/2
    lat_max = center_lat + GRID_RESOLUTION/2
    lon_min = center_lon - GRID_RESOLUTION/2
    lon_max = center_lon + GRID_RESOLUTION/2

    log(f"[get_cmems_grid_cell] 격자 셀: center({center_lat},{center_lon}), bounds({lat_min}-{lat_max}, {lon_min}-{lon_max})")

    result = {}

    # 물리 변수 추출 (격자 셀 평균)
    if os.path.exists(phy_nc):
        result["bottom_temp"] = extract_var_from_grid_area(phy_nc, phy_config["variables"]["temperature"], lat_min, lat_max, lon_min, lon_max, date_str)
        result["bottom_salinity"] = extract_var_from_grid_area(phy_nc, phy_config["variables"]["salinity"], lat_min, lat_max, lon_min, lon_max, date_str)
        result["sea_surface_height"] = extract_var_from_grid_area(phy_nc, phy_config["variables"]["sea_surface_height"], lat_min, lat_max, lon_min, lon_max, date_str)
    else:
        result["bottom_temp"] = float('nan')
        result["bottom_salinity"] = float('nan')
        result["sea_surface_height"] = float('nan')

    # 생지화학 변수 추출 (격자 셀 평균)
    if os.path.exists(bgc_nc):
        result["primary_production"] = extract_var_from_grid_area(bgc_nc, bgc_config["variables"]["primary_production"], lat_min, lat_max, lon_min, lon_max, date_str)
        result["oxygen"] = extract_var_from_grid_area(bgc_nc, bgc_config["variables"]["oxygen"], lat_min, lat_max, lon_min, lon_max, date_str)
    else:
        result["primary_production"] = float('nan')
        result["oxygen"] = float('nan')

    log(f"[get_cmems_grid_cell] 격자 셀 결과: center({center_lat},{center_lon}),{date} -> {result}")
    return result

def get_gbif_data_for_grid_cell(center_lat, center_lon, date):
    """격자 셀 내의 GBIF 관측 데이터 조회 (현재는 빈 리스트 반환)"""
    return []

def process_grid_cell(args):
    """하나의 격자 셀을 처리하여 훈련 데이터 생성"""
    center_lat, center_lon, date, target_species_list = args

    # 격자 셀에서 환경 데이터 추출
    env = get_cmems_data_for_grid_cell(center_lat, center_lon, date)

    # 격자 셀에서 생물 관측 데이터 추출
    observed_species = get_gbif_data_for_grid_cell(center_lat, center_lon, date)

    rows = []
    for species in target_species_list:
        label = 1 if species in observed_species else 0
        row = {
            "lat": center_lat,
            "lon": center_lon,
            "date": date,
            **env,
            "species": species,
            "present": label
        }
        rows.append(row)

    return rows

def process_one_day(date):
    """하드코딩된 격자 셀들에 대해 하루치 데이터 처리"""
    all_rows = []

    # 하드코딩된 격자 셀들만 처리
    for center_lat, center_lon in GRID_CELLS:
        task = (float(center_lat), float(center_lon), str(date), TARGET_SPECIES)
        rows = process_grid_cell(task)
        all_rows.extend(rows)

    day_df = pd.DataFrame(all_rows)

    # 물리 해양 데이터 유효성 체크
    physical_features = ['lat', 'lon', 'bottom_temp', 'bottom_salinity', 'sea_surface_height', 'species', 'present']
    day_df_clean = day_df.dropna(subset=physical_features)

    # 학습 데이터 누적 저장
    if not day_df_clean.empty:
        if not os.path.exists(TRAIN_CSV_PATH):
            day_df_clean.to_csv(TRAIN_CSV_PATH, index=False)
        else:
            day_df_clean.to_csv(TRAIN_CSV_PATH, mode='a', header=False, index=False)

    # 해당 날짜 .nc 파일 삭제
    date_str = str(date)[:10]
    for nc_type in ['phy', 'bgc']:
        nc_path = os.path.join(CMEMS_DIR, f"cmems_{nc_type}_{date_str}.nc")
        if os.path.exists(nc_path):
            try:
                os.remove(nc_path)
            except Exception as e:
                print(f"{nc_path} 삭제 실패: {e}")

def build_training_data():
    """하드코딩된 격자 셀 기반으로 훈련 데이터 수집"""
    # CSV 파일이 없으면 빈 파일 생성
    if not os.path.exists(TRAIN_CSV_PATH):
        log(f"[build_training_data] 훈련 데이터 파일 생성: {TRAIN_CSV_PATH}")
        os.makedirs(os.path.dirname(TRAIN_CSV_PATH), exist_ok=True)

        empty_df = pd.DataFrame(columns=[
            "lat", "lon", "date", "bottom_temp", "bottom_salinity",
            "sea_surface_height", "oxygen", "primary_production", "species", "present"
        ])
        empty_df.to_csv(TRAIN_CSV_PATH, index=False)

    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(process_one_day, DAYS))
    print("모든 날짜 데이터 처리 및 .nc 파일 정리 완료")

def train_and_export_pmml():
    """훈련 데이터로 모델 학습 및 PMML 내보내기"""
    if not os.path.exists(TRAIN_CSV_PATH):
        log(f"[train_and_export_pmml] 훈련 데이터 파일이 없습니다: {TRAIN_CSV_PATH}")
        return

    df = pd.read_csv(TRAIN_CSV_PATH)
    log(f"[train_and_export_pmml] 전체 데이터: {len(df)}행")

    # 모든 환경 특성 사용
    all_features = ['lat', 'lon', 'bottom_temp', 'bottom_salinity', 'sea_surface_height', 'oxygen', 'primary_production']

    # NaN이 있는 행 제거
    df_clean = df.dropna(subset=all_features + ['present'])
    log(f"[train_and_export_pmml] NaN 제거 후: {len(df_clean)}행")

    if len(df_clean) == 0:
        log("[train_and_export_pmml] 학습할 유효 데이터가 없습니다.")
        return

    X = df_clean[all_features]
    y = df_clean["present"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = PMMLPipeline([("classifier", clf)])
    pipeline.fit(X, y)

    sklearn2pmml(pipeline, PMML_PATH, with_repr=True)
    log(f"[train_and_export_pmml] PMML 저장 완료: {PMML_PATH}")

if __name__ == "__main__":
    try:
        log("[main] 해양 생물 예측 시스템 시작")
        log(f"[main] 격자 셀 수: {len(GRID_CELLS)}개")
        log(f"[main] 처리할 날짜 수: {len(DAYS)}일")

        build_training_data()
        train_and_export_pmml()
        log("[main] 완료.")

    except KeyboardInterrupt:
        log("[main] 사용자 중단.")
    except Exception as e:
        log(f"[main] 오류: {e}")
        import traceback
        log(f"[main] 스택트레이스: {traceback.format_exc()}")