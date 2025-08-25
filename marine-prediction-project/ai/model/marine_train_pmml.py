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
    # 실제 CMEMS API 엔드포인트/파라미터로 수정 필요
    url = f"https://your-cmems-api-endpoint?lat={lat}&lon={lon}&date={date}"
    resp = requests.get(url)
    data = resp.json()
    return {
        "temp": data.get("temperature", np.nan),
        "current_speed": data.get("current_speed", np.nan),
        "current_dir": data.get("current_dir", np.nan),
        "oxygen": data.get("oxygen", np.nan),
        "chlorophyll": data.get("chlorophyll", np.nan),
        "salinity": data.get("salinity", np.nan)
    }

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
    # CMEMS/OBIS API 테스트
    test_lat, test_lon = 36.0, 129.0
    test_date = datetime.now().strftime("%Y-%m-%d")
    print("[CMEMS API 테스트]")
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
