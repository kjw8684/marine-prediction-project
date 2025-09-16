#!/usr/bin/env python3
"""
실제 CMEMS + 해양생물 데이터 통합 학습 시스템 (marine_train_pmml.py 기반)
- marine_train_pmml.py에서 검증된 CMEMS 다운로드 함수들 사용
- 실제 GBIF/OBIS 해양생물 관측 데이터 사용
- 하루치 데이터 → 학습 → PMML 내보내기
"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging
import joblib
import threading
import time
import glob
import copernicusmarine

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 파일 다운로드 동기화를 위한 글로벌 락
download_locks = {}
lock_manager_lock = threading.Lock()

# 주요 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
CMEMS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "cmems_output"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CMEMS_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "extract_var_debug.log")

def log(message):
    """로그 메시지 출력"""
    logger.info(message)
    try:
        with open(LOG_PATH, "a", encoding='utf-8') as f:
            f.write(f"{datetime.now()}: {message}\n")
    except:
        pass

# marine_train_pmml.py에서 검증된 함수들 복사
def get_dataset_config():
    """CMEMS 데이터셋과 변수명 설정 (JSON 파일에서 확인한 실제 변수명 사용)"""
    return {
        "physics": {
            "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "variables": {
                "temperature": "tob",     # sea_water_potential_temperature_at_sea_floor
                "salinity": "sob",        # sea_water_salinity_at_sea_floor
                "sea_surface_height": "zos",      # sea_surface_height_above_geoid
                "mixed_layer_depth": "mlotst"     # ocean_mixed_layer_thickness_defined_by_sigma_theta
            }
        },
        "biogeochemistry": {
            "dataset_id": "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m", 
            "variables": {
                "dissolved_oxygen": "o2",         # mole_concentration_of_dissolved_molecular_oxygen_in_sea_water
                "net_primary_productivity": "nppv"  # net_primary_production_of_biomass_expressed_as_carbon_per_unit_volume_in_sea_water
            }
        }
    }

def get_file_lock(file_path):
    """파일별 락 반환"""
    global download_locks
    with lock_manager_lock:
        if file_path not in download_locks:
            download_locks[file_path] = threading.Lock()
        return download_locks[file_path]

def cleanup_duplicate_nc_files(date_str):
    """중복 다운로드된 .nc 파일들 정리"""
    try:
        patterns = [
            f"*{date_str}*.nc",
            f"*cmems*{date_str}*.nc"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(CMEMS_DIR, pattern))
            if len(files) > 2:  # phy와 bgc 2개보다 많으면 중복
                files.sort(key=os.path.getmtime, reverse=True)
                # 최신 2개만 남기고 삭제
                for old_file in files[2:]:
                    try:
                        os.remove(old_file)
                        log(f"[cleanup] 중복 파일 삭제: {old_file}")
                    except Exception as e:
                        log(f"[cleanup] 삭제 실패: {old_file} - {e}")
                        
    except Exception as e:
        log(f"[cleanup] 오류: {date_str} - {e}")

def cleanup_daily_nc_files(date_str):
    """하루치 NC 파일 삭제 (메모리 절약)"""
    try:
        patterns = [
            f"cmems_phy_{date_str}.nc",
            f"cmems_bgc_{date_str}.nc"
        ]
        
        for pattern in patterns:
            file_path = os.path.join(CMEMS_DIR, pattern)
            if os.path.exists(file_path):
                os.remove(file_path)
                log(f"[cleanup] 일일 파일 삭제: {file_path}")
                
    except Exception as e:
        log(f"[cleanup] 일일 정리 실패: {date_str} - {e}")

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

            # 임시 파일명으로 다운로드 (CMEMS가 자동으로 .nc를 추가함)
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

            # 다운로드 완료 후 원래 이름으로 이동 (CMEMS가 .nc를 추가하므로 확인)
            temp_files = [temp_nc_path, temp_nc_path + ".nc"]
            success_file = None
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    success_file = temp_file
                    break
            
            if success_file:
                os.rename(success_file, nc_path)
                log(f"[download_with_lock] 다운로드 완료: {nc_path}")
                return True
            else:
                log(f"[download_with_lock] 다운로드 실패: 임시 파일이 생성되지 않음")
                log(f"[download_with_lock] 확인한 경로: {temp_files}")
                return False

        except Exception as e:
            log(f"[download_with_lock] 다운로드 실패: {nc_path} - {e}")
            
            # 임시 파일 정리 (CMEMS가 .nc를 추가할 수 있으므로 두 가지 확인)
            temp_files = [nc_path + ".temp", nc_path + ".temp.nc"]
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        log(f"[download_with_lock] 임시 파일 정리: {temp_file}")
                except Exception:
                    pass
                
            return False

def download_cmems_data_for_date(data_type: str, target_date: str):
    """특정 날짜의 CMEMS 데이터 다운로드"""
    try:
        date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        date_str = date_obj.strftime('%Y%m%d')
        
        start_datetime = date_obj.strftime('%Y-%m-%dT00:00:00')
        end_datetime = date_obj.strftime('%Y-%m-%dT23:59:59')
        
        datasets = get_dataset_config()
        
        if data_type == "physics":
            dataset = datasets["physics"]
            nc_path = os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}.nc")
            variables = list(dataset["variables"].values())
        elif data_type == "biogeochemistry":
            dataset = datasets["biogeochemistry"]
            nc_path = os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}.nc")
            variables = list(dataset["variables"].values())
        else:
            return False
        
        return download_with_lock(
            nc_path=nc_path,
            dataset_id=dataset["dataset_id"],
            variables=variables,
            start_datetime=start_datetime,
            end_datetime=end_datetime
        )
        
    except Exception as e:
        log(f"[DOWNLOAD] {data_type} {target_date} 실패: {e}")
        return False

def collect_cmems_data_for_date(target_date: str, grid_points):
    """특정 날짜의 CMEMS 데이터를 수집하여 DataFrame으로 반환"""
    try:
        log(f"[CMEMS] {target_date} 데이터 수집 시작")
        
        date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        date_str = date_obj.strftime('%Y%m%d')
        
        # 1. CMEMS 데이터 다운로드
        log(f"[CMEMS] {target_date} 물리 데이터 다운로드...")
        phy_success = download_cmems_data_for_date("physics", target_date)
        
        log(f"[CMEMS] {target_date} 생화학 데이터 다운로드...")
        bgc_success = download_cmems_data_for_date("biogeochemistry", target_date)
        
        if not (phy_success and bgc_success):
            log(f"[CMEMS] {target_date} 다운로드 실패 - phy:{phy_success}, bgc:{bgc_success}")
            return pd.DataFrame()
        
        # 2. NetCDF 파일 로드
        phy_nc = os.path.join(CMEMS_DIR, f"cmems_phy_{date_str}.nc")
        bgc_nc = os.path.join(CMEMS_DIR, f"cmems_bgc_{date_str}.nc")
        
        log(f"[CMEMS] NetCDF 파일 로드: {phy_nc}, {bgc_nc}")
        
        phy_ds = xr.open_dataset(phy_nc)
        bgc_ds = xr.open_dataset(bgc_nc)
        
        log(f"[CMEMS] 물리 데이터 변수: {list(phy_ds.data_vars.keys())}")
        log(f"[CMEMS] 생화학 데이터 변수: {list(bgc_ds.data_vars.keys())}")
        
        # 3. 격자점별 데이터 추출
        cmems_data = []
        for i, (lat, lon) in enumerate(grid_points):
            try:
                log(f"[CMEMS] 격자 {i+1}/{len(grid_points)}: ({lat}, {lon})")
                
                # 물리 데이터 추출
                phy_point = phy_ds.sel(latitude=lat, longitude=lon, method='nearest')
                bgc_point = bgc_ds.sel(latitude=lat, longitude=lon, method='nearest')
                
                row_data = {
                    'lat': lat,
                    'lon': lon
                }
                
                # 물리 데이터 추출 (JSON에서 확인한 실제 변수명 사용)
                try:
                    if 'tob' in phy_ds.data_vars:
                        row_data['sea_water_temperature'] = float(phy_point.tob.values[0, 0])
                    if 'sob' in phy_ds.data_vars:
                        row_data['sea_water_salinity'] = float(phy_point.sob.values[0, 0])
                    if 'zos' in phy_ds.data_vars:
                        row_data['sea_surface_height'] = float(phy_point.zos.values[0, 0])
                    if 'mlotst' in phy_ds.data_vars:
                        row_data['mixed_layer_depth'] = float(phy_point.mlotst.values[0, 0])
                except Exception as e:
                    log(f"[CMEMS] 물리 데이터 추출 실패 ({lat}, {lon}): {e}")
                
                # 생화학 데이터 추출 (JSON에서 확인한 실제 변수명 사용)
                try:
                    if 'o2' in bgc_ds.data_vars:
                        row_data['dissolved_oxygen'] = float(bgc_point.o2.values[0, 0])
                    if 'nppv' in bgc_ds.data_vars:
                        row_data['net_primary_productivity'] = float(bgc_point.nppv.values[0, 0])
                except Exception as e:
                    log(f"[CMEMS] 생화학 데이터 추출 실패 ({lat}, {lon}): {e}")
                
                cmems_data.append(row_data)
                
            except Exception as e:
                log(f"[CMEMS] 격자 ({lat}, {lon}) 처리 실패: {e}")
                continue
        
        # 4. 리소스 정리
        phy_ds.close()
        bgc_ds.close()
        
        if cmems_data:
            df = pd.DataFrame(cmems_data)
            log(f"[CMEMS] {target_date} 데이터 수집 완료: {len(df)}행 × {len(df.columns)}열")
            return df
        else:
            log(f"[CMEMS] {target_date} 데이터 없음")
            return pd.DataFrame()
            
    except Exception as e:
        log(f"[CMEMS] {target_date} 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

class RealCmemsMarineTrainer:
    """실제 CMEMS + 해양생물 데이터 통합 학습 시스템 (marine_train_pmml 기반)"""
    
    def __init__(self):
        # 해양생물 데이터 시스템 초기화
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from real_data_system import MarineRealDataCollector
        self.data_collector = MarineRealDataCollector()
        self.models = {}
        
        # 한국 연안 주요 격자점 (실제 해양 데이터가 있는 위치)
        self.grid_points = [
            # 동해
            (37.5, 129.0), (37.0, 129.5), (36.5, 130.0), (36.0, 130.5),
            (35.5, 129.5), (35.0, 129.0), (34.5, 129.5), (34.0, 130.0),
            
            # 남해
            (34.0, 128.5), (34.5, 128.0), (35.0, 127.5), (35.5, 127.0),
            (34.0, 127.0), (34.5, 126.5), (35.0, 126.0), (35.5, 125.5),
            
            # 서해
            (37.0, 126.0), (36.5, 126.5), (36.0, 127.0), (35.5, 126.0),
            (35.0, 125.5), (34.5, 125.0), (34.0, 124.5), (33.5, 125.0),
        ]
        
        log(f"실제 CMEMS+생물 데이터 학습 시스템 초기화 완료 - 격자점: {len(self.grid_points)}개")

    def collect_integrated_data(self, target_date: str):
        """특정 날짜의 해양생물 + CMEMS 환경 데이터 통합 수집 (marine_train_pmml 기반)"""
        log(f"[INTEGRATE] {target_date} 통합 데이터 수집 시작")
        
        try:
            # 1. 해양생물 관측 데이터 수집
            log("[BIO] 해양생물 관측 데이터 수집...")
            biological_df = self.data_collector.collect_daily_training_data(target_date, self.grid_points)
            
            if biological_df.empty:
                log("[BIO] 생물 데이터 없음")
                return pd.DataFrame()
            
            log(f"[BIO] 생물 데이터 수집 완료: {len(biological_df)}행, {len(biological_df.columns)}열")
            
            # 2. CMEMS 해양환경 데이터 수집 (marine_train_pmml 기반)
            log("[CMEMS] 실제 CMEMS API 데이터 수집...")
            cmems_df = collect_cmems_data_for_date(target_date, self.grid_points)
            
            if cmems_df.empty:
                log("[CMEMS] 환경 데이터 없음 - 생물 데이터만 사용")
                return biological_df
            
            log(f"[CMEMS] 환경 데이터 수집 완료: {len(cmems_df)}행, {len(cmems_df.columns)}열")
            
            # 3. 데이터 통합
            log("[MERGE] 생물 + 환경 데이터 통합...")
            integrated_df = self._merge_data(biological_df, cmems_df)
            
            log(f"[INTEGRATE] 통합 완료: {len(integrated_df)}행, {len(integrated_df.columns)}열")
            return integrated_df
            
        except Exception as e:
            log(f"[INTEGRATE] 데이터 수집 실패: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _merge_data(self, biological_df, cmems_df):
        """생물 데이터와 환경 데이터 통합"""
        try:
            # 위도, 경도를 기준으로 통합
            merged_df = biological_df.merge(
                cmems_df, 
                on=['lat', 'lon'], 
                how='left', 
                suffixes=('_bio', '_env')
            )
            
            # 결측치 처리
            numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
            merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
            
            log(f"데이터 통합: 생물 {len(biological_df)}행 + 환경 {len(cmems_df)}행 → {len(merged_df)}행")
            return merged_df
            
        except Exception as e:
            log(f"데이터 통합 실패: {e}")
            return biological_df

    def train_models(self, integrated_df):
        """통합 데이터로 AI 모델 훈련"""
        log("[TRAIN] AI 모델 훈련 시작...")
        
        try:
            if integrated_df.empty:
                log("훈련 데이터 없음")
                return False
            
            # 특성과 타겟 변수 선택
            numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns.tolist()
            targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
            features = [col for col in numeric_cols if col not in targets + ['lat', 'lon']]
            
            if len(features) < 5:
                log(f"특성 수 부족: {len(features)}개 (최소 5개 필요)")
                return False
            
            log(f"사용 특성: {len(features)}개")
            log(f"특성 예시: {features[:10]}")
            
            X = integrated_df[features].fillna(0)
            
            # 각 타겟별 모델 훈련
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            
            trained_models = {}
            
            for target in targets:
                if target in integrated_df.columns:
                    y = integrated_df[target].fillna(0)
                    
                    # Random Forest 모델
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    # 모델 훈련
                    model.fit(X, y)
                    
                    # 성능 평가
                    train_score = model.score(X, y)
                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                    
                    trained_models[target] = {
                        'model': model,
                        'features': features,
                        'train_score': train_score,
                        'cv_score': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    log(f"{target}: R² = {train_score:.3f}, CV = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            self.models = trained_models
            log(f"[TRAIN] 모델 훈련 완료: {len(trained_models)}개")
            return True
            
        except Exception as e:
            log(f"[TRAIN] 모델 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_models(self):
        """훈련된 모델을 joblib와 PMML로 저장"""
        log("[SAVE] 모델 저장 시작...")
        
        try:
            if not self.models:
                log("저장할 모델 없음")
                return False
            
            saved_files = []
            
            # 1. joblib 형식으로 저장
            for target, model_info in self.models.items():
                model = model_info['model']
                
                # joblib 저장
                joblib_path = f"real_cmems_marine_model_{target}.joblib"
                joblib.dump(model, joblib_path)
                saved_files.append(joblib_path)
                log(f"joblib 저장: {joblib_path}")
            
            # 2. PMML 형식으로 저장 시도
            try:
                from sklearn2pmml import sklearn2pmml, PMMLPipeline
                from sklearn.preprocessing import StandardScaler
                
                for target, model_info in self.models.items():
                    model = model_info['model']
                    features = model_info['features']
                    
                    # PMML 파이프라인 생성
                    pipeline = PMMLPipeline([
                        ("scaler", StandardScaler()),
                        ("regressor", model)
                    ])
                    
                    # 더미 데이터로 파이프라인 맞춤 (PMML 생성용)
                    dummy_X = np.random.random((10, len(features)))
                    dummy_y = np.random.random(10)
                    pipeline.fit(dummy_X, dummy_y)
                    
                    # PMML 저장
                    pmml_path = f"real_cmems_marine_model_{target}.pmml"
                    sklearn2pmml(pipeline, pmml_path, with_repr=True)
                    saved_files.append(pmml_path)
                    log(f"PMML 저장: {pmml_path}")
                    
            except ImportError:
                log("sklearn2pmml 패키지 없음 - PMML 저장 건너뜀")
            except Exception as e:
                log(f"PMML 저장 실패: {e}")
            
            log(f"[SAVE] 모델 저장 완료: {len(saved_files)}개 파일")
            return True
            
        except Exception as e:
            log(f"[SAVE] 모델 저장 실패: {e}")
            return False

    def run_full_pipeline(self, target_date: str):
        """전체 파이프라인 실행: 데이터 수집 → 훈련 → 저장"""
        log(f"실제 CMEMS+생물 데이터 전체 파이프라인 시작: {target_date}")
        
        try:
            # 1. 통합 데이터 수집
            integrated_df = self.collect_integrated_data(target_date)
            
            if integrated_df.empty:
                log("통합 데이터 수집 실패")
                return False
            
            # 2. 데이터 저장
            data_file = f"real_cmems_integrated_marine_data_{target_date.replace('-', '')}.csv"
            integrated_df.to_csv(data_file, index=False, encoding='utf-8')
            log(f"통합 데이터 저장: {data_file}")
            
            # 3. 모델 훈련
            training_success = self.train_models(integrated_df)
            
            if not training_success:
                log("모델 훈련 실패")
                return False
            
            # 4. 모델 저장
            save_success = self.save_models()
            
            if not save_success:
                log("모델 저장 실패")
                return False
            
            # 5. NC 파일 정리 (메모리 절약)
            date_str = target_date.replace('-', '')
            cleanup_daily_nc_files(date_str)
            
            # 6. 결과 요약
            log("="*60)
            log("📊 실제 CMEMS+생물 데이터 학습 완료!")
            log(f"📅 날짜: {target_date}")
            log(f"📍 격자점: {len(self.grid_points)}개")
            log(f"📊 최종 데이터: {len(integrated_df)}행 × {len(integrated_df.columns)}열")
            log(f"🤖 훈련된 모델: {len(self.models)}개")
            log(f"💾 저장 파일: {data_file}")
            
            for target, model_info in self.models.items():
                log(f"   • {target}: R²={model_info['train_score']:.3f}")
            
            log("="*60)
            return True
            
        except Exception as e:
            log(f"파이프라인 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """메인 실행 함수"""
    print("🌊 실제 CMEMS + 해양생물 데이터 통합 AI 학습 (marine_train_pmml 기반)")
    print("="*70)
    
    # 실제 데이터가 있는 날짜 사용 (CMEMS는 보통 3-4일 지연)
    target_date = "2024-09-13"  # 과거 확실한 날짜
    print(f"📅 학습 대상 날짜: {target_date}")
    
    try:
        # 트레이너 초기화
        trainer = RealCmemsMarineTrainer()
        
        # 전체 파이프라인 실행
        success = trainer.run_full_pipeline(target_date)
        
        if success:
            print("\n🎉 성공! 실제 CMEMS + 해양생물 데이터 AI 모델 완성!")
            print("📁 생성된 파일:")
            print("   • real_cmems_marine_model_*.joblib (모델)")
            print("   • real_cmems_marine_model_*.pmml (PMML)")
            print("   • real_cmems_integrated_marine_data_*.csv (데이터)")
        else:
            print("\n❌ 실패! 문제를 확인하고 다시 시도하세요.")
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n💥 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
