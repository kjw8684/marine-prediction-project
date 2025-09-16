#!/usr/bin/env python3
"""
최적화된 3년치 실제 CMEMS + 해양생물 데이터 통합 AI 학습
- 단일 통합 CSV 파일 사용
- .nc 파일 즉시 삭제
- 배치 처리로 메모리 효율성 확보
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import tempfile
import shutil

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)

from real_data_system import MarineRealDataCollector

class OptimizedThreeYearTrainer:
    """최적화된 3년치 데이터 통합 AI 훈련 시스템 + 병렬처리"""
    
    def __init__(self):
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimized_training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 데이터 수집기 초기화 (병렬처리 지원)
        self.data_collector = MarineRealDataCollector(max_workers=8)
        
        # 격자점 설정 (한국 근해) - 직접 생성
        self.grid_points = self.generate_grid_points()
        
        # 통합 데이터 파일
        self.integrated_file = "three_year_integrated_data.csv"
        self.batch_size = 30  # 30일씩 배치 처리
        
        # 날짜 범위 설정 (CMEMS 데이터 가용 기간)
        start_date = datetime(2022, 6, 1)
        end_date = datetime(2024, 9, 13)
        self.training_dates = []
        current = start_date
        while current <= end_date:
            self.training_dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)  # 주간 간격
        
        self.logger.info(f"🎯 훈련 시스템 초기화 완료")
        self.logger.info(f"📅 날짜 범위: {len(self.training_dates)}일")
        self.logger.info(f"🗺️  격자점: {len(self.grid_points)}개")
        self.logger.info(f"📦 배치 크기: {self.batch_size}일")
    
    def generate_grid_points(self):
        """한국 근해 격자점 생성"""
        grid_points = []
        
        # 위도: 33.5°N ~ 37.5°N (0.5도 간격)
        # 경도: 124.5°E ~ 130.5°E (0.5도 간격)
        for lat in np.arange(33.5, 38.0, 0.5):
            for lon in np.arange(124.5, 131.0, 0.5):
                grid_points.append((lat, lon))
        
        return grid_points
    
    def collect_batch_data(self, date_batch):
        """배치 단위로 데이터 병렬 수집"""
        batch_data = []
        
        # 병렬로 여러 날짜 동시 처리
        self.logger.info(f"� 배치 병렬 처리 시작: {len(date_batch)}일")
        all_data = self.data_collector.collect_multiple_days_parallel(date_batch, self.grid_points)
        
        if all_data:
            self.logger.info(f"✅ 배치 처리 완료: {len(all_data)}개 데이터 포인트")
            return all_data
        else:
            self.logger.warning(f"⚠️ 배치 데이터 없음")
            return []
    
    def cleanup_nc_files(self, date_str):
        """특정 날짜의 .nc 파일들 삭제"""
        try:
            nc_files = [
                f"cmems_phy_{date_str.replace('-', '')}.nc",
                f"cmems_bgc_{date_str.replace('-', '')}.nc"
            ]
            
            cmems_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'cmems_output')
            
            for nc_file in nc_files:
                nc_path = os.path.join(cmems_dir, nc_file)
                if os.path.exists(nc_path):
                    os.remove(nc_path)
                    self.logger.debug(f"🗑️ 삭제: {nc_file}")
        
        except Exception as e:
            self.logger.warning(f"⚠️ .nc 파일 삭제 실패: {e}")
    
    def append_to_integrated_file(self, batch_data):
        """배치 데이터를 통합 파일에 추가"""
        if not batch_data:
            return
        
        try:
            # 배치 데이터 통합
            batch_df = pd.concat(batch_data, ignore_index=True)
            batch_df = batch_df.fillna(0)
            
            # 파일 존재 여부 확인
            file_exists = os.path.exists(self.integrated_file)
            
            # 첫 번째 배치면 헤더 포함, 아니면 헤더 제외하고 추가
            batch_df.to_csv(
                self.integrated_file, 
                mode='a' if file_exists else 'w',
                header=not file_exists,
                index=False, 
                encoding='utf-8'
            )
            
            self.logger.info(f"💾 배치 저장: {len(batch_df)}행 추가")
            
        except Exception as e:
            self.logger.error(f"❌ 배치 저장 실패: {e}")
    
    def collect_all_data(self):
        """전체 데이터 수집 (배치 처리)"""
        self.logger.info("="*60)
        self.logger.info("🌊 최적화된 3년치 데이터 수집 시작!")
        self.logger.info("="*60)
        
        # 기존 통합 파일 삭제
        if os.path.exists(self.integrated_file):
            os.remove(self.integrated_file)
            self.logger.info("🗑️ 기존 통합 파일 삭제")
        
        total_batches = (len(self.training_dates) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(self.training_dates), self.batch_size):
            batch_num = i // self.batch_size + 1
            date_batch = self.training_dates[i:i + self.batch_size]
            
            self.logger.info(f"📦 배치 {batch_num}/{total_batches}: {len(date_batch)}일 처리")
            
            # 배치 데이터 수집
            batch_data = self.collect_batch_data(date_batch)
            
            # 통합 파일에 추가
            self.append_to_integrated_file(batch_data)
            
            self.logger.info(f"✅ 배치 {batch_num} 완료")
        
        # 최종 통합 파일 확인
        if os.path.exists(self.integrated_file):
            final_df = pd.read_csv(self.integrated_file)
            self.logger.info(f"🎉 전체 데이터 수집 완료!")
            self.logger.info(f"📊 최종 데이터: {len(final_df)}행 × {len(final_df.columns)}열")
            return final_df
        else:
            self.logger.error("❌ 통합 파일 생성 실패")
            return pd.DataFrame()
    
    def train_models(self, data_df):
        """AI 모델 훈련"""
        self.logger.info("="*60)
        self.logger.info("🤖 AI 모델 훈련 시작!")
        self.logger.info("="*60)
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            import joblib
            
            # 특성 컬럼 준비
            feature_cols = ['lat', 'lon']
            
            # 환경 데이터 컬럼 추가
            env_cols = ['sea_water_temperature', 'sea_water_salinity', 'sea_surface_height', 
                       'mixed_layer_depth', 'dissolved_oxygen', 'net_primary_productivity']
            feature_cols.extend([col for col in env_cols if col in data_df.columns])
            
            # 대상 종 목록
            target_species = self.data_collector.target_species
            
            models = {}
            models_dir = "optimized_models"
            os.makedirs(models_dir, exist_ok=True)
            
            for species in target_species:
                # 정확한 컬럼명 사용 (GBIF 관측 수)
                species_col = f"{species.replace(' ', '_')}_gbif_observations"
                
                if species_col in data_df.columns:
                    # 관측 데이터가 있는지 확인 (0이 아닌 값이 있는지)
                    if data_df[species_col].sum() > 0:
                        self.logger.info(f"🎯 {species} 모델 훈련 중... (관측수: {data_df[species_col].sum()})")
                        
                        # 데이터 준비
                        X = data_df[feature_cols].values
                        y = data_df[species_col].values
                        
                        # 훈련/테스트 분할
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # 모델 훈련
                        model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        
                        # 성능 평가
                        train_score = r2_score(y_train, model.predict(X_train))
                        test_score = r2_score(y_test, model.predict(X_test))
                        
                        # 모델 저장
                        model_file = os.path.join(models_dir, f"{species.replace(' ', '_')}_model.joblib")
                        joblib.dump(model, model_file)
                        
                        # 모델 정보 저장
                        info_file = os.path.join(models_dir, f"{species.replace(' ', '_')}_info.json")
                        model_info = {
                            'species': species,
                            'features': feature_cols,
                            'train_score': train_score,
                            'test_score': test_score,
                            'target_column': species_col,
                            'training_date': datetime.now().isoformat()
                        }
                        
                        with open(info_file, 'w', encoding='utf-8') as f:
                            json.dump(model_info, f, ensure_ascii=False, indent=2)
                        
                        models[species] = model_info
                        
                        self.logger.info(f"✅ {species}: 훈련 R²={train_score:.3f}, 테스트 R²={test_score:.3f}")
                    else:
                        self.logger.warning(f"⚠️ {species}: 관측 데이터 없음 (모든 값이 0)")
                else:
                    self.logger.warning(f"❌ {species}: 컬럼 없음 ({species_col})")
            
            self.logger.info(f"🎉 모델 훈련 완료: {len(models)}개 모델")
            return models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def export_to_pmml(self, models):
        """모델을 PMML 형식으로 내보내기"""
        self.logger.info("="*60)
        self.logger.info("📤 PMML 모델 내보내기!")
        self.logger.info("="*60)
        
        try:
            from sklearn2pmml import PMMLPipeline, sklearn2pmml
            from sklearn.preprocessing import StandardScaler
            
            pmml_dir = "optimized_pmml_models"
            os.makedirs(pmml_dir, exist_ok=True)
            
            # 통합 데이터 로드
            data_df = pd.read_csv(self.integrated_file)
            
            # 특성 컬럼 준비
            feature_cols = ['lat', 'lon']
            env_cols = ['sea_water_temperature', 'sea_water_salinity', 'sea_surface_height', 
                       'mixed_layer_depth', 'dissolved_oxygen', 'net_primary_productivity']
            feature_cols.extend([col for col in env_cols if col in data_df.columns])
            
            for species, model_info in models.items():
                try:
                    # 모델 로드
                    model_file = os.path.join("optimized_models", f"{species.replace(' ', '_')}_model.joblib")
                    if os.path.exists(model_file):
                        import joblib
                        model = joblib.load(model_file)
                        
                        # PMML 파이프라인 생성 (피쳐 이름 포함)
                        pipeline = PMMLPipeline([
                            ("scaler", StandardScaler()),
                            ("regressor", model)
                        ])
                        
                        # 데이터 준비 (DataFrame 형태로 피쳐 이름 보존)
                        species_col = model_info['target_column']
                        X_df = data_df[feature_cols].copy()
                        y_series = data_df[species_col].copy()
                        y_series.name = species_col  # 타겟 필드 이름 명시
                        
                        # 파이프라인 훈련 (DataFrame과 Series 사용)
                        pipeline.fit(X_df, y_series)
                        
                        # PMML 내보내기
                        pmml_file = os.path.join(pmml_dir, f"{species.replace(' ', '_')}_model.pmml")
                        sklearn2pmml(pipeline, pmml_file, with_repr=True)
                        
                        self.logger.info(f"✅ {species} PMML 내보내기 완료")
                
                except Exception as e:
                    self.logger.error(f"❌ {species} PMML 내보내기 실패: {e}")
            
            self.logger.info("🎉 PMML 내보내기 완료!")
            
        except ImportError:
            self.logger.warning("⚠️ sklearn2pmml이 설치되지 않음. pip install sklearn2pmml로 설치하세요.")
        except Exception as e:
            self.logger.error(f"❌ PMML 내보내기 실패: {e}")
    
    def run_full_training(self):
        """전체 훈련 프로세스 실행"""
        try:
            self.logger.info("🚀 최적화된 3년치 실제 CMEMS + 해양생물 데이터 통합 AI 학습")
            self.logger.info("="*70)
            
            # 1. 데이터 수집
            data_df = self.collect_all_data()
            
            if data_df.empty:
                self.logger.error("❌ 데이터 수집 실패")
                return False
            
            # 2. 모델 훈련
            models = self.train_models(data_df)
            
            if not models:
                self.logger.error("❌ 모델 훈련 실패")
                return False
            
            # 3. PMML 내보내기
            self.export_to_pmml(models)
            
            # 4. 최종 정리
            self.cleanup_all_nc_files()
            
            self.logger.info("🎉 전체 훈련 프로세스 완료!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 전체 프로세스 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup_all_nc_files(self):
        """모든 .nc 파일 정리"""
        try:
            cmems_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'cmems_output')
            if os.path.exists(cmems_dir):
                nc_files = [f for f in os.listdir(cmems_dir) if f.endswith('.nc')]
                for nc_file in nc_files:
                    os.remove(os.path.join(cmems_dir, nc_file))
                
                self.logger.info(f"🗑️ {len(nc_files)}개 .nc 파일 정리 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ .nc 파일 정리 실패: {e}")

def main():
    """메인 실행 함수"""
    trainer = OptimizedThreeYearTrainer()
    success = trainer.run_full_training()
    
    if success:
        print("\n🎉 최적화된 3년치 AI 모델 훈련 완료!")
        print("📁 생성된 파일:")
        print("  - three_year_integrated_data.csv (통합 데이터)")
        print("  - optimized_models/ (joblib 모델)")
        print("  - optimized_pmml_models/ (PMML 모델)")
        print("  - optimized_training.log (로그)")
    else:
        print("\n❌ 훈련 실패")

if __name__ == "__main__":
    main()
