#!/usr/bin/env python3
"""
최적화된 3년치 실제 CMEMS + 해양생물 데이터 통합 AI 학습
- 7일 간격 학습으로 효율성 개선
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
    """최적화된 3년치 데이터 통합 AI 훈련 시스템 (7일 간격)"""
    
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
        
        # 데이터 수집기 초기화 (적절한 병렬처리)
        self.data_collector = MarineRealDataCollector(max_workers=2)
        
        # 통합 데이터 파일
        self.integrated_file = "three_year_weekly_integrated_data.csv"
        
        # 날짜 범위 설정 (CMEMS 데이터 가용 기간) - 7일 간격
        start_date = datetime(2022, 6, 1)
        end_date = datetime(2024, 9, 13)
        
        self.logger.info(f"🎯 7일 간격 훈련 시스템 초기화 완료")
        self.logger.info(f"📅 날짜 범위: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"🗺️  격자점: 한국근해 0.5도 해상도")
        self.logger.info(f"📈 학습 방식: 7일 간격, 생물데이터는 ±3일 범위")
    
    def collect_weekly_training_data(self, start_date_str, end_date_str):
        """7일 간격 훈련 데이터 수집"""
        
        self.logger.info(f"🔄 7일 간격 데이터 수집 시작: {start_date_str} ~ {end_date_str}")
        
        # 새로운 주간 데이터 수집 메서드 사용
        collected_data_path = self.data_collector.collect_weekly_training_data(
            start_date=start_date_str,
            end_date=end_date_str,
            lat_range=(33.5, 37.5),
            lon_range=(124.5, 130.5),
            resolution=0.5
        )
        
        if collected_data_path and os.path.exists(collected_data_path):
            # CSV 파일 읽기
            df = pd.read_csv(collected_data_path, encoding='utf-8-sig')
            self.logger.info(f"✅ 7일 간격 데이터 수집 완료: {len(df)}개 데이터 포인트")
            
            return df, collected_data_path
        else:
            self.logger.warning(f"⚠️ 7일 간격 데이터 수집 실패")
            return None, None

    def run_full_training_pipeline(self):
        """전체 훈련 파이프라인 실행 (7일 간격)"""
        
        try:
            self.logger.info("🚀 3년치 7일 간격 해양 AI 훈련 시작")
            
            # 1. 3년치 데이터를 6개월씩 나누어 처리
            start_date = datetime(2022, 6, 1)
            end_date = datetime(2024, 9, 13)
            
            all_data = []
            current_start = start_date
            batch_count = 0
            
            while current_start < end_date:
                # 6개월 배치 설정
                current_end = min(current_start + timedelta(days=180), end_date)
                batch_count += 1
                
                self.logger.info(f"📦 배치 {batch_count}: {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")
                
                # 배치 데이터 수집
                batch_df, temp_file = self.collect_weekly_training_data(
                    current_start.strftime('%Y-%m-%d'),
                    current_end.strftime('%Y-%m-%d')
                )
                
                if batch_df is not None:
                    all_data.append(batch_df)
                    self.logger.info(f"  ✅ 배치 {batch_count} 완료: {len(batch_df)} 행")
                    
                    # 임시 파일 정리
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                else:
                    self.logger.warning(f"  ⚠️ 배치 {batch_count} 실패")
                
                # 다음 배치로
                current_start = current_end + timedelta(days=1)
            
            # 2. 모든 데이터 통합
            if all_data:
                self.logger.info("🔧 데이터 통합 중...")
                integrated_df = pd.concat(all_data, ignore_index=True)
                
                # 통합 파일 저장
                integrated_df.to_csv(self.integrated_file, index=False, encoding='utf-8-sig')
                
                self.logger.info(f"💾 통합 데이터 저장 완료: {self.integrated_file}")
                self.logger.info(f"   총 데이터: {len(integrated_df):,} 행")
                self.logger.info(f"   파일 크기: {os.path.getsize(self.integrated_file) / 1024 / 1024:.1f} MB")
                
                # 3. 데이터 품질 검증
                self.validate_integrated_data(integrated_df)
                
                # 4. AI 모델 훈련
                self.train_ai_models(integrated_df)
                
                return True
            else:
                self.logger.error("❌ 수집된 데이터가 없습니다")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 훈련 파이프라인 실패: {e}")
            return False
    
    def validate_integrated_data(self, df):
        """통합 데이터 품질 검증"""
        
        self.logger.info("🔍 데이터 품질 검증 중...")
        
        # 기본 통계
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # NULL 값 확인
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > total_rows * 0.5]
        
        # 생물 데이터 확인
        bio_cols = [col for col in df.columns if '_density' in col]
        env_cols = [col for col in df.columns if col.startswith('cmems_')]
        
        self.logger.info(f"  📊 기본 정보: {total_rows:,} 행, {total_cols} 열")
        self.logger.info(f"  🐟 생물 변수: {len(bio_cols)}개")
        self.logger.info(f"  🌊 환경 변수: {len(env_cols)}개")
        self.logger.info(f"  ⚠️  높은 NULL 컬럼: {len(high_null_cols)}개")
        
        if len(high_null_cols) > 0:
            self.logger.warning(f"     NULL 비율 높음: {list(high_null_cols.index)}")
        
        # 날짜 분포 확인
        if 'collection_date' in df.columns:
            date_counts = df['collection_date'].value_counts()
            self.logger.info(f"  📅 수집 날짜: {len(date_counts)}개 (7일 간격)")
        
        return True
    
    def train_ai_models(self, df):
        """AI 모델 훈련 (랜덤 포레스트 + PMML 내보내기)"""
        
        try:
            self.logger.info("🧠 AI 모델 훈련 시작...")
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.preprocessing import PMMLLabelEncoder
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            # 특성 및 타겟 준비
            feature_cols = [col for col in df.columns if col.startswith('cmems_') or 
                           col in ['latitude', 'longitude', 'depth_m', 'distance_to_coast_km']]
            
            target_species = [
                'Aurelia_aurita', 'Chrysaora_pacifica', 'Scomber_japonicus',
                'Engraulis_japonicus', 'Todarodes_pacificus', 'Trachurus_japonicus',
                'Sardinops_melanostictus', 'Chaetodon_nippon'
            ]
            
            # 각 종별로 모델 훈련
            for species in target_species:
                density_col = f"{species}_density"
                weight_col = f"{species}_weight"
                
                if density_col not in df.columns:
                    continue
                
                self.logger.info(f"  🐟 {species} 모델 훈련 중...")
                
                # 데이터 준비
                valid_data = df.dropna(subset=feature_cols + [density_col])
                
                if len(valid_data) < 100:
                    self.logger.warning(f"    ⚠️ {species}: 데이터 부족 ({len(valid_data)}행)")
                    continue
                
                X = valid_data[feature_cols]
                y = valid_data[density_col]
                
                # 가중치 적용 (실제 관측 데이터는 높은 가중치)
                sample_weights = valid_data[weight_col] if weight_col in valid_data.columns else None
                
                # 훈련/테스트 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                if sample_weights is not None:
                    weights_train = sample_weights.loc[X_train.index]
                    weights_test = sample_weights.loc[X_test.index]
                else:
                    weights_train = None
                    weights_test = None
                
                # 모델 훈련 (가중치 적용)
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train, sample_weight=weights_train)
                
                # 예측 및 평가
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred, sample_weight=weights_test)
                r2 = r2_score(y_test, y_pred, sample_weight=weights_test)
                
                self.logger.info(f"    ✅ {species}: MSE={mse:.4f}, R²={r2:.4f}")
                
                # 모델 저장 (joblib)
                joblib_path = f"marine_ai_model_{species.lower()}.joblib"
                joblib.dump(model, joblib_path)
                
                # PMML 내보내기
                try:
                    pmml_pipeline = PMMLPipeline([
                        ("regressor", model)
                    ])
                    pmml_pipeline.fit(X_train, y_train)
                    
                    pmml_path = f"marine_ai_model_{species.lower()}.pmml"
                    sklearn2pmml(pmml_pipeline, pmml_path)
                    
                    self.logger.info(f"    💾 PMML 저장: {pmml_path}")
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️ PMML 저장 실패 ({species}): {e}")
            
            self.logger.info("🎉 AI 모델 훈련 완료!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ AI 모델 훈련 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    trainer = OptimizedThreeYearTrainer()
    
    try:
        success = trainer.run_full_training_pipeline()
        
        if success:
            print("\n🎉 3년치 7일 간격 해양 AI 훈련 성공!")
            print(f"📁 통합 데이터: {trainer.integrated_file}")
            print("📄 PMML 파일들이 생성되었습니다.")
        else:
            print("\n❌ 훈련 실패")
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")

if __name__ == "__main__":
    main()
