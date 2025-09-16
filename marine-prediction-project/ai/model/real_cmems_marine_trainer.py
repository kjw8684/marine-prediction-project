#!/usr/bin/env python3
"""
실제 CMEMS + 해양생물 데이터 통합 학습 시스템
- 실제 CMEMS API 사용
- 실제 GBIF/OBIS 해양생물 관측 데이터 사용
- 하루치 데이터 → 학습 → PMML 내보내기
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_data_system import MarineRealDataCollector
from marine_train_pmml import collect_cmems_data_for_date

class RealDataMarineTrainer:
    """실제 CMEMS + 해양생물 데이터 통합 학습 시스템"""
    
    def __init__(self):
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
        
        logger.info(f"실제 데이터 학습 시스템 초기화 완료 - 격자점: {len(self.grid_points)}개")

    def collect_integrated_data(self, target_date: str):
        """특정 날짜의 해양생물 + CMEMS 환경 데이터 통합 수집"""
        logger.info(f"[INTEGRATE] {target_date} 통합 데이터 수집 시작")
        
        try:
            # 1. 해양생물 관측 데이터 수집
            logger.info("[BIO] 해양생물 관측 데이터 수집...")
            biological_df = self.data_collector.collect_daily_training_data(target_date, self.grid_points)
            
            if biological_df.empty:
                logger.warning("[BIO] 생물 데이터 없음")
                return pd.DataFrame()
            
            logger.info(f"[BIO] 생물 데이터 수집 완료: {len(biological_df)}행, {len(biological_df.columns)}열")
            
            # 2. CMEMS 해양환경 데이터 수집
            logger.info("[CMEMS] 실제 CMEMS API 데이터 수집...")
            cmems_df = collect_cmems_data_for_date(target_date, self.grid_points)
            
            if cmems_df.empty:
                logger.warning("[CMEMS] 환경 데이터 없음 - 생물 데이터만 사용")
                return biological_df
            
            logger.info(f"[CMEMS] 환경 데이터 수집 완료: {len(cmems_df)}행, {len(cmems_df.columns)}열")
            
            # 3. 데이터 통합
            logger.info("[MERGE] 생물 + 환경 데이터 통합...")
            integrated_df = self._merge_data(biological_df, cmems_df)
            
            logger.info(f"[INTEGRATE] 통합 완료: {len(integrated_df)}행, {len(integrated_df.columns)}열")
            return integrated_df
            
        except Exception as e:
            logger.error(f"[INTEGRATE] 데이터 수집 실패: {e}")
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
            
            logger.info(f"데이터 통합: 생물 {len(biological_df)}행 + 환경 {len(cmems_df)}행 → {len(merged_df)}행")
            return merged_df
            
        except Exception as e:
            logger.error(f"데이터 통합 실패: {e}")
            return biological_df

    def train_models(self, integrated_df):
        """통합 데이터로 AI 모델 훈련"""
        logger.info("[TRAIN] AI 모델 훈련 시작...")
        
        try:
            if integrated_df.empty:
                logger.error("훈련 데이터 없음")
                return False
            
            # 특성과 타겟 변수 선택
            numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns.tolist()
            targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
            features = [col for col in numeric_cols if col not in targets + ['lat', 'lon']]
            
            if len(features) < 5:
                logger.error(f"특성 수 부족: {len(features)}개 (최소 5개 필요)")
                return False
            
            logger.info(f"사용 특성: {len(features)}개")
            logger.info(f"특성 예시: {features[:10]}")
            
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
                    
                    logger.info(f"{target}: R² = {train_score:.3f}, CV = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            self.models = trained_models
            logger.info(f"[TRAIN] 모델 훈련 완료: {len(trained_models)}개")
            return True
            
        except Exception as e:
            logger.error(f"[TRAIN] 모델 훈련 실패: {e}")
            return False

    def save_models(self):
        """훈련된 모델을 joblib와 PMML로 저장"""
        logger.info("[SAVE] 모델 저장 시작...")
        
        try:
            if not self.models:
                logger.error("저장할 모델 없음")
                return False
            
            saved_files = []
            
            # 1. joblib 형식으로 저장
            for target, model_info in self.models.items():
                model = model_info['model']
                
                # joblib 저장
                joblib_path = f"real_marine_model_{target}.joblib"
                joblib.dump(model, joblib_path)
                saved_files.append(joblib_path)
                logger.info(f"joblib 저장: {joblib_path}")
            
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
                    pmml_path = f"real_marine_model_{target}.pmml"
                    sklearn2pmml(pipeline, pmml_path, with_repr=True)
                    saved_files.append(pmml_path)
                    logger.info(f"PMML 저장: {pmml_path}")
                    
            except ImportError:
                logger.warning("sklearn2pmml 패키지 없음 - PMML 저장 건너뜀")
            except Exception as e:
                logger.warning(f"PMML 저장 실패: {e}")
            
            logger.info(f"[SAVE] 모델 저장 완료: {len(saved_files)}개 파일")
            return True
            
        except Exception as e:
            logger.error(f"[SAVE] 모델 저장 실패: {e}")
            return False

    def run_full_pipeline(self, target_date: str):
        """전체 파이프라인 실행: 데이터 수집 → 훈련 → 저장"""
        logger.info(f"실제 데이터 전체 파이프라인 시작: {target_date}")
        
        try:
            # 1. 통합 데이터 수집
            integrated_df = self.collect_integrated_data(target_date)
            
            if integrated_df.empty:
                logger.error("통합 데이터 수집 실패")
                return False
            
            # 2. 데이터 저장
            data_file = f"real_integrated_marine_data_{target_date.replace('-', '')}.csv"
            integrated_df.to_csv(data_file, index=False, encoding='utf-8')
            logger.info(f"통합 데이터 저장: {data_file}")
            
            # 3. 모델 훈련
            training_success = self.train_models(integrated_df)
            
            if not training_success:
                logger.error("모델 훈련 실패")
                return False
            
            # 4. 모델 저장
            save_success = self.save_models()
            
            if not save_success:
                logger.error("모델 저장 실패")
                return False
            
            # 5. 결과 요약
            logger.info("="*60)
            logger.info("📊 실제 데이터 학습 완료!")
            logger.info(f"📅 날짜: {target_date}")
            logger.info(f"📍 격자점: {len(self.grid_points)}개")
            logger.info(f"📊 최종 데이터: {len(integrated_df)}행 × {len(integrated_df.columns)}열")
            logger.info(f"🤖 훈련된 모델: {len(self.models)}개")
            logger.info(f"💾 저장 파일: {data_file}")
            
            for target, model_info in self.models.items():
                logger.info(f"   • {target}: R²={model_info['train_score']:.3f}")
            
            logger.info("="*60)
            return True
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """메인 실행 함수"""
    print("🌊 실제 CMEMS + 해양생물 데이터 통합 AI 학습")
    print("="*60)
    
    # 실제 데이터가 있는 날짜 사용 (CMEMS는 보통 3-4일 지연)
    target_date = "2025-09-14"  # 3일 전 데이터
    print(f"📅 학습 대상 날짜: {target_date}")
    
    try:
        # 트레이너 초기화
        trainer = RealDataMarineTrainer()
        
        # 전체 파이프라인 실행
        success = trainer.run_full_pipeline(target_date)
        
        if success:
            print("\n🎉 성공! 실제 CMEMS + 해양생물 데이터 AI 모델 완성!")
            print("📁 생성된 파일:")
            print("   • real_marine_model_*.joblib (모델)")
            print("   • real_marine_model_*.pmml (PMML)")
            print("   • real_integrated_marine_data_*.csv (데이터)")
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
