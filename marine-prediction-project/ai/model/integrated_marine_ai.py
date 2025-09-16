#!/usr/bin/env python3
"""
CMEMS + 해양생물 데이터 통합 학습 (데모 버전)
실제 CMEMS API 대신 시뮬레이션된 환경 데이터 사용
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_data_system import MarineRealDataCollector

def generate_demo_cmems_data(grid_points, target_date):
    """데모용 CMEMS 환경 데이터 생성"""
    logger.info(f"[DEMO_CMEMS] {target_date} 환경 데이터 생성...")
    
    cmems_data = []
    for lat, lon in grid_points:
        # 위치와 날짜에 따른 현실적인 해양환경 데이터 시뮬레이션
        month = int(target_date.split('-')[1])
        
        # 계절별 수온 패턴
        base_temp = 15 + 8 * np.sin((month - 3) * np.pi / 6)  # 계절 변화
        lat_temp_effect = (lat - 35) * -0.5  # 위도 효과
        temperature = base_temp + lat_temp_effect + np.random.normal(0, 1)
        
        # 위치별 염분도
        if lon < 127:  # 서해 (낮은 염분)
            salinity = 30 + np.random.normal(0, 1)
        else:  # 동해/남해 (높은 염분)
            salinity = 34 + np.random.normal(0, 0.5)
        
        # 기타 환경 변수들
        row_data = {
            'lat': lat,
            'lon': lon,
            'sea_surface_temperature': round(temperature, 2),
            'sea_surface_salinity': round(salinity, 2),
            'sea_surface_height': round(np.random.normal(0, 0.2), 3),
            'mixed_layer_depth': round(np.random.uniform(10, 50), 1),
            'chlorophyll': round(np.random.lognormal(0, 1), 3),
            'dissolved_oxygen': round(np.random.normal(250, 30), 1),
            'nitrate': round(np.random.uniform(0.1, 15), 2),
            'phosphate': round(np.random.uniform(0.01, 2), 3),
            'ph': round(np.random.normal(8.1, 0.1), 2),
            'net_primary_productivity': round(np.random.uniform(0.1, 10), 2)
        }
        
        cmems_data.append(row_data)
    
    df = pd.DataFrame(cmems_data)
    logger.info(f"[DEMO_CMEMS] 생성 완료: {len(df)}행, {len(df.columns)}열")
    return df

def merge_biological_and_environmental(biological_df, cmems_df):
    """생물 데이터와 환경 데이터 통합"""
    try:
        if cmems_df is None or cmems_df.empty:
            logger.warning("CMEMS 데이터 없음, 생물 데이터만 사용")
            return biological_df
        
        # 위도, 경도를 기준으로 데이터 통합
        merged_df = biological_df.merge(
            cmems_df, 
            on=['lat', 'lon'], 
            how='left', 
            suffixes=('_bio', '_env')
        )
        
        logger.info(f"데이터 통합 완료: 생물 {len(biological_df)}행 + 환경 {len(cmems_df)}행 → {len(merged_df)}행")
        return merged_df
        
    except Exception as e:
        logger.error(f"데이터 통합 실패: {e}")
        return biological_df

def train_integrated_model(combined_df):
    """통합 데이터로 AI 모델 훈련"""
    try:
        if combined_df.empty:
            logger.error("훈련 데이터 없음")
            return None
        
        # 특성 선택
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
        features = [col for col in numeric_cols if col not in targets + ['lat', 'lon']]
        
        if len(features) < 3:
            logger.error(f"특성 수 부족: {len(features)}개")
            return None
        
        logger.info(f"사용 특성: {len(features)}개")
        logger.info(f"특성 목록: {features[:10]}...")  # 처음 10개만 표시
        
        X = combined_df[features].fillna(0)
        models = {}
        
        # 각 타겟별 모델 훈련
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        for target in targets:
            if target in combined_df.columns:
                y = combined_df[target].fillna(0)
                
                # Random Forest 모델 훈련
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                
                # 교차 검증 점수
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                
                models[target] = model
                logger.info(f"{target}: R² = {model.score(X, y):.3f}, CV = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        return models
        
    except Exception as e:
        logger.error(f"모델 훈련 실패: {e}")
        return None

def export_models_to_pmml(models, combined_df):
    """훈련된 모델을 PMML로 내보내기"""
    try:
        if not models:
            logger.error("내보낼 모델 없음")
            return False
        
        # 특성 선택 (모델 훈련 시와 동일)
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
        features = [col for col in numeric_cols if col not in targets + ['lat', 'lon']]
        
        # joblib로 모델 저장
        for target, model in models.items():
            model_path = f"integrated_model_{target}.joblib"
            import joblib
            joblib.dump(model, model_path)
            logger.info(f"모델 저장: {model_path}")
        
        # PMML 내보내기 시도
        try:
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn.preprocessing import StandardScaler
            
            for target, model in models.items():
                # PMML 파이프라인 생성
                pipeline = PMMLPipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", model)
                ])
                
                # 파이프라인 훈련
                X = combined_df[features].fillna(0)
                y = combined_df[target].fillna(0)
                pipeline.fit(X, y)
                
                # PMML 내보내기
                pmml_path = f"integrated_model_{target}.pmml"
                sklearn2pmml(pipeline, pmml_path, with_repr=True)
                logger.info(f"PMML 저장: {pmml_path}")
                
        except ImportError:
            logger.warning("sklearn2pmml 패키지 없음 - PMML 내보내기 건너뜀")
        except Exception as e:
            logger.warning(f"PMML 내보내기 실패: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"모델 내보내기 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🌊 CMEMS + 해양생물 통합 학습 시스템")
    print("="*60)
    
    # 테스트 설정
    test_date = "2024-12-01"
    test_grid_points = [
        (35.0, 129.0), (35.5, 129.5), (36.0, 130.0),  # 동해
        (34.5, 127.0), (35.0, 127.5), (35.5, 128.0),  # 남해  
        (36.5, 126.0), (37.0, 126.5), (37.5, 127.0),  # 서해
    ]
    
    print(f"📅 테스트 날짜: {test_date}")
    print(f"🗺️ 테스트 격자: {len(test_grid_points)}개")
    
    try:
        # 1. 해양생물 데이터 수집
        print("\n1️⃣ 해양생물 데이터 수집...")
        data_collector = MarineRealDataCollector()
        biological_df = data_collector.collect_daily_training_data(test_date, test_grid_points)
        
        if biological_df.empty:
            print("❌ 생물 데이터 없음")
            return False
        
        print(f"✅ 생물 데이터: {len(biological_df)}행, {len(biological_df.columns)}열")
        
        # 2. 환경 데이터 생성 (CMEMS 데모)
        print("\n2️⃣ 해양환경 데이터 생성...")
        cmems_df = generate_demo_cmems_data(test_grid_points, test_date)
        print(f"✅ 환경 데이터: {len(cmems_df)}행, {len(cmems_df.columns)}열")
        
        # 3. 데이터 통합
        print("\n3️⃣ 데이터 통합...")
        combined_df = merge_biological_and_environmental(biological_df, cmems_df)
        print(f"✅ 통합 데이터: {len(combined_df)}행, {len(combined_df.columns)}열")
        
        # 4. 통합 데이터 저장
        output_file = f"integrated_marine_data_{test_date.replace('-', '')}.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 데이터 저장: {output_file}")
        
        # 5. AI 모델 훈련
        print("\n4️⃣ AI 모델 훈련...")
        models = train_integrated_model(combined_df)
        
        if models:
            print(f"✅ 모델 훈련 완료: {len(models)}개")
            
            # 6. 모델 내보내기
            print("\n5️⃣ 모델 내보내기...")
            export_success = export_models_to_pmml(models, combined_df)
            
            if export_success:
                print("✅ 모델 내보내기 완료")
            else:
                print("⚠️ 모델 내보내기 부분 실패")
        else:
            print("❌ 모델 훈련 실패")
            return False
        
        # 7. 결과 요약
        print("\n6️⃣ 결과 요약")
        print(f"   📊 최종 데이터: {len(combined_df)}행 × {len(combined_df.columns)}열")
        print(f"   🤖 훈련된 모델: {len(models)}개")
        print(f"   📁 출력 파일: {output_file}")
        
        # 환경 변수 통계
        env_cols = [col for col in combined_df.columns if col.endswith('_env') or col in ['sea_surface_temperature', 'chlorophyll']]
        if env_cols:
            print(f"   🌊 환경 변수: {len(env_cols)}개")
            print(f"      예시: {env_cols[:3]}...")
        
        print("\n🎉 통합 학습 완료!")
        print("이제 실제 예측에 사용할 수 있습니다.")
        
        return True
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ 성공: CMEMS + 생물 데이터 통합 AI 모델 완성!")
    else:
        print("\n❌ 실패: 문제를 해결한 후 다시 시도하세요.")
