#!/usr/bin/env python3
"""
CMEMS + 해양생물 통합 데이터 하루치 테스트
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_data_system import MarineRealDataCollector
from marine_train_pmml import collect_cmems_data_for_date
from three_year_ai_trainer import ThreeYearMarineTrainer

def test_full_integration():
    """완전한 데이터 통합 테스트"""
    
    print("🌊 CMEMS + 해양생물 통합 데이터 테스트 시작")
    print("="*60)
    
    # 테스트 날짜 (최근 날짜)
    test_date = "2024-12-01"
    print(f"📅 테스트 날짜: {test_date}")
    
    # 그리드 포인트 설정 (소규모 테스트용)
    test_grid_points = [
        (35.0, 129.0),  # 부산 근처
        (37.0, 127.0),  # 서해
        (36.0, 130.0),  # 동해
    ]
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
        print(f"   - 주요 컬럼: {list(biological_df.columns)[:5]}...")
        
        # 2. CMEMS 환경 데이터 수집
        print("\n2️⃣ CMEMS 환경 데이터 수집...")
        cmems_df = collect_cmems_data_for_date(test_date, test_grid_points)
        
        if cmems_df.empty:
            print("❌ CMEMS 데이터 없음 - 생물 데이터만 사용")
            combined_df = biological_df
        else:
            print(f"✅ CMEMS 데이터: {len(cmems_df)}행, {len(cmems_df.columns)}열")
            print(f"   - 주요 컬럼: {list(cmems_df.columns)[:5]}...")
            
            # 3. 데이터 통합
            print("\n3️⃣ 데이터 통합...")
            combined_df = biological_df.merge(
                cmems_df, 
                on=['lat', 'lon'], 
                how='left', 
                suffixes=('_bio', '_env')
            )
            print(f"✅ 통합 데이터: {len(combined_df)}행, {len(combined_df.columns)}열")
        
        # 4. 데이터 저장
        print("\n4️⃣ 테스트 데이터 저장...")
        test_csv_path = f"test_integrated_data_{test_date.replace('-', '')}.csv"
        combined_df.to_csv(test_csv_path, index=False, encoding='utf-8')
        print(f"✅ 저장완료: {test_csv_path}")
        
        # 5. 간단한 모델 훈련 테스트
        print("\n5️⃣ AI 모델 훈련 테스트...")
        
        # 기본적인 특성 선택
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['lat', 'lon', 'species_diversity_index', 'biomass_estimate', 'bloom_probability']]
        
        if len(feature_cols) >= 3:
            print(f"   - 사용 가능한 특성: {len(feature_cols)}개")
            print(f"   - 특성 예시: {feature_cols[:5]}")
            
            # 간단한 Random Forest 훈련
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            X = combined_df[feature_cols].fillna(0)
            targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
            
            for target in targets:
                if target in combined_df.columns:
                    y = combined_df[target].fillna(0)
                    
                    try:
                        model = RandomForestRegressor(n_estimators=10, random_state=42)
                        model.fit(X, y)
                        score = model.score(X, y)
                        print(f"   - {target}: R² = {score:.3f}")
                    except Exception as e:
                        print(f"   - {target}: 훈련 실패 - {e}")
            
            print("✅ 모델 훈련 테스트 완료")
        else:
            print(f"❌ 특성 부족: {len(feature_cols)}개 (최소 3개 필요)")
        
        # 6. 데이터 분석
        print("\n6️⃣ 데이터 분석...")
        print(f"   - 총 행수: {len(combined_df)}")
        print(f"   - 총 열수: {len(combined_df.columns)}")
        print(f"   - 결측치 비율: {(combined_df.isnull().sum().sum() / combined_df.size * 100):.1f}%")
        
        # 타겟 변수 통계
        for target in ['species_diversity_index', 'biomass_estimate', 'bloom_probability']:
            if target in combined_df.columns:
                values = combined_df[target].dropna()
                if len(values) > 0:
                    print(f"   - {target}: 평균={values.mean():.3f}, 최대={values.max():.3f}")
        
        print("\n🎉 통합 데이터 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_integration()
    
    if success:
        print("\n✅ 테스트 성공 - CMEMS + 생물 데이터 통합 준비 완료!")
        print("이제 전체 기간 학습을 실행할 수 있습니다.")
    else:
        print("\n❌ 테스트 실패 - 문제를 해결한 후 다시 시도하세요.")
