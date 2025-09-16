#!/usr/bin/env python3
"""
간단한 CMEMS + 생물 데이터 통합 테스트
"""

print("🌊 해양 데이터 통합 시스템 테스트")
print("="*50)

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    print("✅ 기본 라이브러리 로드 성공")
    
    # real_data_system 모듈 테스트
    from real_data_system import MarineRealDataCollector
    print("✅ real_data_system 모듈 로드 성공")
    
    # marine_train_pmml 모듈 테스트
    from marine_train_pmml import collect_cmems_data_for_date
    print("✅ marine_train_pmml 모듈 로드 성공")
    
    # 테스트 날짜와 격자
    test_date = "2024-12-01"
    test_grid = [(35.0, 129.0), (37.0, 127.0)]
    
    print(f"\n📅 테스트 날짜: {test_date}")
    print(f"🗺️ 테스트 격자: {len(test_grid)}개")
    
    # 1. 생물 데이터 수집 테스트
    print("\n1️⃣ 해양생물 데이터 수집 테스트...")
    data_collector = MarineRealDataCollector()
    bio_df = data_collector.collect_daily_training_data(test_date, test_grid)
    
    if not bio_df.empty:
        print(f"✅ 생물 데이터: {len(bio_df)}행, {len(bio_df.columns)}열")
        print(f"   주요 컬럼: {', '.join(bio_df.columns[:5])}")
    else:
        print("⚠️ 생물 데이터 없음")
    
    # 2. CMEMS 데이터 수집 테스트
    print("\n2️⃣ CMEMS 데이터 수집 테스트...")
    cmems_df = collect_cmems_data_for_date(test_date, test_grid)
    
    if not cmems_df.empty:
        print(f"✅ CMEMS 데이터: {len(cmems_df)}행, {len(cmems_df.columns)}열")
        print(f"   주요 컬럼: {', '.join(cmems_df.columns[:5])}")
    else:
        print("⚠️ CMEMS 데이터 없음 (정상 - 최신 날짜는 데이터 없을 수 있음)")
    
    # 3. 데이터 통합 테스트
    print("\n3️⃣ 데이터 통합 테스트...")
    if not bio_df.empty:
        if not cmems_df.empty:
            # 실제 통합
            merged_df = bio_df.merge(cmems_df, on=['lat', 'lon'], how='left', suffixes=('_bio', '_env'))
            print(f"✅ 데이터 통합: {len(merged_df)}행, {len(merged_df.columns)}열")
        else:
            # 생물 데이터만 사용
            merged_df = bio_df.copy()
            print(f"✅ 생물 데이터만 사용: {len(merged_df)}행, {len(merged_df.columns)}열")
        
        # 4. CSV 저장
        output_file = f"integrated_test_data_{test_date.replace('-', '')}.csv"
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 데이터 저장: {output_file}")
        
        # 5. 기본 통계
        print(f"\n📊 데이터 요약:")
        print(f"   - 행수: {len(merged_df)}")
        print(f"   - 열수: {len(merged_df.columns)}")
        print(f"   - 결측치: {merged_df.isnull().sum().sum()}개")
        
        # 타겟 변수 확인
        targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
        for target in targets:
            if target in merged_df.columns:
                values = merged_df[target].dropna()
                if len(values) > 0:
                    print(f"   - {target}: 평균={values.mean():.3f}")
    
    print("\n🎉 테스트 완료!")

except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()
