#!/usr/bin/env python3
"""
최종 학습 완료된 PMML 모델 성능 검증 스크립트
환경 기반 예측 시스템이 제대로 작동하는지 확인
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def test_pmml_models():
    """최종 PMML 모델들의 예측 성능을 테스트"""
    
    # 모델 파일들 확인
    model_files = {
        'Aurelia_aurita': 'marine_ai_model_aurelia_aurita.joblib',
        'Chrysaora_pacifica': 'marine_ai_model_chrysaora_pacifica.joblib',
        'Scomber_japonicus': 'marine_ai_model_scomber_japonicus.joblib',
        'Engraulis_japonicus': 'marine_ai_model_engraulis_japonicus.joblib',
        'Todarodes_pacificus': 'marine_ai_model_todarodes_pacificus.joblib',
        'Trachurus_japonicus': 'marine_ai_model_trachurus_japonicus.joblib',
        'Sardinops_melanostictus': 'marine_ai_model_sardinops_melanostictus.joblib',
        'Chaetodon_nippon': 'marine_ai_model_chaetodon_nippon.joblib'
    }
    
    # 테스트용 환경 데이터 생성 (다양한 시나리오)
    test_scenarios = [
        # 겨울철 차가운 바다
        {
            'name': '겨울철 차가운 바다',
            'temperature': 8.0,
            'salinity': 33.5,
            'ph': 8.1,
            'chlorophyll': 0.8,
            'oxygen': 7.2,
            'nitrate': 15.0,
            'month': 1,  # 1월
            'latitude': 35.0,
            'longitude': 127.0,
            'depth_estimate': 50.0,
            'distance_to_coast': 10.0
        },
        
        # 여름철 따뜻한 바다
        {
            'name': '여름철 따뜻한 바다',
            'temperature': 25.0,
            'salinity': 32.8,
            'ph': 8.0,
            'chlorophyll': 2.5,
            'oxygen': 6.5,
            'nitrate': 8.0,
            'month': 8,  # 8월
            'latitude': 35.0,
            'longitude': 127.0,
            'depth_estimate': 50.0,
            'distance_to_coast': 10.0
        },
        
        # 봄철 연안
        {
            'name': '봄철 연안 지역',
            'temperature': 18.0,
            'salinity': 31.5,
            'ph': 8.2,
            'chlorophyll': 3.2,
            'oxygen': 7.8,
            'nitrate': 12.0,
            'month': 5,  # 5월
            'latitude': 37.0,
            'longitude': 126.5,
            'depth_estimate': 20.0,
            'distance_to_coast': 2.0
        },
        
        # 가을철 외해
        {
            'name': '가을철 외해',
            'temperature': 15.0,
            'salinity': 34.2,
            'ph': 8.1,
            'chlorophyll': 1.2,
            'oxygen': 7.0,
            'nitrate': 18.0,
            'month': 10,  # 10월
            'latitude': 33.5,
            'longitude': 129.0,
            'depth_estimate': 100.0,
            'distance_to_coast': 50.0
        }
    ]
    
    print("🔍 최종 PMML 모델 성능 검증")
    print("="*60)
    
    results = {}
    
    for species_name, model_file in model_files.items():
        try:
            # Joblib 모델 로드
            model = joblib.load(model_file)
            print(f"\n📊 {species_name} 모델 테스트:")
            
            species_results = []
            
            for scenario in test_scenarios:
                # 입력 데이터 준비
                input_data = pd.DataFrame([{
                    'temperature': scenario['temperature'],
                    'salinity': scenario['salinity'], 
                    'ph': scenario['ph'],
                    'chlorophyll': scenario['chlorophyll'],
                    'oxygen': scenario['oxygen'],
                    'nitrate': scenario['nitrate'],
                    'month': scenario['month'],
                    'latitude': scenario['latitude'],
                    'longitude': scenario['longitude'],
                    'depth_estimate': scenario['depth_estimate'],
                    'distance_to_coast': scenario['distance_to_coast']
                }])
                
                # 예측 수행
                prediction = model.predict(input_data)
                pred_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else float(prediction)
                
                species_results.append(pred_value)
                print(f"   {scenario['name']:15s}: {pred_value:.6f}")
            
            results[species_name] = species_results
            
            # 다양성 분석
            unique_predictions = len(set(np.round(species_results, 6)))
            min_pred = min(species_results)
            max_pred = max(species_results)
            std_pred = np.std(species_results)
            
            print(f"   📈 예측 다양성: {unique_predictions}/4 시나리오")
            print(f"   📏 범위: {min_pred:.6f} ~ {max_pred:.6f}")
            print(f"   📐 표준편차: {std_pred:.6f}")
            
            if unique_predictions > 1:
                print("   ✅ 환경에 따른 예측 변화 확인")
            else:
                print("   ❌ 모든 예측이 동일 (모델 문제 가능성)")
                
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results[species_name] = None
    
    print("\n" + "="*60)
    print("🎯 전체 결과 요약")
    
    working_models = 0
    varied_predictions = 0
    
    for species_name, species_results in results.items():
        if species_results is not None:
            working_models += 1
            unique_count = len(set(np.round(species_results, 6)))
            if unique_count > 1:
                varied_predictions += 1
    
    print(f"✅ 작동하는 모델: {working_models}/8")
    print(f"🎨 다양한 예측 모델: {varied_predictions}/8")
    
    if varied_predictions == working_models and working_models > 0:
        print("\n🎉 성공! 환경 기반 예측 시스템이 제대로 작동합니다!")
        print("   - 모든 모델이 환경 조건에 따라 다른 예측을 생성합니다")
        print("   - Heatmap 생성용 PMML 파일들이 준비되었습니다")
        return "success"
    elif working_models > 0:
        print("\n⚠️  부분적 성공: 일부 모델은 작동하지만 개선이 필요합니다")
        print("   - 환경 조건에 상관없이 동일한 예측을 하는 모델이 있습니다")
        return "partial_success"
    else:
        print("\n❌ 실패: 모든 모델에 문제가 있습니다")
        return "failure"
    
    return results

def check_data_quality():
    """학습 데이터 품질 확인"""
    print("\n" + "="*60)
    print("📊 학습 데이터 품질 분석")
    
    try:
        # 통합 데이터 로드
        data = pd.read_csv('three_year_weekly_integrated_data.csv')
        print(f"📁 데이터 크기: {data.shape[0]:,} 행, {data.shape[1]} 열")
        
        # 생물 데이터 컬럼들 확인
        bio_columns = [col for col in data.columns if any(species in col.lower() for species in 
                      ['aurelia', 'chrysaora', 'scomber', 'engraulis', 'todarodes', 'trachurus', 'sardinops', 'chaetodon'])]
        
        print(f"🐟 생물 데이터 컬럼: {len(bio_columns)}개")
        
        for col in bio_columns:
            non_zero_count = (data[col] != 0).sum()
            unique_values = data[col].nunique()
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            print(f"   {col:25s}: 0이 아닌 값 {non_zero_count:,}개, 고유값 {unique_values:,}개")
            print(f"                            평균: {mean_val:.6f}, 표준편차: {std_val:.6f}")
        
        # 환경 데이터 확인
        env_columns = ['temperature', 'salinity', 'ph', 'chlorophyll', 'oxygen', 'nitrate']
        print(f"\n🌊 환경 데이터 요약:")
        for col in env_columns:
            if col in data.columns:
                print(f"   {col:12s}: 평균 {data[col].mean():.3f}, 범위 {data[col].min():.3f}~{data[col].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 분석 오류: {e}")
        return False

if __name__ == "__main__":
    print("🚀 최종 해양 AI 모델 성능 검증 시작")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 품질 확인
    data_ok = check_data_quality()
    
    # 모델 성능 테스트
    results = test_pmml_models()
    
    print(f"\n⏰ 검증 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
