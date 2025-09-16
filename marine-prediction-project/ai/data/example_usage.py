"""
실제 격자점들을 활용한 효율적인 데이터 처리 예제
속도 우선의 접근법
"""

import pandas as pd
import numpy as np
from grid_points import get_grid_points, get_grid_array, get_grid_info, get_nearest_grid_point

def process_environmental_data_for_actual_grids():
    """
    실제 데이터가 있는 격자점들에 대해서만 환경 데이터 처리
    """
    grid_points = get_grid_points()
    grid_info = get_grid_info()
    
    print(f"실제 사용할 격자점 수: {grid_info['total_points']}")
    print(f"위도 범위: {grid_info['lat_range']}")
    print(f"경도 범위: {grid_info['lon_range']}")
    
    # 예시: 각 격자점에 대한 처리
    results = []
    for i, (lat, lon) in enumerate(grid_points):
        # 실제로는 여기서 CMEMS 데이터 추출, 모델 예측 등을 수행
        result = {
            'grid_id': i,
            'lat': lat,
            'lon': lon,
            'processed': True
        }
        results.append(result)
    
    return pd.DataFrame(results)

def batch_predict_for_actual_grids(date_str="2024-01-01"):
    """
    실제 격자점들에 대해서만 배치 예측 수행
    """
    grid_points = get_grid_points()
    grid_array = get_grid_array()
    
    print(f"{date_str}에 대해 {len(grid_points)}개 격자점에서 예측 수행")
    
    # 벡터화된 처리 (NumPy 배열 활용)
    predictions = []
    
    for lat, lon in grid_points:
        # 실제로는 여기서 모델 예측을 수행
        # 예시: 간단한 더미 예측
        prediction = {
            'date': date_str,
            'lat': lat,
            'lon': lon,
            'species': 'target_species',
            'probability': np.random.random(),  # 실제로는 모델 예측값
            'temperature': 15.0 + np.random.random() * 10,  # 실제로는 CMEMS 데이터
            'salinity': 34.0 + np.random.random() * 2
        }
        predictions.append(prediction)
    
    return pd.DataFrame(predictions)

def filter_training_data_by_actual_grids():
    """
    훈련 데이터를 실제 격자점들로 필터링
    """
    # 이미 훈련 데이터는 실제 격자점들만 포함하고 있지만,
    # 새로운 데이터가 들어올 때 필터링하는 예제
    
    grid_points = set(get_grid_points())
    
    # 예시: 새로운 데이터 포인트들
    new_data = [
        (35.2, 129.7),  # 실제 격자점이 아님
        (35.0, 129.5),  # 실제 격자점
        (34.7, 128.3),  # 실제 격자점이 아님
        (34.5, 128.0),  # 실제 격자점
    ]
    
    # 실제 격자점만 유지하거나 가장 가까운 격자점으로 매핑
    filtered_data = []
    mapped_data = []
    
    for lat, lon in new_data:
        if (lat, lon) in grid_points:
            filtered_data.append((lat, lon))
        
        # 가장 가까운 격자점으로 매핑
        nearest = get_nearest_grid_point(lat, lon)
        mapped_data.append(nearest)
    
    print(f"원본 데이터: {new_data}")
    print(f"필터링된 데이터: {filtered_data}")
    print(f"매핑된 데이터: {mapped_data}")
    
    return filtered_data, mapped_data

def create_prediction_grid_output():
    """
    예측 결과를 격자 형태로 출력하기 위한 구조 생성
    """
    grid_points = get_grid_points()
    grid_info = get_grid_info()
    
    # 실제 격자점들로 구성된 예측 결과 템플릿
    prediction_template = {
        'grid_info': grid_info,
        'points': [],
        'metadata': {
            'total_points': len(grid_points),
            'data_type': 'actual_grid_only',
            'irregular_grid': True  # 불규칙 격자임을 표시
        }
    }
    
    for i, (lat, lon) in enumerate(grid_points):
        point_data = {
            'grid_id': i,
            'lat': lat,
            'lon': lon,
            'prediction': None,  # 실제 예측값이 들어갈 자리
            'confidence': None,
            'environmental_data': {}
        }
        prediction_template['points'].append(point_data)
    
    return prediction_template

if __name__ == "__main__":
    print("=== 실제 격자점 기반 효율적인 처리 예제 ===\n")
    
    # 1. 격자점 정보 출력
    grid_info = get_grid_info()
    print(f"격자 정보: {grid_info}\n")
    
    # 2. 환경 데이터 처리
    print("1. 환경 데이터 처리:")
    processed_df = process_environmental_data_for_actual_grids()
    print(f"처리된 격자점 수: {len(processed_df)}\n")
    
    # 3. 배치 예측
    print("2. 배치 예측:")
    predictions_df = batch_predict_for_actual_grids()
    print(f"예측 수행된 격자점 수: {len(predictions_df)}")
    print(f"평균 예측 확률: {predictions_df['probability'].mean():.3f}\n")
    
    # 4. 데이터 필터링/매핑
    print("3. 데이터 필터링 및 매핑:")
    filtered, mapped = filter_training_data_by_actual_grids()
    print()
    
    # 5. 예측 결과 구조
    print("4. 예측 결과 템플릿:")
    template = create_prediction_grid_output()
    print(f"템플릿 생성 완료 - {template['metadata']['total_points']}개 격자점")
    
    print("\n=== 속도 최적화된 격자점 처리 완료 ===")
