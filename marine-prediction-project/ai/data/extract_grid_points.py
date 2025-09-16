"""
실제 데이터가 있는 격자점들을 추출하여 하드코딩된 리스트로 저장
속도 우선의 접근 방식
"""

import pandas as pd

# 훈련 데이터에서 실제 사용된 격자점들 추출
df = pd.read_csv('training_data_korea_full.csv')
coords = df[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon'])

print(f"총 {len(coords)}개의 격자점이 실제 데이터에 사용됨")
print(f"위도 범위: {coords['lat'].min()} ~ {coords['lat'].max()}")
print(f"경도 범위: {coords['lon'].min()} ~ {coords['lon'].max()}")

# 하드코딩을 위한 Python 리스트 형태로 출력
print("\n# 실제 데이터가 있는 격자점들 (lat, lon)")
print("GRID_POINTS = [")
for _, row in coords.iterrows():
    print(f"    ({row['lat']}, {row['lon']}),")
print("]")

# 격자점들을 별도 파일로도 저장
coords.to_csv('actual_grid_points.csv', index=False)
print(f"\n격자점들이 'actual_grid_points.csv'에 저장되었습니다.")

# NumPy 배열 형태로도 출력
import numpy as np
coords_array = coords.values
print(f"\n# NumPy 배열 형태:")
print(f"import numpy as np")
print(f"GRID_POINTS_ARRAY = np.array({coords_array.tolist()})")

print(f"\n통계:")
print(f"- 총 격자점 수: {len(coords)}")
print(f"- 고유한 위도 값: {coords['lat'].nunique()}개")
print(f"- 고유한 경도 값: {coords['lon'].nunique()}개")
print(f"- 격자 간격: 위도 {(coords['lat'].max() - coords['lat'].min()) / (coords['lat'].nunique() - 1):.1f}도, 경도 {(coords['lon'].max() - coords['lon'].min()) / (coords['lon'].nunique() - 1):.1f}도")
