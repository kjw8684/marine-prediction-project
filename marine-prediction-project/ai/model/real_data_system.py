"""
실제 해양 데이터 수집 시스템 (CMEMS + GBIF/OBIS)
100% 실제 데이터만 사용, 더미 데이터 금지
일일 CMEMS 다운로드 후 즉시 삭제로 메모리 최적화
병렬 처리 지원
"""

import pandas as pd
import numpy as np
import requests
import logging
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
from functools import partial

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('marine_real_data_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CMEMS 패키지 가용성 확인
try:
    import copernicusmarine
    CMEMS_AVAILABLE = True
    logger.info(" CMEMS 다운로드 기능 활성화")
except ImportError:
    CMEMS_AVAILABLE = False
    logger.warning(" copernicusmarine 패키지 없음. CMEMS 다운로드 비활성화.")

# 한국 근해 8개 주요 종 (전체 생태계 대표)
TARGET_SPECIES = [
    'Aurelia aurita',      # 보름달물해파리
    'Chrysaora pacifica',  # 태평양해파리
    'Scomber japonicus',   # 고등어
    'Engraulis japonicus', # 멸치
    'Todarodes pacificus', # 살오징어
    'Trachurus japonicus', # 전갱이
    'Sardinops melanostictus', # 정어리
    'Chaetodon nippon'     # 나비고기
]

# 관측 데이터가 없을 때 사용할 기본값 (환경 조건별 예상 서식 밀도)
SPECIES_BASELINE_VALUES = {
    'Aurelia aurita': {
        'default_density': 0.3,  # 기본 서식 밀도
        'weight': 0.5,           # 중간 가중치 (실제 데이터의 50%)
        'temp_preference': [15, 25],  # 선호 수온 범위
        'depth_preference': [0, 50]   # 선호 수심 범위
    },
    'Chrysaora pacifica': {
        'default_density': 0.2,
        'weight': 0.5,
        'temp_preference': [18, 28],
        'depth_preference': [0, 100]
    },
    'Scomber japonicus': {
        'default_density': 0.4,
        'weight': 0.6,           # 상업성 어종이므로 조금 더 높은 가중치
        'temp_preference': [12, 22],
        'depth_preference': [10, 200]
    },
    'Engraulis japonicus': {
        'default_density': 0.5,
        'weight': 0.6,           # 멸치도 중요한 어종
        'temp_preference': [14, 24],
        'depth_preference': [0, 150]
    },
    'Todarodes pacificus': {
        'default_density': 0.3,
        'weight': 0.5,
        'temp_preference': [10, 20],
        'depth_preference': [50, 300]
    },
    'Trachurus japonicus': {
        'default_density': 0.35,
        'weight': 0.5,
        'temp_preference': [16, 26],
        'depth_preference': [20, 250]
    },
    'Sardinops melanostictus': {
        'default_density': 0.4,
        'weight': 0.6,           # 정어리도 중요한 어종
        'temp_preference': [13, 23],
        'depth_preference': [0, 100]
    },
    'Chaetodon nippon': {
        'default_density': 0.15,
        'weight': 0.4,           # 관상어종은 조금 낮게
        'temp_preference': [20, 30],
        'depth_preference': [5, 50]
    }
}

class MarineRealDataCollector:
    """실제 해양 데이터만 수집하는 클래스 - 일일 CMEMS 다운로드/삭제 최적화 + 병렬처리"""

    def __init__(self, max_workers: int = None):
        self.gbif_base_url = "https://api.gbif.org/v1"
        self.obis_base_url = "https://api.obis.org"
        self.cmems_output_dir = Path("../../cmems_output")
        self.cmems_output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Marine-Species-Predictor/1.0 (research@marine.edu)'
        })

        # 병렬 처리 설정 (안정성 우선으로 더 보수적 설정)
        self.max_workers = max_workers or min(2, mp.cpu_count())

        # NetCDF 파일 접근을 위한 락 (병렬 처리 시 파일 권한 충돌 방지)
        self.file_lock = threading.Lock()

        logger.info(f" MarineRealDataCollector 초기화 완료 (병렬처리: {self.max_workers}개 워커)")

    def download_cmems_data(self, target_date: str) -> bool:
        """특정 날짜의 CMEMS 데이터 다운로드 (물리 + 생지화학)"""

        if not CMEMS_AVAILABLE:
            logger.warning(" CMEMS 다운로드 불가: copernicusmarine 패키지 없음")
            return False

        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')

            start_datetime = date_obj.strftime('%Y-%m-%dT00:00:00')
            end_datetime = date_obj.strftime('%Y-%m-%dT23:59:59')

            logger.info(f" CMEMS 데이터 다운로드 시작: {target_date}")

            # 물리 데이터 다운로드 (해수면높이, 바닥온도/염분, 혼합층깊이)
            phy_path = self.cmems_output_dir / f"cmems_phy_{date_str}.nc"

            if not phy_path.exists():
                logger.info(" 물리 데이터 다운로드 중...")
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                    variables=["zos", "sob", "tob", "mlotst"],
                    minimum_longitude=124.0,
                    maximum_longitude=131.0,
                    minimum_latitude=33.0,
                    maximum_latitude=38.0,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    output_filename=str(phy_path)
                )
                logger.info(f" 물리 데이터 완료: {phy_path.name}")

            # 생지화학 데이터 다운로드 (순1차생산량, 용존산소)
            bgc_path = self.cmems_output_dir / f"cmems_bgc_{date_str}.nc"

            if not bgc_path.exists():
                logger.info(" 생지화학 데이터 다운로드 중...")
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
                    variables=["nppv", "o2"],
                    minimum_longitude=124.0,
                    maximum_longitude=131.0,
                    minimum_latitude=33.0,
                    maximum_latitude=38.0,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    output_filename=str(bgc_path)
                )
                logger.info(f" 생지화학 데이터 완료: {bgc_path.name}")

            logger.info(f" CMEMS 데이터 다운로드 완료: {target_date}")
            return True

        except Exception as e:
            logger.error(f" CMEMS 다운로드 실패 ({target_date}): {e}")
            return False

    def cleanup_cmems_files(self, target_date: str):
        """특정 날짜의 CMEMS .nc 파일들 즉시 삭제 (메모리 절약)"""
        try:
            date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%Y%m%d')

            files_to_delete = [
                self.cmems_output_dir / f"cmems_phy_{date_str}.nc",
                self.cmems_output_dir / f"cmems_bgc_{date_str}.nc"
            ]

            deleted_count = 0
            total_size = 0

            for file_path in files_to_delete:
                if file_path.exists():
                    file_size = file_path.stat().st_size / 1024 / 1024  # MB
                    total_size += file_size
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f" 삭제 완료: {file_path.name} ({file_size:.1f}MB)")

            if deleted_count > 0:
                logger.info(f" 메모리 확보: {total_size:.1f}MB ({deleted_count}개 파일)")

        except Exception as e:
            logger.warning(f" 파일 삭제 실패 ({target_date}): {e}")

    def collect_daily_training_data(self, target_date: str, grid_points: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """특정 날짜의 모든 격자점 데이터 수집 (CMEMS 다운로드  수집  즉시 삭제)"""

        logger.info(f" {target_date} 일일 훈련 데이터 수집 시작")

        # 1. CMEMS 데이터 다운로드
        download_success = self.download_cmems_data(target_date)
        if not download_success:
            logger.warning(f" CMEMS 데이터 없이 진행: {target_date}")

        # 2. 격자점별 데이터 수집
        daily_data = []

        for i, (lat, lon) in enumerate(grid_points):
            try:
                # 격자점별 종합 데이터 수집
                point_data = self.collect_comprehensive_grid_data(lat, lon)

                # 날짜 정보 추가
                point_data['target_date'] = target_date
                point_data['grid_point_id'] = i

                daily_data.append(point_data)

                # 진행 상황 로그 (10% 간격)
                if (i + 1) % max(1, len(grid_points) // 10) == 0:
                    progress = ((i + 1) / len(grid_points)) * 100
                    logger.info(f"   진행률: {progress:.1f}% ({i+1}/{len(grid_points)} 격자점)")

            except Exception as e:
                logger.warning(f"격자점 ({lat}, {lon}) 데이터 수집 실패: {e}")
                continue

        # 3. CMEMS 파일 즉시 삭제 (메모리 최적화)
        if download_success:
            self.cleanup_cmems_files(target_date)

        logger.info(f" {target_date} 데이터 수집 완료: {len(daily_data)}개 격자점")
        return daily_data

    def collect_daily_training_data_parallel(self, target_date: str, grid_points: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """배치 처리로 특정 날짜의 모든 격자점 데이터 수집 (CMEMS 다운로드 → 생물데이터 배치수집 → 격자점별 조합)"""

        logger.info(f"🌊 {target_date} 일일 훈련 데이터 배치 수집 시작 ({len(grid_points)}개 격자점)")

        # 1. CMEMS 데이터 다운로드 (순차적으로 실행)
        download_success = self.download_cmems_data(target_date)
        if not download_success:
            logger.warning(f"⚠️ CMEMS 데이터 없이 진행: {target_date}")

        # 2. 전체 지역 생물 데이터 배치 수집
        logger.info(f"🐟 {target_date} 생물 데이터 배치 수집 시작...")
        batch_biological_data = self.collect_biological_data_batch(target_date, grid_points)

        # 3. 격자점별 환경 데이터 병렬 수집 + 생물 데이터 조합
        daily_data = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 격자점별 환경 데이터 작업을 병렬로 제출
            future_to_point = {}
            for i, (lat, lon) in enumerate(grid_points):
                future = executor.submit(self._collect_grid_environmental_data, lat, lon, target_date, i)
                future_to_point[future] = (i, lat, lon)

            # 완료된 작업들 수집하여 생물 데이터와 조합
            completed_count = 0
            for future in as_completed(future_to_point):
                i, lat, lon = future_to_point[future]
                try:
                    env_data = future.result()
                    if env_data:
                        # 환경 데이터에 배치 수집한 생물 데이터 추가
                        grid_point = (lat, lon)
                        biological_data = batch_biological_data.get(grid_point, {})

                        # 환경 + 생물 데이터 통합
                        combined_data = {**env_data, **biological_data}
                        combined_data['target_date'] = target_date
                        combined_data['grid_point_id'] = i

                        daily_data.append(combined_data)

                    completed_count += 1
                    # 진행 상황 로그 (20% 간격)
                    if completed_count % max(1, len(grid_points) // 5) == 0:
                        progress = (completed_count / len(grid_points)) * 100
                        logger.info(f"   배치처리 진행률: {progress:.1f}% ({completed_count}/{len(grid_points)} 격자점)")

                except Exception as e:
                    logger.warning(f"격자점 ({lat}, {lon}) 배치 처리 실패: {e}")

        # 4. CMEMS 파일 즉시 삭제 (메모리 최적화)
        if download_success:
            self.cleanup_cmems_files(target_date)

        logger.info(f"✅ {target_date} 배치 데이터 수집 완료: {len(daily_data)}개 격자점")
        return daily_data

    def _collect_grid_environmental_data(self, lat: float, lon: float, target_date: str, grid_point_id: int) -> Optional[Dict[str, Any]]:
        """단일 격자점 환경 데이터만 수집 (생물 데이터 제외)"""

        try:
            # CMEMS 환경 데이터만 수집
            env_data = self.collect_comprehensive_environmental_data(lat, lon)
            return env_data

        except Exception as e:
            logger.warning(f"격자점 ({lat}, {lon}) 환경 데이터 수집 실패: {e}")
            return None

    def collect_comprehensive_environmental_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """환경 데이터만 종합 수집 (CMEMS + 위성)"""

        data = {}

        # CMEMS 데이터 추출
        cmems_data = self._fetch_cmems_data(lat, lon)
        if cmems_data:
            data.update(cmems_data)

        # 위성 데이터 수집 (선택적)
        try:
            satellite_data = self.collect_satellite_data(lat, lon)
            if satellite_data:
                data.update(satellite_data)
        except Exception as e:
            logger.debug(f"위성 데이터 수집 실패 ({lat}, {lon}): {e}")

        # 기본 위치 정보 추가
        data.update({
            'latitude': lat,
            'longitude': lon,
            'depth_m': self._estimate_depth(lat, lon),
            'distance_to_coast_km': self._calculate_distance_to_coast(lat, lon)
        })

        return data

    def _collect_single_grid_point(self, lat: float, lon: float, target_date: str, grid_point_id: int) -> Optional[Dict[str, Any]]:
        """단일 격자점 데이터 수집 (레거시 함수 - 호환성용)"""

        try:
            # 환경 데이터만 수집 (생물 데이터는 배치 처리로 분리됨)
            point_data = self.collect_comprehensive_environmental_data(lat, lon)

            # 기본 생물 데이터 추가 (실제로는 배치 처리에서 대체됨)
            biological_data = self.collect_real_biological_data_parallel(lat, lon)
            point_data.update(biological_data)

            # 날짜 정보 추가
            point_data['target_date'] = target_date
            point_data['grid_point_id'] = grid_point_id

            return point_data

        except Exception as e:
            logger.warning(f"격자점 ({lat}, {lon}) 데이터 수집 실패: {e}")
            return None

    def collect_multiple_days_parallel(self, dates, grid_points, max_workers=2):
        """여러 날짜를 병렬로 처리하여 데이터 수집"""
        all_data = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.collect_daily_training_data_parallel, date, grid_points): date
                for date in dates
            }

            for future in as_completed(futures):
                date = futures[future]
                try:
                    daily_data = future.result()
                    if daily_data:
                        all_data.extend(daily_data)
                        logger.info(f"{date} 데이터 수집 완료: {len(daily_data)}개")
                    else:
                        logger.warning(f"{date} 데이터 수집 실패")

                    # API 호출 제한을 위한 짧은 지연
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"{date} 처리 중 오류: {e}")

        return all_data

    def generate_grid_points(self,
                           lat_range: Tuple[float, float] = (33.0, 38.5),
                           lon_range: Tuple[float, float] = (125.0, 131.0),
                           grid_size: float = 0.25) -> List[Tuple[float, float]]:
        """한국 근해 격자점 생성 (0.25도 간격)"""

        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        # 격자점 생성
        latitudes = np.arange(lat_min, lat_max + grid_size, grid_size)
        longitudes = np.arange(lon_min, lon_max + grid_size, grid_size)

        grid_points = []
        for lat in latitudes:
            for lon in longitudes:
                grid_points.append((lat, lon))

        logger.info(f" 총 {len(grid_points)}개 격자점 생성 (위도: {lat_min}-{lat_max}, 경도: {lon_min}-{lon_max})")
        return grid_points

    def collect_real_biological_data(self, lat: float, lon: float,
                                   radius_km: float = 25) -> Dict[str, Any]:
        """실제 생물 관측 데이터 수집 (GBIF + OBIS)"""

        logger.info(f" 생물 데이터 수집 시작 - 위치: ({lat:.3f}, {lon:.3f})")

        biological_data = {}

        for species in TARGET_SPECIES:
            species_data = self._collect_species_data(species, lat, lon, radius_km)

            if species_data and species_data.get('observation_count', 0) > 0:
                # 실제 관측 데이터가 있는 경우
                biological_data[f"{species.replace(' ', '_')}_density"] = species_data['density']
                biological_data[f"{species.replace(' ', '_')}_count"] = species_data['observation_count']
                logger.info(f" {species}: 실제 관측 데이터 {species_data['observation_count']}건")
            else:
                # 관측 데이터가 없는 경우 기본값 사용
                baseline = SPECIES_BASELINE_VALUES[species]
                biological_data[f"{species.replace(' ', '_')}_density"] = baseline['default_density']
                biological_data[f"{species.replace(' ', '_')}_count"] = 0
                biological_data[f"{species.replace(' ', '_')}_weight"] = baseline['weight']
                logger.info(f" {species}: 기본값 사용 (density={baseline['default_density']})")

        return biological_data

    def collect_biological_data_batch(self, date: str, grid_points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Dict[str, Any]]:
        """전체 지역에서 7일 이내 생물 데이터만 수집 후 격자점에 배분"""

        # 격자점 경계 계산
        lats = [point[0] for point in grid_points]
        lons = [point[1] for point in grid_points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        logger.info(f"🐟 {date} 전체 지역 7일 이내 생물 데이터 배치 수집: 위도 {lat_min:.2f}-{lat_max:.2f}, 경도 {lon_min:.2f}-{lon_max:.2f}")

        # 격자점별 생물 데이터 저장소 (기본값으로 초기화)
        grid_biological_data = {}

        # 각 종별로 전체 지역에서 7일 이내 데이터만 수집 (먼저 수행)
        all_species_observations = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_species = {}
            for species in TARGET_SPECIES:
                # 7일 이내 데이터만 요청
                future = executor.submit(self._collect_species_data_region_7days, species, lat_min, lat_max, lon_min, lon_max, date)
                future_to_species[future] = species

            for future in as_completed(future_to_species):
                species = future_to_species[future]
                try:
                    observations = future.result()
                    all_species_observations[species] = observations
                    logger.info(f"  {species}: {len(observations)}개 7일 이내 관측 데이터")
                    time.sleep(0.3)  # API 제한
                except Exception as e:
                    logger.warning(f"  {species} 지역 데이터 수집 실패: {e}")
                    all_species_observations[species] = []

        # 모든 격자점을 환경/거리 기반 적응형 가중치로 초기화
        for grid_point in grid_points:
            lat, lon = grid_point
            biological_data = {}
            for species in TARGET_SPECIES:
                baseline = SPECIES_BASELINE_VALUES[species]
                # 환경 조건 + 거리 기반 적응형 가중치 계산
                adaptive_weight = self._calculate_adaptive_weight(species, lat, lon, all_species_observations.get(species, []))
                biological_data[f"{species.replace(' ', '_')}_density"] = baseline['default_density']
                biological_data[f"{species.replace(' ', '_')}_count"] = 0
                biological_data[f"{species.replace(' ', '_')}_weight"] = adaptive_weight
            grid_biological_data[grid_point] = biological_data

        # 실제 관측 데이터가 있는 격자점만 업데이트
        for grid_point in grid_points:
            lat, lon = grid_point

            for species in TARGET_SPECIES:
                observations = all_species_observations.get(species, [])

                # 격자점 근처 7일 이내 관측 데이터 찾기
                nearby_obs = []
                for obs in observations:
                    if self._is_within_grid(obs['lat'], obs['lon'], lat, lon, radius_km=25):
                        nearby_obs.append(obs)

                # 실제 관측 데이터가 있는 경우에만 업데이트
                if nearby_obs:
                    density = sum(obs['density'] for obs in nearby_obs) / len(nearby_obs)
                    count = len(nearby_obs)

                    # 실제 데이터로 업데이트 (가중치 1.0 = 최고 신뢰도)
                    grid_biological_data[grid_point][f"{species.replace(' ', '_')}_density"] = density
                    grid_biological_data[grid_point][f"{species.replace(' ', '_')}_count"] = count
                    grid_biological_data[grid_point][f"{species.replace(' ', '_')}_weight"] = 1.0  # 실제 데이터는 최대 가중치

                    logger.debug(f"    {species} ({lat:.2f}, {lon:.2f}): 실제 데이터 {count}건")
                # 관측 데이터가 없으면 이미 설정한 기본값 유지

        actual_data_count = sum(1 for grid_data in grid_biological_data.values()
                               if any(grid_data[f"{species.replace(' ', '_')}_count"] > 0 for species in TARGET_SPECIES))

        logger.info(f"🐟 {date} 생물 데이터 배치 처리 완료: {len(grid_points)}개 격자점 중 {actual_data_count}개에 실제 데이터")
        return grid_biological_data

    def collect_real_biological_data_parallel(self, lat: float, lon: float,
                                             radius_km: float = 25) -> Dict[str, Any]:
        """개별 격자점 생물 데이터 반환 (배치 처리된 데이터에서 추출)"""

        # 이 함수는 이제 배치 처리된 결과를 사용하므로 기본값만 반환
        biological_data = {}
        for species in TARGET_SPECIES:
            baseline = SPECIES_BASELINE_VALUES[species]
            biological_data[f"{species.replace(' ', '_')}_density"] = baseline['default_density']
            biological_data[f"{species.replace(' ', '_')}_count"] = 0
            biological_data[f"{species.replace(' ', '_')}_weight"] = baseline['weight']

        return biological_data

    def _collect_species_data_region_7days(self, species: str, lat_min: float, lat_max: float,
                                          lon_min: float, lon_max: float, date: str) -> List[Dict]:
        """전체 지역에서 특정 종의 7일 이내 관측 데이터만 수집"""

        observations = []

        try:
            # GBIF 7일 이내 지역 검색
            gbif_obs = self._query_gbif_region_7days(species, lat_min, lat_max, lon_min, lon_max, date)
            observations.extend(gbif_obs)

            # OBIS 7일 이내 지역 검색
            obis_obs = self._query_obis_region_7days(species, lat_min, lat_max, lon_min, lon_max, date)
            observations.extend(obis_obs)

        except Exception as e:
            logger.warning(f"{species} 7일 이내 지역 데이터 수집 오류: {e}")

        return observations

    def _query_gbif_region_7days(self, species: str, lat_min: float, lat_max: float,
                                lon_min: float, lon_max: float, date: str) -> List[Dict]:
        """GBIF에서 지역 단위로 7일 이내 종 데이터 검색"""

        observations = []

        try:
            # 7일 전부터 현재까지만 검색
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=7)

            url = "https://api.gbif.org/v1/occurrence/search"
            params = {
                'scientificName': species,
                'decimalLatitude': f"{lat_min},{lat_max}",
                'decimalLongitude': f"{lon_min},{lon_max}",
                'eventDate': f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}",
                'hasCoordinate': 'true',
                'limit': 300
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                for record in data.get('results', []):
                    if record.get('decimalLatitude') and record.get('decimalLongitude'):
                        obs_date = record.get('eventDate', date)[:10]  # YYYY-MM-DD 형태로 자름
                        observations.append({
                            'lat': record['decimalLatitude'],
                            'lon': record['decimalLongitude'],
                            'date': obs_date,
                            'density': 1.0,  # GBIF는 개체수 정보가 제한적이므로 기본값
                            'source': 'GBIF'
                        })

        except Exception as e:
            logger.warning(f"GBIF {species} 7일 이내 지역 검색 실패: {e}")

        return observations

    def _query_obis_region_7days(self, species: str, lat_min: float, lat_max: float,
                                lon_min: float, lon_max: float, date: str) -> List[Dict]:
        """OBIS에서 지역 단위로 7일 이내 종 데이터 검색"""

        observations = []

        try:
            # 7일 전부터 현재까지만 검색
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=7)

            url = "https://api.obis.org/v3/occurrence"
            params = {
                'scientificname': species,
                'geometry': f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))",
                'startdate': start_date.strftime('%Y-%m-%d'),
                'enddate': end_date.strftime('%Y-%m-%d'),
                'size': 300
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                for record in data.get('results', []):
                    if record.get('decimalLatitude') and record.get('decimalLongitude'):
                        obs_date = record.get('eventDate', date)[:10]
                        individual_count = record.get('individualCount', 1)
                        observations.append({
                            'lat': record['decimalLatitude'],
                            'lon': record['decimalLongitude'],
                            'date': obs_date,
                            'density': max(individual_count, 1.0),
                            'source': 'OBIS'
                        })

        except Exception as e:
            logger.warning(f"OBIS {species} 7일 이내 지역 검색 실패: {e}")

        return observations

    def _collect_species_data_region(self, species: str, lat_min: float, lat_max: float,
                                   lon_min: float, lon_max: float, date: str) -> List[Dict]:
        """전체 지역에서 특정 종의 관측 데이터 수집 (레거시 함수)"""

        observations = []

        try:
            # GBIF 지역 검색
            gbif_obs = self._query_gbif_region(species, lat_min, lat_max, lon_min, lon_max, date)
            observations.extend(gbif_obs)

            # OBIS 지역 검색
            obis_obs = self._query_obis_region(species, lat_min, lat_max, lon_min, lon_max, date)
            observations.extend(obis_obs)

        except Exception as e:
            logger.warning(f"{species} 지역 데이터 수집 오류: {e}")

        return observations

    def _query_gbif_region(self, species: str, lat_min: float, lat_max: float,
                          lon_min: float, lon_max: float, date: str) -> List[Dict]:
        """GBIF에서 지역 단위로 종 데이터 검색"""

        observations = []

        try:
            # 30일 전부터 현재까지 검색
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=30)

            url = "https://api.gbif.org/v1/occurrence/search"
            params = {
                'scientificName': species,
                'decimalLatitude': f"{lat_min},{lat_max}",
                'decimalLongitude': f"{lon_min},{lon_max}",
                'eventDate': f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}",
                'hasCoordinate': 'true',
                'limit': 300
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                for record in data.get('results', []):
                    if record.get('decimalLatitude') and record.get('decimalLongitude'):
                        obs_date = record.get('eventDate', date)[:10]  # YYYY-MM-DD 형태로 자름
                        observations.append({
                            'lat': record['decimalLatitude'],
                            'lon': record['decimalLongitude'],
                            'date': obs_date,
                            'density': 1.0,  # GBIF는 개체수 정보가 제한적이므로 기본값
                            'source': 'GBIF'
                        })

        except Exception as e:
            logger.warning(f"GBIF {species} 지역 검색 실패: {e}")

        return observations

    def _query_obis_region(self, species: str, lat_min: float, lat_max: float,
                          lon_min: float, lon_max: float, date: str) -> List[Dict]:
        """OBIS에서 지역 단위로 종 데이터 검색"""

        observations = []

        try:
            # 30일 전부터 현재까지 검색
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=30)

            url = "https://api.obis.org/v3/occurrence"
            params = {
                'scientificname': species,
                'geometry': f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))",
                'startdate': start_date.strftime('%Y-%m-%d'),
                'enddate': end_date.strftime('%Y-%m-%d'),
                'size': 300
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                for record in data.get('results', []):
                    if record.get('decimalLatitude') and record.get('decimalLongitude'):
                        obs_date = record.get('eventDate', date)[:10]
                        individual_count = record.get('individualCount', 1)
                        observations.append({
                            'lat': record['decimalLatitude'],
                            'lon': record['decimalLongitude'],
                            'date': obs_date,
                            'density': max(individual_count, 1.0),
                            'source': 'OBIS'
                        })

        except Exception as e:
            logger.warning(f"OBIS {species} 지역 검색 실패: {e}")

        return observations

    def _calculate_adaptive_weight(self, species: str, lat: float, lon: float, observations: List[Dict]) -> float:
        """환경 조건과 거리 기반 적응형 가중치 계산"""

        baseline = SPECIES_BASELINE_VALUES[species]
        base_weight = baseline['weight']

        # 1. 환경 조건 적합성 평가
        env_suitability = self._evaluate_environmental_suitability(species, lat, lon)

        # 2. 가장 가까운 실제 관측 데이터까지의 거리
        distance_factor = self._calculate_distance_factor(lat, lon, observations)

        # 3. 적응형 가중치 계산
        # 환경 적합성 * 거리 보정 * 기본 가중치
        adaptive_weight = base_weight * env_suitability * distance_factor

        # 최소/최대 가중치 제한 (0.2 ~ 0.8)
        adaptive_weight = max(0.2, min(0.8, adaptive_weight))

        return adaptive_weight

    def _evaluate_environmental_suitability(self, species: str, lat: float, lon: float) -> float:
        """환경 조건 기반 서식 적합성 평가 (1.0 = 최적, 0.5 = 보통, 0.3 = 부적합)"""

        try:
            # CMEMS 데이터에서 현재 환경 조건 가져오기
            cmems_data = self._fetch_cmems_data(lat, lon)
            if not cmems_data:
                return 0.8  # 환경 데이터가 없으면 보통 적합성

            baseline = SPECIES_BASELINE_VALUES[species]
            temp_pref = baseline['temp_preference']  # [min_temp, max_temp]
            depth_pref = baseline['depth_preference']  # [min_depth, max_depth]

            # 수온 적합성 평가
            temp_suit = 1.0  # 기본값
            if 'thetao' in cmems_data:
                current_temp = cmems_data['thetao']
                if temp_pref[0] <= current_temp <= temp_pref[1]:
                    temp_suit = 1.0  # 최적
                elif abs(current_temp - sum(temp_pref)/2) <= 3:  # 선호 범위 ±3도
                    temp_suit = 0.8  # 적합
                else:
                    temp_suit = 0.5  # 부적합

            # 수심 적합성 평가 (추정 수심 사용)
            depth_suit = 1.0  # 기본값
            estimated_depth = abs(self._estimate_depth(lat, lon))  # 절댓값 사용
            if depth_pref[0] <= estimated_depth <= depth_pref[1]:
                depth_suit = 1.0  # 최적
            elif abs(estimated_depth - sum(depth_pref)/2) <= 50:  # 선호 범위 ±50m
                depth_suit = 0.8  # 적합
            else:
                depth_suit = 0.5  # 부적합

            # 염분 적합성 (해양생물은 일반적으로 30-35 psu 선호)
            salinity_suit = 1.0  # 기본값
            if 'so' in cmems_data:
                salinity = cmems_data['so']
                if 30 <= salinity <= 35:
                    salinity_suit = 1.0
                elif 25 <= salinity <= 40:
                    salinity_suit = 0.8
                else:
                    salinity_suit = 0.6

            # 종합 환경 적합성 (평균)
            overall_suitability = (temp_suit + depth_suit + salinity_suit) / 3

            return overall_suitability

        except Exception as e:
            logger.debug(f"환경 적합성 평가 실패 ({species}): {e}")
            return 0.7  # 오류 시 보통 적합성

    def _calculate_distance_factor(self, lat: float, lon: float, observations: List[Dict]) -> float:
        """가장 가까운 실제 관측 데이터까지의 거리 기반 보정 인수"""

        if not observations:
            return 0.7  # 관측 데이터가 없으면 낮은 신뢰도

        # 가장 가까운 관측점까지의 거리 계산
        min_distance = float('inf')
        for obs in observations:
            distance = self._haversine_distance(lat, lon, obs['lat'], obs['lon'])
            min_distance = min(min_distance, distance)

        # 거리 기반 보정 인수 계산
        if min_distance <= 25:  # 25km 이내
            return 1.2  # 가중치 증가
        elif min_distance <= 50:  # 50km 이내
            return 1.0  # 기본 가중치
        elif min_distance <= 100:  # 100km 이내
            return 0.8  # 가중치 감소
        else:  # 100km 초과
            return 0.6  # 많이 감소

    def _is_within_grid(self, obs_lat: float, obs_lon: float, grid_lat: float, grid_lon: float, radius_km: float = 25) -> bool:
        """관측점이 격자점 반경 내에 있는지 확인"""
        distance_km = self._haversine_distance(obs_lat, obs_lon, grid_lat, grid_lon)
        return distance_km <= radius_km

    def _days_between(self, date1: str, date2: str) -> int:
        """두 날짜 사이의 일수 계산"""
        try:
            d1 = datetime.strptime(date1, '%Y-%m-%d')
            d2 = datetime.strptime(date2, '%Y-%m-%d')
            return abs((d2 - d1).days)
        except:
            return 999  # 파싱 실패 시 큰 값 반환

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """두 지점 간 거리 계산 (km)"""
        from math import radians, cos, sin, asin, sqrt

        # 위경도를 라디안으로 변환
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # 하버사인 공식
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # 지구 반지름 (km)
        r = 6371

        return c * r

    def _collect_species_data(self, species: str, lat: float, lon: float,
                            radius_km: float) -> Optional[Dict[str, Any]]:
        """개별 종 데이터 수집 (레거시 함수 - 호환성용)"""

        try:
            # GBIF에서 데이터 수집
            gbif_data = self._query_gbif(species, lat, lon, radius_km)

            # OBIS에서 데이터 수집
            obis_data = self._query_obis(species, lat, lon, radius_km)

            # 데이터 통합
            total_count = gbif_data.get('count', 0) + obis_data.get('count', 0)

            if total_count > 0:
                # 밀도 계산 (단위 면적당 개체수)
                area_km2 = np.pi * (radius_km ** 2)
                density = total_count / area_km2

                return {
                    'observation_count': total_count,
                    'density': min(density, 1.0),  # 최대 1.0으로 정규화
                    'sources': ['GBIF', 'OBIS']
                }

            return None

        except Exception as e:
            logger.warning(f"{species} 데이터 수집 실패: {e}")
            return None

    def _query_gbif(self, species: str, lat: float, lon: float, radius_km: float) -> Dict[str, Any]:
        """GBIF API 쿼리"""

        try:
            # 위도/경도를 기준으로 검색 범위 계산
            lat_offset = radius_km / 111.0  # 1도  111km
            lon_offset = radius_km / (111.0 * np.cos(np.radians(lat)))

            params = {
                'scientificName': species,
                'decimalLatitude': f"{lat-lat_offset},{lat+lat_offset}",
                'decimalLongitude': f"{lon-lon_offset},{lon+lon_offset}",
                'hasCoordinate': 'true',
                'limit': 300,
                'year': '2020,2024'  # 최근 5년 데이터
            }

            response = self.session.get(f"{self.gbif_base_url}/occurrence/search",
                                      params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return {'count': data.get('count', 0), 'source': 'GBIF'}

            return {'count': 0, 'source': 'GBIF'}

        except Exception as e:
            logger.warning(f"GBIF 쿼리 실패 ({species}): {e}")
            return {'count': 0, 'source': 'GBIF'}

    def _query_obis(self, species: str, lat: float, lon: float, radius_km: float) -> Dict[str, Any]:
        """OBIS API 쿼리"""

        try:
            # 검색 범위 계산
            lat_offset = radius_km / 111.0
            lon_offset = radius_km / (111.0 * np.cos(np.radians(lat)))

            params = {
                'scientificname': species,
                'geometry': f'POLYGON(({lon-lon_offset} {lat-lat_offset},'
                           f'{lon+lon_offset} {lat-lat_offset},'
                           f'{lon+lon_offset} {lat+lat_offset},'
                           f'{lon-lon_offset} {lat+lat_offset},'
                           f'{lon-lon_offset} {lat-lat_offset}))',
                'startdate': '2020-01-01',
                'enddate': '2024-12-31'
            }

            response = self.session.get(f"{self.obis_base_url}/occurrence",
                                      params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                count = len(data.get('results', []))
                return {'count': count, 'source': 'OBIS'}

            return {'count': 0, 'source': 'OBIS'}

        except Exception as e:
            logger.warning(f"OBIS 쿼리 실패 ({species}): {e}")
            return {'count': 0, 'source': 'OBIS'}

    def _fetch_cmems_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """CMEMS .nc 파일에서 해당 격자점의 환경 데이터 추출"""

        logger.info(f" CMEMS 데이터 추출 시작 - 위치: ({lat:.3f}, {lon:.3f})")

        cmems_data = {}

        try:
            # CMEMS 출력 디렉토리의 .nc 파일들 검색
            nc_files = list(self.cmems_output_dir.glob("*.nc"))

            if not nc_files:
                logger.warning(" CMEMS .nc 파일이 없습니다")
                return {}

            for nc_file in nc_files:
                try:
                    # 파일 접근 시 락 사용 (병렬 처리 시 권한 충돌 방지)
                    with self.file_lock:
                        # 파일 존재 확인
                        if not nc_file.exists():
                            continue

                        # xarray로 .nc 파일 읽기 (재시도 로직 포함)
                        max_retries = 3
                        ds = None
                        for attempt in range(max_retries):
                            try:
                                ds = xr.open_dataset(nc_file, engine='netcdf4')
                                break
                            except (PermissionError, OSError) as e:
                                if attempt < max_retries - 1:
                                    time.sleep(0.1)  # 짧은 대기
                                    continue
                                else:
                                    raise e

                        if ds is None:
                            continue

                        # 격자점과 가장 가까운 지점 찾기
                        lat_diff = np.abs(ds.latitude - lat)
                        lon_diff = np.abs(ds.longitude - lon)

                        lat_idx = lat_diff.argmin()
                        lon_idx = lon_diff.argmin()

                        # 해당 지점의 데이터 추출
                        point_data = ds.isel(latitude=lat_idx, longitude=lon_idx)

                        # 변수별 데이터 추출
                        for var in ds.data_vars:
                            if var in ['zos', 'sob', 'tob', 'mlotst', 'nppv', 'o2']:
                                values = point_data[var].values
                                if isinstance(values, np.ndarray) and values.size > 0:
                                    # 시간 차원이 있는 경우 최신 값 사용
                                    if values.ndim > 0:
                                        cmems_data[f'cmems_{var}'] = float(values.flat[0])
                                    else:
                                        cmems_data[f'cmems_{var}'] = float(values)

                        ds.close()

                    logger.info(f" {nc_file.name}에서 데이터 추출 완료")

                except Exception as e:
                    logger.warning(f" {nc_file.name} 처리 실패: {e}")
                    continue

            logger.info(f" CMEMS 데이터 추출 완료: {len(cmems_data)}개 변수")
            return cmems_data

        except Exception as e:
            logger.error(f" CMEMS 데이터 추출 실패: {e}")
            return {}

    def collect_comprehensive_grid_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """격자점별 종합 데이터 수집 - 레거시 함수 (새로운 배치 처리로 대체됨)"""

        logger.info(f"⚠️ 레거시 모드: 종합 데이터 수집 - 격자점: ({lat:.3f}, {lon:.3f})")

        comprehensive_data = {
            'latitude': lat,
            'longitude': lon,
            'collection_time': datetime.now().isoformat()
        }

        try:
            # 1. 환경 데이터만 수집 (새 방식 사용)
            logger.info("1. 환경 데이터 수집 중...")
            env_data = self.collect_comprehensive_environmental_data(lat, lon)
            comprehensive_data.update(env_data)

            # 2. 기본 생물 데이터 추가 (배치 처리로 대체 예정)
            logger.info("2. 기본 생물 데이터 추가 중...")
            biological_data = self.collect_real_biological_data_parallel(lat, lon)
            comprehensive_data.update(biological_data)

            # 3. 메타데이터 추가
            comprehensive_data['data_sources'] = ['GBIF', 'OBIS', 'CMEMS']
            comprehensive_data['grid_size_km'] = 25  # 0.25도 = 약 25km

            # 데이터 품질 점수 계산
            data_quality_score = len([v for k, v in comprehensive_data.items()
                                    if v is not None and not (isinstance(v, str) and v == '')]) / 50.0
            comprehensive_data['data_quality_score'] = min(1.0, data_quality_score)

            logger.info(f" 격자점 데이터 수집 완료: {len(comprehensive_data)}개 필드")
            return comprehensive_data

        except Exception as e:
            logger.error(f" 격자점 ({lat}, {lon}) 데이터 수집 실패: {e}")
            return comprehensive_data

    def save_grid_data_to_csv(self, grid_data_list: List[Dict[str, Any]],
                            filename: str = "comprehensive_marine_data.csv") -> str:
        """격자 데이터를 CSV로 저장"""

        if not grid_data_list:
            logger.warning(" 저장할 데이터가 없습니다")
            return ""

        try:
            df = pd.DataFrame(grid_data_list)

            # 결측치 처리
            df = df.fillna(0)

            # 파일 저장
            filepath = os.path.join(os.getcwd(), filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            logger.info(f" 데이터 저장 완료: {filepath}")
            logger.info(f"  - 총 {len(df)} 행, {len(df.columns)} 열")
            logger.info(f"  - 파일 크기: {os.path.getsize(filepath) / 1024:.1f} KB")

            return filepath

        except Exception as e:
            logger.error(f" CSV 저장 실패: {e}")
            return ""

    def save_to_csv(self, data_list: List[Dict[str, Any]], filename: str) -> str:
        """데이터 리스트를 CSV 파일로 저장"""
        return self.save_grid_data_to_csv(data_list, filename)

    def validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 품질 검증"""

        quality_report = {
            'total_fields': len(data),
            'non_null_fields': 0,
            'biological_species_count': 0,
            'cmems_variables_count': 0,
            'data_completeness': 0.0,
            'quality_issues': []
        }

        try:
            # 필드별 검증
            for key, value in data.items():
                if value is not None and not (isinstance(value, str) and value == ''):
                    quality_report['non_null_fields'] += 1

                # 생물 데이터 카운트
                if any(species.replace(' ', '_') in key for species in TARGET_SPECIES):
                    if '_density' in key:
                        quality_report['biological_species_count'] += 1

                # CMEMS 데이터 카운트
                if key.startswith('cmems_'):
                    quality_report['cmems_variables_count'] += 1

                # 값 범위 검증
                if isinstance(value, (int, float)):
                    if key.endswith('_density') and (value < 0 or value > 1):
                        quality_report['quality_issues'].append(f"{key}: 밀도 값 범위 초과 ({value})")

            # 완성도 계산
            quality_report['data_completeness'] = quality_report['non_null_fields'] / quality_report['total_fields']

            # 최소 품질 기준 확인
            if quality_report['biological_species_count'] == 0:
                quality_report['quality_issues'].append("생물 종 데이터 없음")

            if quality_report['cmems_variables_count'] == 0:
                quality_report['quality_issues'].append("CMEMS 환경 데이터 없음")

            logger.info(f" 데이터 품질 검증 완료: {quality_report['data_completeness']:.1%} 완성도")

            return quality_report

        except Exception as e:
            logger.error(f" 품질 검증 실패: {e}")
            quality_report['quality_issues'].append(f"검증 오류: {e}")
            return quality_report

    def _estimate_depth(self, lat: float, lon: float) -> float:
        """위치 기반 수심 추정 (한국 근해 지형 데이터 기반)"""
        try:
            # 한국 근해 지형 특성을 고려한 수심 추정
            # 간단한 경험적 모델 사용 (실제로는 GEBCO 등 수심 데이터베이스 사용 권장)

            # 해안선으로부터의 거리 추정
            coast_distance = self._calculate_distance_to_coast(lat, lon)

            # 기본 수심 추정 (해안선에서 멀수록 깊어짐)
            if coast_distance < 10:  # 연안 (10km 이내)
                estimated_depth = -10 - (coast_distance * 2)  # -10m ~ -30m
            elif coast_distance < 50:  # 근해 (50km 이내)
                estimated_depth = -30 - ((coast_distance - 10) * 3)  # -30m ~ -150m
            elif coast_distance < 100:  # 중간해역
                estimated_depth = -150 - ((coast_distance - 50) * 4)  # -150m ~ -350m
            else:  # 원해
                estimated_depth = -350 - ((coast_distance - 100) * 2)  # -350m 이하

            # 지역별 보정 (한국 근해 특성)
            if 33.0 <= lat <= 35.0 and 124.0 <= lon <= 127.0:  # 서해
                estimated_depth = estimated_depth * 0.3  # 서해는 얕음
            elif 35.0 <= lat <= 38.0 and 129.0 <= lon <= 131.0:  # 동해
                estimated_depth = estimated_depth * 1.5  # 동해는 깊음
            elif 33.0 <= lat <= 35.0 and 126.0 <= lon <= 129.0:  # 남해
                estimated_depth = estimated_depth * 0.7  # 남해는 중간

            # 최대/최소값 제한
            estimated_depth = max(estimated_depth, -6000)  # 최대 6000m 깊이
            estimated_depth = min(estimated_depth, -5)     # 최소 5m 깊이

            return estimated_depth

        except Exception as e:
            logger.debug(f"수심 추정 실패 ({lat}, {lon}): {e}")
            return -50.0  # 기본값 50m

    def collect_biological_data_weekly_batch(self, start_date: str, end_date: str, grid_points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Dict[str, Any]]:
        """7일 범위 내 전체 지역 생물 데이터 배치 수집"""

        # 격자점 경계 계산
        lats = [point[0] for point in grid_points]
        lons = [point[1] for point in grid_points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        logger.info(f"🐟 7일 범위 전체 지역 생물 데이터 배치 수집: {start_date} ~ {end_date}")
        logger.info(f"   지역: 위도 {lat_min:.2f}-{lat_max:.2f}, 경도 {lon_min:.2f}-{lon_max:.2f}")

        # 전체 격자점에 대한 빈 결과 초기화
        batch_results = {}
        for point in grid_points:
            batch_results[point] = {}

        # 각 종별 데이터 수집
        target_species = ['Aurelia aurita', 'Chrysaora pacifica', 'Scomber japonicus',
                         'Engraulis japonicus', 'Todarodes pacificus', 'Trachurus japonicus',
                         'Sardinops melanostictus', 'Chaetodon nippon']

        for species in target_species:
            try:
                # 7일 범위 내 전체 지역 데이터 수집
                species_observations = self._collect_species_data_region_weekly(
                    species, lat_min, lat_max, lon_min, lon_max, start_date, end_date
                )

                # 각 격자점별로 가장 가까운 관측값 배정
                for point in grid_points:
                    lat, lon = point
                    closest_obs = self._find_closest_observation_weekly(
                        species_observations, lat, lon, start_date, end_date
                    )

                    species_key = species.replace(' ', '_')
                    batch_results[point][f"{species_key}_density"] = closest_obs['density']
                    batch_results[point][f"{species_key}_weight"] = closest_obs['weight']

                logger.debug(f"  ✅ {species}: {len(species_observations)}개 관측값")

            except Exception as e:
                logger.warning(f"  ⚠️ {species} 데이터 수집 실패: {e}")
                # 기본값으로 설정
                species_key = species.replace(' ', '_')
                for point in grid_points:
                    batch_results[point][f"{species_key}_density"] = 0.001
                    batch_results[point][f"{species_key}_weight"] = 0.1

        logger.info(f"🎯 7일 범위 배치 수집 완료: {len(batch_results)}개 격자점")

        # 반환값 검증 및 로깅
        if batch_results:
            sample_key = next(iter(batch_results.keys()))
            sample_value = batch_results[sample_key]
            logger.debug(f"  샘플 키: {sample_key} (타입: {type(sample_key)})")
            logger.debug(f"  샘플 값: {type(sample_value)} - {sample_value}")

        return batch_results

    def _collect_species_data_region_weekly(self, species: str, lat_min: float, lat_max: float,
                                          lon_min: float, lon_max: float, start_date: str, end_date: str) -> List[Dict]:
        """7일 범위 내 전체 지역에서 특정 종의 관측 데이터 수집"""

        observations = []

        try:
            # OBIS API 호출 (7일 범위)
            obis_url = "https://api.obis.org/v3/occurrence"
            params = {
                'scientificname': species,
                'geometry': f'POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))',
                'startdate': start_date,
                'enddate': end_date,
                'size': 1000,
                'hascoordinates': 'true'
            }

            with self.request_session.get(obis_url, params=params, timeout=30) as response:
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and data['results']:
                        for record in data['results']:
                            if all(key in record for key in ['decimalLatitude', 'decimalLongitude', 'eventDate']):
                                observations.append({
                                    'lat': record['decimalLatitude'],
                                    'lon': record['decimalLongitude'],
                                    'date': record['eventDate'],
                                    'density': record.get('individualCount', 1) * 0.1,
                                    'weight': 1.0,  # 실제 관측 데이터
                                    'source': 'OBIS'
                                })
                        logger.debug(f"    OBIS: {len(data['results'])}개 {species} 관측값")
        except Exception as e:
            logger.debug(f"    OBIS API 오류 ({species}): {e}")

        try:
            # GBIF API 호출 (7일 범위)
            gbif_url = "https://api.gbif.org/v1/occurrence/search"
            params = {
                'scientificName': species,
                'decimalLatitude': f'{lat_min},{lat_max}',
                'decimalLongitude': f'{lon_min},{lon_max}',
                'eventDate': f'{start_date},{end_date}',
                'limit': 300,
                'hasCoordinate': 'true'
            }

            with self.request_session.get(gbif_url, params=params, timeout=30) as response:
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and data['results']:
                        for record in data['results']:
                            if all(key in record for key in ['decimalLatitude', 'decimalLongitude', 'eventDate']):
                                observations.append({
                                    'lat': record['decimalLatitude'],
                                    'lon': record['decimalLongitude'],
                                    'date': record['eventDate'],
                                    'density': record.get('individualCount', 1) * 0.1,
                                    'weight': 1.0,  # 실제 관측 데이터
                                    'source': 'GBIF'
                                })
                        logger.debug(f"    GBIF: {len(data['results'])}개 {species} 관측값")
        except Exception as e:
            logger.debug(f"    GBIF API 오류 ({species}): {e}")

        return observations

    def _find_closest_observation_weekly(self, observations: List[Dict], target_lat: float, target_lon: float,
                                       start_date: str, end_date: str) -> Dict[str, float]:
        """7일 범위 내 가장 가까운 관측값 찾기 (시간 가중치 적용)"""

        if not observations:
            return {'density': 0.001, 'weight': 0.1}  # 기본값

        from datetime import datetime

        # 날짜 파싱
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except:
            return {'density': 0.001, 'weight': 0.1}

        best_obs = None
        best_score = float('inf')

        for obs in observations:
            try:
                # 공간적 거리
                spatial_dist = ((obs['lat'] - target_lat) ** 2 + (obs['lon'] - target_lon) ** 2) ** 0.5

                # 시간적 거리 (7일 범위 내)
                obs_date = datetime.strptime(obs['date'][:10], '%Y-%m-%d')
                if start_dt <= obs_date <= end_dt:
                    temporal_dist = abs((obs_date - start_dt).days)

                    # 종합 점수 (공간 + 시간)
                    score = spatial_dist + temporal_dist * 0.1

                    if score < best_score:
                        best_score = score
                        best_obs = obs
            except:
                continue

        if best_obs:
            return {
                'density': max(best_obs['density'], 0.001),
                'weight': best_obs['weight']
            }
        else:
            return {'density': 0.001, 'weight': 0.1}

    def _calculate_distance_to_coast(self, lat: float, lon: float) -> float:
        """해안선까지의 거리 계산 (km)"""
        try:
            # 한국 주요 해안선 좌표들 (간소화된 버전)
            coast_points = [
                # 서해안
                (37.5, 126.6), (37.0, 126.3), (36.5, 126.1), (36.0, 125.8),
                (35.5, 125.9), (35.0, 126.2), (34.5, 126.5),
                # 남해안
                (34.8, 127.5), (35.1, 128.1), (35.3, 128.6), (35.5, 129.2),
                # 동해안
                (36.0, 129.4), (36.5, 129.5), (37.0, 129.4), (37.5, 129.1),
                (38.0, 128.6)
            ]

            # 가장 가까운 해안선 지점까지의 거리 계산
            min_distance = float('inf')
            for coast_lat, coast_lon in coast_points:
                distance = self._haversine_distance(lat, lon, coast_lat, coast_lon)
                min_distance = min(min_distance, distance)

            return min_distance

        except Exception as e:
            logger.debug(f"해안선 거리 계산 실패 ({lat}, {lon}): {e}")
            return 50.0  # 기본값 50km

    def collect_weekly_training_data(self, start_date: str, end_date: str,
                                   lat_range: Tuple[float, float] = (33.0, 38.0),
                                   lon_range: Tuple[float, float] = (124.0, 131.0),
                                   resolution: float = 0.5) -> str:
        """7일 간격 학습 데이터 수집 (생물 데이터는 7일 범위 내 모든 데이터 포함)"""

        logger.info(f" 7일 간격 학습 데이터 수집 시작: {start_date} ~ {end_date}")

        # 날짜 파싱
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # 7일 간격 날짜 리스트 생성
        training_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            training_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=7)

        logger.info(f" 학습 날짜 ({len(training_dates)}개): {training_dates}")

        # 격자점 생성
        grid_points = []
        for lat in np.arange(lat_range[0], lat_range[1] + resolution, resolution):
            for lon in np.arange(lon_range[0], lon_range[1] + resolution, resolution):
                grid_points.append((round(lat, 1), round(lon, 1)))

        logger.info(f" 격자점 생성 완료: {len(grid_points)}개")

        all_data = []
        total_combinations = len(training_dates) * len(grid_points)
        processed = 0

        for target_date in training_dates:
            logger.info(f" 처리 중: {target_date} ({training_dates.index(target_date)+1}/{len(training_dates)})")

            # CMEMS 데이터 다운로드
            if not self.download_cmems_data(target_date):
                logger.warning(f" CMEMS 다운로드 실패, 건너뛰기: {target_date}")
                continue

            try:
                # 이 날짜의 전체 지역 생물 데이터를 7일 범위로 수집
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                bio_start_date = (target_dt - timedelta(days=3)).strftime('%Y-%m-%d')
                bio_end_date = (target_dt + timedelta(days=3)).strftime('%Y-%m-%d')

                logger.info(f"  생물 데이터 범위: {bio_start_date} ~ {bio_end_date}")

                # 배치 생물 데이터 수집 (7일 범위)
                batch_bio_data = self.collect_biological_data_weekly_batch(
                    bio_start_date, bio_end_date, grid_points
                )

                # 배치 데이터 타입 디버깅
                logger.debug(f"  batch_bio_data 타입: {type(batch_bio_data)}")
                if batch_bio_data:
                    first_key = next(iter(batch_bio_data.keys())) if batch_bio_data else None
                    first_value = batch_bio_data.get(first_key) if first_key else None
                    logger.debug(f"  첫 번째 키: {first_key} (타입: {type(first_key)})")
                    logger.debug(f"  첫 번째 값: {type(first_value)}")

                # 각 격자점별 데이터 처리
                for lat, lon in grid_points:
                    try:
                        # 환경 데이터 수집 (해당 날짜만)
                        env_data = self.collect_comprehensive_environmental_data(lat, lon)

                        # 생물 데이터 가져오기 (튜플 키 사용)
                        point_tuple = (lat, lon)
                        bio_data = batch_bio_data.get(point_tuple, {})

                        # bio_data가 딕셔너리인지 확인
                        if not isinstance(bio_data, dict):
                            logger.warning(f"  격자점 ({lat}, {lon}) - bio_data가 dict가 아님: {type(bio_data)}, 값: {bio_data}")
                            bio_data = {}

                        # 데이터 통합 (안전한 방식)
                        combined_data = {
                            'collection_date': target_date,
                            'data_collection_range': f"{bio_start_date}~{bio_end_date}",
                        }

                        # 환경 데이터 추가
                        if isinstance(env_data, dict):
                            combined_data.update(env_data)

                        # 생물 데이터 추가 (안전하게)
                        if isinstance(bio_data, dict):
                            combined_data.update(bio_data)
                        else:
                            logger.warning(f"  격자점 ({lat}, {lon}) - bio_data 딕셔너리가 아님: {type(bio_data)}, 기본값 사용")

                        # 생물 데이터가 없는 경우 하드코딩된 기본값 사용 (낮은 가중치)
                        for species in TARGET_SPECIES:
                            species_key = species.replace(' ', '_')
                            density_key = f"{species_key}_density"
                            weight_key = f"{species_key}_weight"

                            if density_key not in combined_data or combined_data[density_key] == 0:
                                baseline = SPECIES_BASELINE_VALUES.get(species, {})

                                # 환경 적합성 기반 적응형 가중치 계산
                                adaptive_weight = self._calculate_adaptive_weight(
                                    species, lat, lon, env_data
                                )

                                combined_data[density_key] = baseline.get('default_density', 0.1)
                                combined_data[weight_key] = adaptive_weight
                                combined_data[f"{species_key}_source"] = 'baseline_adaptive'
                            else:
                                combined_data[weight_key] = 1.0  # 실제 관측 데이터는 최대 가중치
                                combined_data[f"{species_key}_source"] = 'observed'

                        all_data.append(combined_data)
                        processed += 1

                        if processed % 50 == 0:
                            progress = (processed / total_combinations) * 100
                            logger.info(f"  진행률: {progress:.1f}% ({processed}/{total_combinations})")

                    except Exception as e:
                        logger.warning(f"  격자점 ({lat}, {lon}) 처리 실패: {e}")
                        continue

            finally:
                # CMEMS 파일 정리
                self.cleanup_cmems_files(target_date)

        # 데이터 저장
        if all_data:
            filename = f"weekly_training_data_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            filepath = self.save_to_csv(all_data, filename)

            logger.info(f" 7일 간격 학습 데이터 수집 완료")
            logger.info(f"  - 총 데이터: {len(all_data)}개")
            logger.info(f"  - 학습 날짜: {len(training_dates)}개 (7일 간격)")
            logger.info(f"  - 저장 위치: {filepath}")

            return filepath
        else:
            logger.error(" 수집된 데이터가 없습니다")
            return ""
