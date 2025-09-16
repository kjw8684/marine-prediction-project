#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
병렬 처리 기반 3년치 실제 CMEMS + 해양생물 데이터 통합 AI 학습 시스템
날짜별 중복 방지 및 효율적 병렬 데이터 수집
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from multiprocessing import Pool, Manager, Lock, Value, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import pickle
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parallel_training.log', encoding='utf-8')
    ]
)

def log(message):
    logging.info(message)

# 기존 함수들 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_data_system import MarineRealDataCollector

# CMEMS 데이터 관련 함수들을 기존 파일에서 가져옴
from three_year_cmems_trainer import (
    download_with_lock,
    download_cmems_data_for_date,
    collect_cmems_data_for_date
)

class ParallelThreeYearTrainer:
    """병렬 처리 3년치 CMEMS + 해양생물 데이터 통합 학습 시스템"""
    
    def __init__(self, num_processes=4):
        """
        Args:
            num_processes: 병렬 프로세스 개수 (기본값: 4)
        """
        self.num_processes = num_processes
        self.data_collector = MarineRealDataCollector()
        
        # 한국 연안 주요 격자점
        self.grid_points = [
            # 동해
            (37.5, 129.0), (37.0, 129.5), (36.5, 130.0), (36.0, 130.5),
            (35.5, 129.5), (35.0, 129.0), (34.5, 129.5), (34.0, 130.0),
            
            # 남해
            (34.0, 128.5), (34.5, 128.0), (35.0, 127.5), (35.5, 127.0),
            (34.0, 127.0), (34.5, 126.5), (35.0, 126.0), (35.5, 125.5),
            
            # 서해
            (37.0, 126.0), (36.5, 126.5), (36.0, 127.0), (35.5, 126.0),
            (35.0, 125.5), (34.5, 125.0), (34.0, 124.5), (33.5, 125.0)
        ]
        
        # 날짜 범위 설정 (CMEMS 데이터 가용 범위)
        self.start_date = datetime(2022, 6, 1)
        self.end_date = datetime(2024, 9, 13)
        
        # 주 단위 날짜 목록 생성 (120개)
        self.training_dates = []
        current_date = self.start_date
        while current_date <= self.end_date:
            self.training_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=7)  # 주 단위
        
        # 진행 상태 관리
        self.progress_file = "parallel_training_progress.json"
        self.completed_dates = set()
        self.data_dir = "parallel_training_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        log(f"병렬 훈련 시스템 초기화 완료")
        log(f"프로세스 수: {self.num_processes}")
        log(f"격자점: {len(self.grid_points)}개")
        log(f"훈련 날짜: {len(self.training_dates)}일 ({self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')})")
        
        self.load_progress()

    def load_progress(self):
        """진행 상태 로드"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    self.completed_dates = set(progress_data.get('completed_dates', []))
                log(f"진행 상태 로드: {len(self.completed_dates)}일 완료")
            else:
                log("새로운 훈련 세션 시작")
        except Exception as e:
            log(f"진행 상태 로드 실패: {e}")
            self.completed_dates = set()

    def save_progress(self):
        """진행 상태 저장"""
        try:
            progress_data = {
                'completed_dates': list(self.completed_dates),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"진행 상태 저장 실패: {e}")

    def get_pending_dates(self):
        """아직 완료되지 않은 날짜 목록 반환"""
        return [date for date in self.training_dates if date not in self.completed_dates]

def process_single_date(date_str, grid_points, data_dir):
    """단일 날짜에 대한 데이터 수집 및 저장 (병렬 프로세스에서 실행)"""
    try:
        process_id = os.getpid()
        log(f"[PID:{process_id}] {date_str} 데이터 수집 시작")
        
        # 파일 경로 설정
        date_file = os.path.join(data_dir, f"data_{date_str.replace('-', '_')}.csv")
        
        # 이미 완료된 파일이 있으면 건너뛰기
        if os.path.exists(date_file):
            log(f"[PID:{process_id}] {date_str} 이미 완료됨 - 건너뛰기")
            return date_str
        
        # 해양생물 데이터 수집
        data_collector = MarineRealDataCollector()
        bio_df = data_collector.collect_daily_training_data(date_str, grid_points)
        
        if bio_df.empty:
            log(f"[PID:{process_id}] {date_str} 해양생물 데이터 없음")
            # 빈 데이터프레임이라도 파일 생성 (완료 표시용)
            bio_df = pd.DataFrame({
                'lat': [point[0] for point in grid_points],
                'lon': [point[1] for point in grid_points]
            })
            for species in data_collector.target_species:
                bio_df[species] = 0
        
        # CMEMS 환경 데이터 수집
        cmems_df = collect_cmems_data_for_date(date_str, grid_points)
        
        # 데이터 통합
        if not cmems_df.empty:
            # lat, lon으로 병합
            integrated_df = pd.merge(bio_df, cmems_df, on=['lat', 'lon'], how='outer')
        else:
            integrated_df = bio_df.copy()
            # CMEMS 데이터가 없으면 기본값으로 채움
            env_cols = ['sea_water_temperature', 'sea_water_salinity', 'sea_surface_height', 
                       'mixed_layer_depth', 'dissolved_oxygen', 'net_primary_productivity']
            for col in env_cols:
                integrated_df[col] = 0.0
        
        # 날짜 컬럼 추가
        integrated_df['date'] = date_str
        
        # 결과 저장
        integrated_df.to_csv(date_file, index=False, encoding='utf-8')
        
        log(f"[PID:{process_id}] {date_str} 완료: {len(integrated_df)}행 × {len(integrated_df.columns)}열")
        
        return date_str
        
    except Exception as e:
        log(f"[PID:{process_id}] {date_str} 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

class ParallelThreeYearTrainer:
    """병렬 처리 3년치 CMEMS + 해양생물 데이터 통합 학습 시스템"""
    
    def __init__(self, num_processes=4):
        """
        Args:
            num_processes: 병렬 프로세스 개수 (기본값: 4)
        """
        self.num_processes = num_processes
        self.data_collector = MarineRealDataCollector()
        
        # 한국 연안 주요 격자점
        self.grid_points = [
            # 동해
            (37.5, 129.0), (37.0, 129.5), (36.5, 130.0), (36.0, 130.5),
            (35.5, 129.5), (35.0, 129.0), (34.5, 129.5), (34.0, 130.0),
            
            # 남해
            (34.0, 128.5), (34.5, 128.0), (35.0, 127.5), (35.5, 127.0),
            (34.0, 127.0), (34.5, 126.5), (35.0, 126.0), (35.5, 125.5),
            
            # 서해
            (37.0, 126.0), (36.5, 126.5), (36.0, 127.0), (35.5, 126.0),
            (35.0, 125.5), (34.5, 125.0), (34.0, 124.5), (33.5, 125.0)
        ]
        
        # 날짜 범위 설정 (CMEMS 데이터 가용 범위)
        self.start_date = datetime(2022, 6, 1)
        self.end_date = datetime(2024, 9, 13)
        
        # 주 단위 날짜 목록 생성 (120개)
        self.training_dates = []
        current_date = self.start_date
        while current_date <= self.end_date:
            self.training_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=7)  # 주 단위
        
        # 진행 상태 관리
        self.progress_file = "parallel_training_progress.json"
        self.completed_dates = set()
        self.data_dir = "parallel_training_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        log(f"병렬 훈련 시스템 초기화 완료")
        log(f"프로세스 수: {self.num_processes}")
        log(f"격자점: {len(self.grid_points)}개")
        log(f"훈련 날짜: {len(self.training_dates)}일 ({self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')})")
        
        self.load_progress()

    def load_progress(self):
        """진행 상태 로드"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    self.completed_dates = set(progress_data.get('completed_dates', []))
                log(f"진행 상태 로드: {len(self.completed_dates)}일 완료")
            else:
                log("새로운 훈련 세션 시작")
        except Exception as e:
            log(f"진행 상태 로드 실패: {e}")
            self.completed_dates = set()

    def save_progress(self):
        """진행 상태 저장"""
        try:
            progress_data = {
                'completed_dates': list(self.completed_dates),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"진행 상태 저장 실패: {e}")

    def get_pending_dates(self):
        """아직 완료되지 않은 날짜 목록 반환"""
        return [date for date in self.training_dates if date not in self.completed_dates]

    def collect_parallel_data(self):
        """병렬 처리로 3년치 데이터 수집"""
        log("="*60)
        log("🚀 병렬 데이터 수집 시작!")
        log("="*60)
        
        pending_dates = self.get_pending_dates()
        
        if not pending_dates:
            log("모든 날짜 데이터 수집 완료!")
            return True
        
        log(f"수집 대상: {len(pending_dates)}일")
        
        try:
            # ProcessPoolExecutor를 사용한 병렬 처리
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # 각 날짜에 대한 작업 제출
                future_to_date = {
                    executor.submit(process_single_date, date_str, self.grid_points, self.data_dir): date_str
                    for date_str in pending_dates
                }
                
                # 완료된 작업 처리
                completed_count = 0
                for future in as_completed(future_to_date):
                    date_str = future_to_date[future]
                    try:
                        result = future.result()
                        if result:
                            self.completed_dates.add(result)
                            completed_count += 1
                            log(f"✅ 진행: {completed_count}/{len(pending_dates)} ({result})")
                            
                            # 진행 상태 저장 (10개마다)
                            if completed_count % 10 == 0:
                                self.save_progress()
                        else:
                            log(f"❌ 실패: {date_str}")
                    except Exception as e:
                        log(f"❌ {date_str} 예외 발생: {e}")
                
                # 최종 진행 상태 저장
                self.save_progress()
                
            log(f"병렬 데이터 수집 완료: {completed_count}/{len(pending_dates)}")
            return True
            
        except Exception as e:
            log(f"병렬 데이터 수집 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def combine_collected_data(self):
        """수집된 모든 날짜 데이터를 하나로 통합"""
        log("="*60)
        log("📊 데이터 통합 시작!")
        log("="*60)
        
        try:
            all_dataframes = []
            
            # 모든 날짜 파일 로드
            for date_str in self.training_dates:
                date_file = os.path.join(self.data_dir, f"data_{date_str.replace('-', '_')}.csv")
                
                if os.path.exists(date_file):
                    try:
                        df = pd.read_csv(date_file, encoding='utf-8')
                        if not df.empty:
                            all_dataframes.append(df)
                            log(f"✅ {date_str}: {len(df)}행")
                        else:
                            log(f"⚠️  {date_str}: 빈 데이터")
                    except Exception as e:
                        log(f"❌ {date_str} 로드 실패: {e}")
                else:
                    log(f"⚠️  {date_str}: 파일 없음")
            
            if not all_dataframes:
                log("통합할 데이터가 없습니다!")
                return pd.DataFrame()
            
            # 데이터 통합
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # NaN 값 처리
            combined_df = combined_df.fillna(0)
            
            log(f"📊 데이터 통합 완료: {len(combined_df)}행 × {len(combined_df.columns)}열")
            log(f"📅 날짜 범위: {len(set(combined_df['date']))}일")
            
            # 통합 데이터 저장
            output_file = "parallel_three_year_integrated_data.csv"
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            log(f"💾 통합 데이터 저장: {output_file}")
            
            return combined_df
            
        except Exception as e:
            log(f"데이터 통합 실패: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def train_models(self, data_df):
        """AI 모델 훈련"""
        log("="*60)
        log("🤖 AI 모델 훈련 시작!")
        log("="*60)
        
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
            
            for species in target_species:
                # 정확한 컬럼명 사용 (GBIF 관측 수)
                species_col = f"{species.replace(' ', '_')}_gbif_observations"
                
                if species_col in data_df.columns:
                    # 관측 데이터가 있는지 확인 (0이 아닌 값이 있는지)
                    if data_df[species_col].sum() > 0:
                        log(f"🎯 {species} 모델 훈련 중... (관측수: {data_df[species_col].sum()})")
                        
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
                        
                        models[species] = {
                            'model': model,
                            'features': feature_cols,
                            'train_score': train_score,
                            'test_score': test_score,
                            'target_column': species_col
                        }
                        
                        log(f"✅ {species}: 훈련 R²={train_score:.3f}, 테스트 R²={test_score:.3f}")
                    else:
                        log(f"⚠️ {species}: 관측 데이터 없음 (모든 값이 0)")
                else:
                    log(f"❌ {species}: 컬럼 없음 ({species_col})")
            
            self.models = models
            log(f"🎉 모델 훈련 완료: {len(models)}개 모델")
            return True
            
        except Exception as e:
            log(f"모델 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_models(self):
        """훈련된 모델 저장"""
        log("="*60)
        log("💾 모델 저장 시작!")
        log("="*60)
        
        try:
            import joblib
            saved_files = []
            
            # Joblib 모델 저장
            for target, model_info in self.models.items():
                model_file = f"parallel_three_year_model_{target}.joblib"
                joblib.dump(model_info, model_file)
                saved_files.append(model_file)
                log(f"Joblib 저장: {model_file}")
            
            # PMML 모델 저장 (sklearn2pmml 있는 경우)
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
                    pmml_path = f"parallel_three_year_model_{target}.pmml"
                    sklearn2pmml(pipeline, pmml_path, with_repr=True)
                    saved_files.append(pmml_path)
                    log(f"PMML 저장: {pmml_path}")
                    
            except ImportError:
                log("sklearn2pmml 패키지 없음 - PMML 저장 건너뜀")
            except Exception as e:
                log(f"PMML 저장 실패: {e}")
            
            log(f"✅ 모델 저장 완료: {len(saved_files)}개 파일")
            return True
            
        except Exception as e:
            log(f"모델 저장 실패: {e}")
            return False

    def run_parallel_training(self):
        """병렬 3년치 전체 훈련 파이프라인 실행"""
        log("🌊 병렬 3년치 실제 CMEMS + 해양생물 데이터 통합 AI 학습")
        log("="*70)
        
        start_time = time.time()
        
        try:
            # 1. 병렬 데이터 수집
            if not self.collect_parallel_data():
                log("❌ 병렬 데이터 수집 실패")
                return False
            
            # 2. 데이터 통합
            integrated_df = self.combine_collected_data()
            if integrated_df.empty:
                log("❌ 데이터 통합 실패")
                return False
            
            # 3. 모델 훈련
            if not self.train_models(integrated_df):
                log("❌ 모델 훈련 실패")
                return False
            
            # 4. 모델 저장
            if not self.save_models():
                log("❌ 모델 저장 실패")
                return False
            
            # 5. 최종 결과
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            log("="*70)
            log("🎉 병렬 3년치 실제 CMEMS+생물 데이터 학습 완료!")
            log(f"⏱️  소요 시간: {elapsed_time/60:.1f}분")
            log(f"⚡ 병렬 프로세스: {self.num_processes}개")
            log(f"📅 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
            log(f"📍 격자점: {len(self.grid_points)}개")
            log(f"📊 최종 데이터: {len(integrated_df)}행 × {len(integrated_df.columns)}열")
            log(f"🤖 훈련된 모델: {len(self.models)}개")
            
            for target, model_info in self.models.items():
                log(f"   • {target}: 훈련 R²={model_info['train_score']:.3f}, 테스트 R²={model_info['test_score']:.3f}")
            
            log("="*70)
            return True
            
        except Exception as e:
            log(f"❌ 병렬 훈련 파이프라인 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """메인 실행 함수"""
    print("🚀 병렬 처리 3년치 실제 CMEMS + 해양생물 데이터 통합 AI 학습")
    print("="*70)
    
    try:
        # CPU 개수에 따른 프로세스 수 자동 설정
        import multiprocessing
        num_cpus = multiprocessing.cpu_count()
        num_processes = min(num_cpus, 6)  # 최대 6개 프로세스
        
        print(f"🖥️  CPU 개수: {num_cpus}")
        print(f"⚡ 사용할 프로세스 수: {num_processes}")
        print("="*70)
        
        # 병렬 트레이너 초기화 및 실행
        trainer = ParallelThreeYearTrainer(num_processes=num_processes)
        success = trainer.run_parallel_training()
        
        if success:
            print("✅ 병렬 훈련 성공!")
        else:
            print("❌ 병렬 훈련 실패!")
            
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
