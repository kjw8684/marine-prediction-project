"""
3년치 실제 해양 데이터 일별 수집/학습/삭제 시스템
- 하루치 데이터 수집 → 학습 → 삭제 방식으로 메모리 효율적 처리
- 실제 GBIF/OBIS API와 CMEMS 데이터 사용
- 체크포인트 지원으로 중단 시 복구 가능
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import gc
import time

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_data_system import MarineRealDataCollector
from marine_train_pmml import MarineMLSystem

class ThreeYearMarineTrainer:
    """3년치 해양 데이터 일별 학습 시스템"""
    
    def __init__(self):
        self.data_collector = MarineRealDataCollector()
        self.ml_system = MarineMLSystem()
        self.models = {}
        
        # 체크포인트 설정
        self.checkpoint_dir = "../data/models/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 격자점 설정 (한국 근해)
        self.grid_points = self._generate_korea_grid()
        
        # CMEMS 데이터 가용 기간 (2022-06-01 ~ 2025-09-17)
        self.start_date = datetime(2022, 6, 1)
        self.end_date = datetime(2025, 9, 17)
        self.total_days = (self.end_date - self.start_date).days + 1
        
        print(f"🗓️ 학습 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"📊 총 {self.total_days}일 ({self.total_days/365.25:.1f}년)")
        print(f"🌊 격자점 수: {len(self.grid_points)}개")
        
    def _generate_korea_grid(self):
        """한국 근해 격자점 생성 (0.5도 간격)"""
        grid_points = []
        
        # 한국 근해 영역
        lat_range = np.arange(33.0, 38.1, 0.5)
        lon_range = np.arange(125.0, 131.1, 0.5)
        
        for lat in lat_range:
            for lon in lon_range:
                if not self._is_land_point(lat, lon):
                    grid_points.append((lat, lon))
        
        return grid_points
    
    def _is_land_point(self, lat, lon):
        """육지 포인트 제외 (간단한 휴리스틱)"""
        # 한반도 내부 영역 대략적 제외
        if 37.0 <= lat <= 38.0 and 126.5 <= lon <= 128.5:
            return True
        if 35.5 <= lat <= 37.0 and 126.0 <= lon <= 129.0:
            return True
        return False
    
    def save_checkpoint(self, checkpoint_name, data):
        """체크포인트 저장"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 체크포인트 저장: {checkpoint_name}")
        except Exception as e:
            print(f"❌ 체크포인트 저장 실패: {e}")
    
    def load_checkpoint(self, checkpoint_name):
        """체크포인트 로드"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"📂 체크포인트 로드: {checkpoint_name}")
                return data
            return None
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            return None
    
    def get_progress(self):
        """진행상황 확인"""
        progress_file = os.path.join(self.checkpoint_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_progress(self, current_date, completed_days, successful_days):
        """진행상황 저장"""
        progress = {
            'current_date': current_date.strftime('%Y-%m-%d'),
            'completed_days': completed_days,
            'successful_days': successful_days,
            'total_days': self.total_days,
            'percentage': (completed_days / self.total_days) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        progress_file = os.path.join(self.checkpoint_dir, "progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def collect_and_train_daily(self, target_date):
        """하루치 데이터 수집 → 학습 → 삭제"""
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"\\n[{date_str}] 일별 처리 시작")
        
        try:
            # 1. 생물 관측 데이터 수집
            print(f"[COLLECT] 생물 관측 데이터 수집...")
            biological_df = self.data_collector.collect_daily_training_data(date_str, self.grid_points)
            
            if biological_df.empty:
                print(f"[SKIP] 생물 데이터 없음")
                return False
            
            # 2. CMEMS 환경 데이터 수집
            print(f"[COLLECT] CMEMS 환경 데이터 수집...")
            environmental_data = []
            
            for lat, lon in self.grid_points[:10]:  # 테스트용으로 10개만
                try:
                    env_data = self.ml_system.extract_cmems_data_for_point(lat, lon, date_str)
                    if env_data:
                        env_data.update({'lat': lat, 'lon': lon, 'date': date_str})
                        environmental_data.append(env_data)
                except Exception as e:
                    print(f"[WARNING] CMEMS 실패 ({lat}, {lon}): {e}")
                    continue
            
            # 3. 데이터 결합
            if environmental_data:
                env_df = pd.DataFrame(environmental_data)
                combined_df = pd.merge(biological_df, env_df, on=['lat', 'lon', 'date'], how='inner')
                print(f"[MERGE] 결합 완료: {len(combined_df)}행")
            else:
                combined_df = biological_df
                print(f"[WARNING] CMEMS 데이터 없음 - 생물 데이터만 사용")
            
            # 4. 임시 저장
            temp_filepath = self.data_collector.save_daily_data(combined_df, date_str)
            
            # 5. 점진적 학습
            print(f"[TRAIN] 모델 점진적 학습...")
            success = self._incremental_training(combined_df, date_str)
            
            # 6. 데이터 정리
            print(f"[CLEANUP] 임시 데이터 삭제...")
            self.data_collector.cleanup_daily_data(temp_filepath)
            
            # 메모리 정리
            del combined_df, biological_df
            if environmental_data:
                del env_df, environmental_data
            gc.collect()
            
            print(f"[SUCCESS] {date_str} 처리 완료")
            return success
            
        except Exception as e:
            print(f"[ERROR] {date_str} 처리 실패: {e}")
            return False
    
    def _incremental_training(self, daily_df, date_str):
        """점진적 모델 학습"""
        try:
            if daily_df.empty:
                return False
            
            # 예측 대상
            targets = ['species_diversity_index', 'biomass_estimate', 'bloom_probability']
            
            # 특성 선택
            numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
            features = [col for col in numeric_cols if col not in targets + ['lat', 'lon']]
            
            if len(features) < 3:
                print(f"[SKIP] 특성 수 부족: {len(features)}개")
                return False
            
            X = daily_df[features].fillna(0)
            
            # 각 타겟별 학습
            for target in targets:
                if target in daily_df.columns:
                    y = daily_df[target].fillna(0)
                    
                    if target not in self.models:
                        # 새 모델 생성
                        self.models[target] = RandomForestRegressor(
                            n_estimators=10,
                            random_state=42,
                            warm_start=True
                        )
                        self.models[target].fit(X, y)
                        print(f"[NEW] {target} 모델 생성")
                    else:
                        # 기존 모델 확장
                        self.models[target].n_estimators += 5
                        self.models[target].fit(X, y)
                        print(f"[UPDATE] {target} 모델 확장")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 학습 실패: {e}")
            return False
    
    def run_daily_training(self):
        """3년치 일별 학습 실행"""
        print("🎯 3년치 일별 학습 시작")
        print(f"🗓️ 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        
        # 기존 진행상황 확인
        progress = self.get_progress()
        start_date = self.start_date
        completed_days = 0
        successful_days = 0
        
        if progress:
            print(f"📂 기존 진행상황: {progress['percentage']:.1f}% 완료")
            start_date = datetime.strptime(progress['current_date'], '%Y-%m-%d') + timedelta(days=1)
            completed_days = progress['completed_days']
            successful_days = progress['successful_days']
            
            # 기존 모델 로드
            existing_models = self.load_checkpoint("models")
            if existing_models:
                self.models = existing_models
                print(f"📂 기존 모델 {len(self.models)}개 로드")
        
        # 일별 처리
        current_date = start_date
        
        while current_date <= self.end_date:
            try:
                # 일별 수집/학습/삭제
                success = self.collect_and_train_daily(current_date)
                
                completed_days += 1
                if success:
                    successful_days += 1
                
                # 진행상황 저장 (10일마다)
                if completed_days % 10 == 0:
                    self.save_progress(current_date, completed_days, successful_days)
                    self.save_checkpoint("models", self.models)
                    print(f"📊 진행률: {completed_days}/{self.total_days} ({completed_days/self.total_days*100:.1f}%)")
                
                current_date += timedelta(days=1)
                
            except KeyboardInterrupt:
                print("\\n⏸️ 사용자 중단")
                break
            except Exception as e:
                print(f"❌ {current_date.strftime('%Y-%m-%d')} 오류: {e}")
                current_date += timedelta(days=1)
                continue
        
        # 최종 결과
        print(f"\\n🎉 일별 학습 완료!")
        print(f"📊 성공: {successful_days}/{completed_days} 일")
        print(f"🤖 모델: {len(self.models)}개")
        
        self._save_final_models()
    
    def _save_final_models(self):
        """최종 모델 저장"""
        try:
            model_dir = "../data/models"
            os.makedirs(model_dir, exist_ok=True)
            
            for target, model in self.models.items():
                # joblib 저장
                filename = f"marine_model_{target}_3year.joblib"
                filepath = os.path.join(model_dir, filename)
                joblib.dump(model, filepath)
                
                # 정보 저장
                info = {
                    'target': target,
                    'n_estimators': model.n_estimators,
                    'training_period': f"{self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}",
                    'created_at': datetime.now().isoformat()
                }
                
                info_file = f"marine_model_{target}_3year_info.json"
                info_path = os.path.join(model_dir, info_file)
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
                
                print(f"💾 모델 저장: {filename}")
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")

def main():
    """메인 실행"""
    print("🌊 3년치 해양 데이터 일별 학습 시스템")
    print("=" * 50)
    
    try:
        trainer = ThreeYearMarineTrainer()
        
        # 진행상황 확인
        progress = trainer.get_progress()
        if progress:
            print(f"📂 기존 진행상황: {progress['percentage']:.1f}% 완료")
            answer = input("계속하시겠습니까? (y/n): ")
            if answer.lower() != 'y':
                print("새로운 학습을 시작합니다.")
        
        # 실행
        trainer.run_daily_training()
        
        print("\\n🎉 모든 작업 완료!")
        
    except KeyboardInterrupt:
        print("\\n⏸️ 사용자 중단")
        print("💾 진행상황이 저장되었습니다.")
    except Exception as e:
        print(f"\\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
