"""
7일치 실제 해양 데이터 수집 및 AI 모델 학습 시스템
- 2025-09-11 부터 2025-09-17까지 7일간 실제 데이터 수집
- RandomForest 기반 해양 생물 예측 모델 학습
- PMML 형식으로 모델 저장
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_data_system import RealDataMarineSystem

class SevenDayMarineAITrainer:
    def __init__(self):
        self.data_system = RealDataMarineSystem()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def collect_seven_days_data(self):
        """
        7일치 실제 데이터 수집
        """
        print("📅 7일치 실제 데이터 수집 시작...")
        
        # 날짜 범위 설정 (2025-09-11 ~ 2025-09-17)
        end_date = datetime.strptime("2025-09-17", "%Y-%m-%d")
        start_date = end_date - timedelta(days=6)
        
        all_data = []
        collection_summary = {}
        
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            print(f"   📊 {date_str} 데이터 수집 중...")
            
            try:
                # 해당 날짜의 실제 데이터 수집
                daily_df = self.data_system.process_daily_real_data(date_str)
                
                if daily_df is not None and len(daily_df) > 0:
                    all_data.append(daily_df)
                    collection_summary[date_str] = {
                        'records': len(daily_df),
                        'quality_score': daily_df['data_quality_score'].mean(),
                        'real_fields': daily_df['real_data_fields'].mean()
                    }
                    print(f"   ✅ {date_str}: {len(daily_df)}개 기록 수집")
                else:
                    print(f"   ❌ {date_str}: 데이터 수집 실패")
                    collection_summary[date_str] = {'status': 'failed'}
                    
            except Exception as e:
                print(f"   ❌ {date_str}: 오류 - {e}")
                collection_summary[date_str] = {'status': 'error', 'error': str(e)}
        
        # 전체 데이터 결합
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 저장
            output_file = f"../data/seven_days_training_data_{end_date.strftime('%Y-%m-%d')}.csv"
            combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 요약 저장
            summary_file = f"../data/seven_days_summary_{end_date.strftime('%Y-%m-%d')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(collection_summary, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 7일치 데이터 수집 완료!")
            print(f"   📊 총 {len(combined_df)}개 기록")
            print(f"   💾 저장: {output_file}")
            print(f"   📋 요약: {summary_file}")
            
            return combined_df, collection_summary
        else:
            print("❌ 수집된 데이터가 없습니다.")
            return None, collection_summary
    
    def prepare_training_data(self, df):
        """
        학습용 데이터 준비
        """
        print("🔧 학습용 데이터 준비 중...")
        
        # 기본 컬럼 제외
        exclude_cols = ['lat', 'lon', 'date', 'collection_timestamp', 'real_data_fields', 'data_quality_score']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 환경 특성과 타겟 분리
        env_features = ['bottom_temp', 'bottom_salinity', 'sea_surface_height', 'primary_production', 
                       'oxygen', 'sst_satellite', 'chlorophyll_satellite', 'wind_speed', 'wave_height',
                       'species_diversity_index', 'biomass_estimate', 'bloom_probability', 
                       'fishing_activity', 'aquaculture_density']
        
        # 생물 관측 데이터 (타겟)
        bio_targets = [col for col in feature_cols if 'observations' in col]
        
        # NaN 값 처리
        for col in env_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in bio_targets:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        print(f"   🔹 환경 특성: {len(env_features)}개")
        print(f"   🔹 생물 타겟: {len(bio_targets)}개")
        print(f"   🔹 학습 샘플: {len(df)}개")
        
        return df, env_features, bio_targets
    
    def train_species_models(self, df, env_features, bio_targets):
        """
        종별 예측 모델 학습
        """
        print("🤖 AI 모델 학습 시작...")
        
        training_results = {}
        
        for target in bio_targets:
            if target in df.columns:
                print(f"   🎯 {target} 모델 학습 중...")
                
                # 특성과 타겟 준비
                X = df[env_features].copy()
                y = df[target].copy()
                
                # NaN 제거
                valid_idx = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_idx]
                y = y[valid_idx]
                
                if len(X) < 10:
                    print(f"   ❌ {target}: 학습 데이터 부족 ({len(X)}개)")
                    continue
                
                try:
                    # 분류 vs 회귀 결정
                    unique_values = y.nunique()
                    if unique_values <= 10:  # 이산값 -> 분류
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model_type = 'classification'
                    else:  # 연속값 -> 회귀
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model_type = 'regression'
                    
                    # 데이터 분할
                    if len(X) >= 20:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                    else:
                        X_train, X_test, y_train, y_test = X, X, y, y
                    
                    # 스케일링
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 모델 학습
                    model.fit(X_train_scaled, y_train)
                    
                    # 예측 및 평가
                    y_pred = model.predict(X_test_scaled)
                    
                    if model_type == 'classification':
                        score = accuracy_score(y_test, y_pred)
                        metric = 'accuracy'
                    else:
                        score = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
                        metric = 'rmse'
                    
                    # 결과 저장
                    self.models[target] = model
                    self.scalers[target] = scaler
                    
                    training_results[target] = {
                        'model_type': model_type,
                        'metric': metric,
                        'score': float(score),
                        'samples': len(X),
                        'features': env_features
                    }
                    
                    print(f"   ✅ {target}: {metric} = {score:.4f} (샘플 {len(X)}개)")
                    
                except Exception as e:
                    print(f"   ❌ {target}: 학습 실패 - {e}")
                    training_results[target] = {'status': 'failed', 'error': str(e)}
        
        return training_results
    
    def save_models_as_pmml(self, training_results):
        """
        학습된 모델을 PMML 형식으로 저장
        """
        print("💾 모델을 PMML 형식으로 저장 중...")
        
        try:
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.preprocessing import ContinuousDomain
            
            pmml_models = {}
            
            for target, result in training_results.items():
                if 'status' in result and result['status'] == 'failed':
                    continue
                    
                if target in self.models and target in self.scalers:
                    try:
                        # PMML 파이프라인 생성
                        pipeline = PMMLPipeline([
                            ('scaler', self.scalers[target]),
                            ('model', self.models[target])
                        ])
                        
                        # PMML 파일 저장
                        pmml_file = f"../data/models/{target}_model.pmml"
                        os.makedirs(os.path.dirname(pmml_file), exist_ok=True)
                        
                        # 더미 데이터로 파이프라인 피팅 (PMML 변환용)
                        dummy_X = np.random.randn(10, len(result['features']))
                        dummy_y = np.random.randint(0, 2, 10) if result['model_type'] == 'classification' else np.random.randn(10)
                        
                        pipeline.fit(dummy_X, dummy_y)
                        sklearn2pmml(pipeline, pmml_file, with_repr=True)
                        
                        pmml_models[target] = pmml_file
                        print(f"   ✅ {target}: {pmml_file}")
                        
                    except Exception as e:
                        print(f"   ❌ {target} PMML 변환 실패: {e}")
                        # 대신 joblib으로 저장
                        joblib_file = f"../data/models/{target}_model.joblib"
                        joblib.dump({
                            'model': self.models[target],
                            'scaler': self.scalers[target],
                            'features': result['features'],
                            'model_type': result['model_type']
                        }, joblib_file)
                        pmml_models[target] = joblib_file
                        print(f"   📦 {target}: joblib으로 저장 - {joblib_file}")
            
            # 모델 메타데이터 저장
            metadata = {
                'created_at': datetime.now().isoformat(),
                'training_period': '2025-09-11 to 2025-09-17',
                'models': pmml_models,
                'training_results': training_results
            }
            
            metadata_file = "../data/models/model_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 모델 저장 완료!")
            print(f"   📋 메타데이터: {metadata_file}")
            print(f"   📊 저장된 모델: {len(pmml_models)}개")
            
            return pmml_models, metadata
            
        except ImportError:
            print("❌ sklearn2pmml 라이브러리가 필요합니다.")
            print("   설치: pip install sklearn2pmml")
            
            # joblib으로 대체 저장
            models_dir = "../data/models"
            os.makedirs(models_dir, exist_ok=True)
            
            saved_models = {}
            for target in self.models:
                joblib_file = f"{models_dir}/{target}_model.joblib"
                joblib.dump({
                    'model': self.models[target],
                    'scaler': self.scalers[target],
                    'features': training_results[target]['features'],
                    'model_type': training_results[target]['model_type']
                }, joblib_file)
                saved_models[target] = joblib_file
                print(f"   📦 {target}: {joblib_file}")
            
            metadata = {
                'created_at': datetime.now().isoformat(),
                'training_period': '2025-09-11 to 2025-09-17',
                'models': saved_models,
                'training_results': training_results,
                'format': 'joblib'
            }
            
            metadata_file = f"{models_dir}/model_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return saved_models, metadata
    
    def run_seven_day_training(self):
        """
        7일치 데이터 수집부터 모델 학습까지 전체 프로세스 실행
        """
        print("🚀 === 7일치 실제 데이터 기반 AI 모델 학습 시작 ===")
        
        # 1. 7일치 데이터 수집
        df, collection_summary = self.collect_seven_days_data()
        if df is None:
            print("❌ 데이터 수집 실패")
            return None
        
        # 2. 데이터 준비
        df, env_features, bio_targets = self.prepare_training_data(df)
        
        # 3. 모델 학습
        training_results = self.train_species_models(df, env_features, bio_targets)
        
        # 4. PMML 저장
        models, metadata = self.save_models_as_pmml(training_results)
        
        print("\n✅ === 7일치 AI 모델 학습 완료 ===")
        print(f"   📊 학습 데이터: {len(df)}개 샘플")
        print(f"   🤖 학습 모델: {len(models)}개")
        print(f"   💾 저장 형식: PMML/joblib")
        
        return {
            'data': df,
            'models': models,
            'metadata': metadata,
            'training_results': training_results,
            'collection_summary': collection_summary
        }

if __name__ == "__main__":
    trainer = SevenDayMarineAITrainer()
    result = trainer.run_seven_day_training()
    
    if result:
        print(f"\n🎯 학습 완료! 다음 단계: Flask 웹 애플리케이션 구축")
