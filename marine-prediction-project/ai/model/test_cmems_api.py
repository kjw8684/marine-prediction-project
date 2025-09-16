#!/usr/bin/env python3
"""
CMEMS API 실제 다운로드 테스트
"""

import os
import sys
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cmems_download():
    """CMEMS API 실제 다운로드 테스트"""
    try:
        import copernicusmarine
        logger.info("✅ copernicusmarine 패키지 로드 성공")
        
        # 테스트용 다운로드 (3일 전 데이터 - 확실히 있는 날짜)
        test_date = datetime.now() - timedelta(days=4)
        target_date = test_date.strftime('%Y-%m-%d')
        
        logger.info(f"🎯 테스트 날짜: {target_date}")
        
        # CMEMS 출력 디렉터리
        cmems_dir = os.path.abspath("../../cmems_output")
        os.makedirs(cmems_dir, exist_ok=True)
        logger.info(f"📁 출력 디렉터리: {cmems_dir}")
        
        # 물리 데이터 다운로드 테스트
        test_nc = os.path.join(cmems_dir, f"test_cmems_phy_{target_date.replace('-', '')}.nc")
        
        logger.info("🌊 CMEMS 물리 데이터 다운로드 시작...")
        
        start_datetime = f"{target_date}T00:00:00"
        end_datetime = f"{target_date}T23:59:59"
        
        # CMEMS 다운로드 실행
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["thetao", "so", "uo", "vo"],  # 수온, 염도, 해류
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_longitude=124.0,
            maximum_longitude=132.0,
            minimum_latitude=33.0,
            maximum_latitude=39.0,
            output_filename=test_nc,
            overwrite=True
        )
        
        if os.path.exists(test_nc):
            file_size = os.path.getsize(test_nc)
            logger.info(f"✅ CMEMS 다운로드 성공!")
            logger.info(f"📄 파일: {test_nc}")
            logger.info(f"💾 크기: {file_size/1024/1024:.1f} MB")
            
            # NetCDF 파일 내용 확인
            try:
                import xarray as xr
                ds = xr.open_dataset(test_nc)
                logger.info(f"🔍 데이터 변수: {list(ds.data_vars.keys())}")
                logger.info(f"🔍 좌표: {list(ds.coords.keys())}")
                logger.info(f"🔍 시간 범위: {ds.time.values[0]} ~ {ds.time.values[-1]}")
                ds.close()
                logger.info("✅ NetCDF 파일 읽기 성공!")
                
            except Exception as e:
                logger.error(f"❌ NetCDF 파일 읽기 실패: {e}")
                
            return True
        else:
            logger.error("❌ CMEMS 다운로드 실패: 파일이 생성되지 않음")
            return False
            
    except ImportError:
        logger.error("❌ copernicusmarine 패키지 없음")
        return False
    except Exception as e:
        logger.error(f"❌ CMEMS 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 실행"""
    print("🧪 CMEMS API 다운로드 테스트")
    print("="*50)
    
    success = test_cmems_download()
    
    if success:
        print("\n🎉 CMEMS API 테스트 성공!")
        print("✅ 실제 해양 환경 데이터 다운로드 확인됨")
    else:
        print("\n💥 CMEMS API 테스트 실패!")
        print("❌ 네트워크 연결이나 API 설정을 확인하세요")

if __name__ == "__main__":
    main()
