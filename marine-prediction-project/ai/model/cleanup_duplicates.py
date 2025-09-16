#!/usr/bin/env python3
"""
중복 다운로드된 CMEMS .nc 파일들을 정리하는 스크립트
"""

import os
import glob
import shutil

def cleanup_duplicate_files():
    """중복된 .nc 파일들을 정리"""
    
    # CMEMS 출력 디렉토리 경로
    cmems_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cmems_output")
    
    if not os.path.exists(cmems_dir):
        print(f"❌ CMEMS 디렉토리가 존재하지 않습니다: {cmems_dir}")
        return
    
    print(f"🔍 중복 파일 검색 중: {cmems_dir}")
    
    # 중복 파일 패턴 검색
    duplicate_patterns = [
        "cmems_phy_*_(*).nc",  # cmems_phy_2022-06-17_(1).nc 등
        "cmems_bgc_*_(*).nc",  # cmems_bgc_2022-06-17_(1).nc 등
    ]
    
    total_deleted = 0
    total_size_saved = 0
    
    for pattern in duplicate_patterns:
        pattern_path = os.path.join(cmems_dir, pattern)
        duplicate_files = glob.glob(pattern_path)
        
        print(f"📋 패턴 '{pattern}'으로 {len(duplicate_files)}개 중복 파일 발견")
        
        for file_path in duplicate_files:
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                total_deleted += 1
                total_size_saved += file_size
                print(f"   🗑️ 삭제: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
            except Exception as e:
                print(f"   ❌ 삭제 실패: {os.path.basename(file_path)} - {e}")
    
    # 결과 요약
    print(f"\n✅ 정리 완료:")
    print(f"   📂 삭제된 파일: {total_deleted}개")
    print(f"   💾 절약된 공간: {total_size_saved/1024/1024:.1f}MB")
    
    # 남은 파일 확인
    remaining_files = []
    for ext_type in ["phy", "bgc"]:
        pattern = os.path.join(cmems_dir, f"cmems_{ext_type}_*.nc")
        remaining_files.extend(glob.glob(pattern))
    
    print(f"   📄 남은 파일: {len(remaining_files)}개")
    
    if remaining_files:
        print("\n📋 남은 파일 목록 (최근 10개):")
        for file_path in sorted(remaining_files)[-10:]:
            file_size = os.path.getsize(file_path)
            print(f"   📄 {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")

if __name__ == "__main__":
    cleanup_duplicate_files()
