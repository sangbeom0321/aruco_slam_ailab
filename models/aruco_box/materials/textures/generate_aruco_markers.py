#!/usr/bin/env python3
"""
ArUco 마커 생성 스크립트
4x4_50 사전에서 ID 0부터 9까지 10개의 마커를 생성합니다.
"""

import cv2
import numpy as np
import os

def generate_aruco_markers(output_dir, num_markers=10, marker_size=200, dict_type=cv2.aruco.DICT_4X4_50):
    """
    ArUco 마커를 생성하고 저장합니다.
    
    Args:
        output_dir: 마커를 저장할 디렉토리 경로
        num_markers: 생성할 마커 개수
        marker_size: 마커 이미지 크기 (픽셀)
        dict_type: ArUco 사전 타입
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # ArUco 사전 가져오기
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    
    print(f"ArUco 마커 {num_markers}개 생성 중...")
    
    for marker_id in range(num_markers):
        # 마커 이미지 생성
        marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
        
        # 파일명 생성
        filename = f"aruco_4x4_50_id{marker_id}.png"
        filepath = os.path.join(output_dir, filename)
        
        # 이미지 저장
        cv2.imwrite(filepath, marker_image)
        print(f"생성 완료: {filename}")
    
    print(f"\n모든 마커가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    # 현재 스크립트의 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 마커 생성
    generate_aruco_markers(script_dir, num_markers=10, marker_size=200)
