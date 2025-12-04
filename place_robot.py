import time
import random
from pymycobot.mycobot import MyCobot
from ultralytics import YOLO
import cv2
import numpy as np

# MyCobot 시리얼 설정 (사용 환경에 맞게 반드시 수정)
PORT = "COM11"  # 예: Windows에서는 "COM3", Linux에서는 "/dev/ttyACM0" 등
BAUD = 115200

# 로봇 이동 속도 및 그리퍼 값
SPEED = 30
GRIPPER_OPEN = 75
GRIPPER_CLOSE = 20 
REQUIRED_CONSISTENCY_TIME = 1.5
WINDOW_NAME = "YOLOv8 Block Detection"

# --- 조인트 각도 상수 정의 (Degree) ---
HOME = [0, 0, 0, 0, 0, 0]
WORKSPACE = [38.64, -28.2, -21.47, -29.13, 88.46, -47.1]
GRIP_POINT = [24.16, -24.25, -39.28, -24.87, 90.43, -59.94]
STAY_POINT1 = [39.46, -13.71, 1.23, -71.89, 89.29, -44.56]

COLOR_TARGETS = {
    "Red Block": {
        "stay": [12.91, 5.62, 33.48, -57.39, -107.57, -2.1],
        "drop": [12.91, 43.94, -4.21, -44.73, -107.31, 0.79]
    },
    "Green Block": {
        "stay": [-107.4, 103, -140, 40.25, 20.47, 2.1],
        "drop": [-44.03, 119.68, -109.42, -9.49, -53.87, -0.08]
    },
    "Blue Block": {
        "stay": [-88.68, -3.77, 7.2, -13.63, -2.81, 7.2],
        "drop": [-41.04, 25.75, 23.9, -50.27, -51.06, -3.86]
    },
    "yellow block": {
        "stay": [43.59, 117.86, -99.58, -27.68, -139.04, -2.19],
        "drop": [20.3, 103.18, -44.03, -64.16, -117.86, 1.14]
    }
}

# --- MyCobot 제어 함수 ---
def move_mycobot(mc, joints, speed=SPEED):
    """지정된 조인트 각도로 MyCobot을 이동시킵니다."""
    if mc:
        mc.send_angles(joints, speed)
        time.sleep(2) 
    pass 

def control_gripper(mc, action):
    """그리퍼를 열거나 닫습니다."""
    if action == "open":
        if mc: mc.set_gripper_value(GRIPPER_OPEN, 50, 1)
    elif action == "close":
        if mc: mc.set_gripper_value(GRIPPER_CLOSE, 50, 1)
    time.sleep(0.5) 
    pass

def initialize_mycobot(mc):
    """0. Home으로 이동하고 그리퍼를 엽니다."""
    move_mycobot(mc, HOME, speed=50)
    control_gripper(mc, "open")

# --- 단일 프레임 색상 탐지 함수 (카메라 켜고 끄기 및 화면 표시) ---
def capture_and_detect(yolo_model, consistent_color, consistent_start_time):
    """
    카메라를 켜고, 단일 프레임을 캡처하며, YOLO 추론 결과를 화면에 표시합니다.
    """

    if yolo_model is None:
        return None, None
        
    detected_class_name = None
    frame_to_show = None
    
    cap = cv2.VideoCapture(0) # 1번 카메라 사용 (0번이 아닐 경우 1로 유지)
    
    if cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)

            S_factor = 1.4 
            V_factor = 1.4 
            S_boosted = np.clip(S * S_factor, 0, 255).astype(np.uint8)
            V_boosted = np.clip(V * V_factor, 0, 255).astype(np.uint8)

            hsv_boosted = cv2.merge([H, S_boosted, V_boosted])
            processed_frame = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
            
            results = yolo_model(processed_frame, verbose=False, conf=0.7)

            frame_to_show = results[0].plot() 
            
            if results and results[0].boxes and len(results[0].boxes) > 0:
                idx = int(results[0].boxes[0].cls)
                detected_class_name = results[0].names[idx]
                
                if detected_class_name not in COLOR_TARGETS:
                    detected_class_name = None

            if consistent_color and consistent_start_time:
                elapsed_time = time.time() - consistent_start_time
                text = f"Consistent: {consistent_color} ({elapsed_time:.1f}/{REQUIRED_CONSISTENCY_TIME}s)"
                color = (0, 255, 0) if elapsed_time >= REQUIRED_CONSISTENCY_TIME else (255, 255, 255) 
                cv2.putText(frame_to_show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame_to_show)
            
        cap.release() 
        return detected_class_name, frame_to_show
    
    return None, None

# --- 주요 자동화 시퀀스 함수 ---
def perform_pick_and_place(mc, detected_color):
    """
    안정화된 색상에 따라 블록을 집고 지정된 위치에 놓는 시퀀스를 실행합니다.
    """
    print(f"\nStarting sequence for **{detected_color}**")
    
    target = COLOR_TARGETS.get(detected_color)
    if not target:
        print(f"ERROR: Unknown color target: {detected_color}")
        return

    control_gripper(mc, "open")

    move_mycobot(mc, GRIP_POINT)
    control_gripper(mc, "close")
    time.sleep(0.5)

    move_mycobot(mc, STAY_POINT1)
    time.sleep(1)
    move_mycobot(mc, HOME)

    color_stay = target["stay"]
    color_drop = target["drop"]
    
    move_mycobot(mc, color_stay)
    move_mycobot(mc, color_drop)
    
    control_gripper(mc, "open")
    time.sleep(0.5)
    
    move_mycobot(mc, color_stay)
    move_mycobot(mc, HOME)

    print("-> Pick and Place sequence finished.")


# --- 메인 루프 ---
def main_loop():
    mc_instance = None 
    yolo_instance = None 
    
    if MyCobot and YOLO:
        try:
            mc_instance = MyCobot(PORT, BAUD)
            yolo_instance = YOLO(r'C:\Users\yuniw\Desktop\py_learning\PJT_trial\mycobot\best.pt') 
            print(f"MyCobot connected and YOLO model loaded. Starting detection loop...")
        except Exception as e:
            print(f"Initialization Error: {e}")
            if 'best.pt' in str(e):
                print("YOLO 모델 파일('best.pt')을 찾을 수 없습니다. 경로를 확인하세요.")
            
            mc_instance = None
            yolo_instance = None
            if mc_instance is None:
                print("MyCobot 연결 실패. 로봇 동작은 비활성화됩니다.")

    if yolo_instance is None:
        print("\n YOLO 모델 로딩 실패로 프로그램 실행을 중단합니다.")
        return 
    
    current_block_color = None
    consistent_color = None
    consistent_start_time = None
    
    try:
        initialize_mycobot(mc_instance)
        
        while True:
            if cv2 is not None:
                if cv2.waitKey(1) == 27: 
                    print("ESC 키 입력으로 프로그램 종료.")
                    break
            
            if current_block_color is None:
                
                if mc_instance:
                    move_mycobot(mc_instance, WORKSPACE)
                    control_gripper(mc_instance, "close")

                detected_color, frame_to_show = capture_and_detect(yolo_instance, consistent_color, consistent_start_time) 
                
                print(f"DEBUG: Detected={detected_color}, Consistent={consistent_color}, Current Time={time.time()}")

                if detected_color is not None and detected_color == consistent_color:
                    
                    elapsed_time = time.time() - consistent_start_time
                    
                    if elapsed_time >= REQUIRED_CONSISTENCY_TIME:
                        current_block_color = detected_color
                        consistent_color = None
                        consistent_start_time = None
                        print(f"Stable recognition confirmed. Processing **{current_block_color}**.")
                    else:
                        print(f"-> Consistent: **{detected_color}**. Time: {elapsed_time:.1f}/{REQUIRED_CONSISTENCY_TIME}s")

                
                elif detected_color is not None and detected_color != consistent_color:
                    consistent_color = detected_color
                    consistent_start_time = time.time()
                    print(f"-> New consistency check started for **{detected_color}**.")
                    
                else:
                    if consistent_color is not None:
                        print("-> Consistency lost. Resetting timer.")
                    consistent_color = None
                    consistent_start_time = None
                    print("No recognizable block. Waiting...")
                
                time.sleep(0.5) 
                continue 

            if current_block_color is not None:
                perform_pick_and_place(mc_instance, current_block_color)
                current_block_color = None 
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Automation failed: {e}")
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows() 
        print("\n--- Automation process terminated. ---")

if __name__ == "__main__":
    main_loop()