import importlib
import math
import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration

# 安全動態載入 mediapipe solutions 模組
mp = importlib.import_module("mediapipe")
mp_drawing = importlib.import_module("mediapipe.solutions.drawing_utils")
mp_pose = importlib.import_module("mediapipe.solutions.pose")

# 計算關節角度函式
def calculate_angle(a, b, c):
    a = np.array(a)  # 首節點 (例如：髖關節)
    b = np.array(b)  # 中間點 (例如：膝蓋)
    c = np.array(c)  # 尾節點 (例如：腳踝)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# WebRTC 影像處理器
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = None  # "up" 或 "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 轉換顏色空間給 MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 取得左側關節座標 (以左側為主要判定點)
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # 計算膝蓋角度
            angle = calculate_angle(hip, knee, ankle)
            
            # 深蹲計數邏輯 (角度低於 90 度視為深蹲，高於 160 度視為站直)
            if angle < 90:
                self.stage = "down"
            if angle > 160 and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            # 繪製人體骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # 在影像畫面上印出次數與狀態
            cv2.rectangle(img, (0, 0), (250, 80), (245, 117, 16), -1)
            cv2.putText(img, f'REPS: {self.counter}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'STAGE: {self.stage}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# Streamlit UI 設定
st.set_page_config(page_title="AI 深蹲計數器", layout="centered")
st.title("🏋️‍♂️ AI 深蹲計數器")
st.info("目前選擇：深蹲 (Squat) - 雙側追蹤。請確保全身入鏡，完成深蹲時將自動計數。")

# WebRTC 伺服器設定 (使用 Google 公開 STUN 伺服器以確保穿透力)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="squat-pose",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)