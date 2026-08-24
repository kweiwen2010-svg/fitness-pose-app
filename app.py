import math
import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. 解決 MediaPipe 0.10+ solutions 屬性丟失的標準相容寫法
import mediapipe as mp

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
except AttributeError:
    # 針對 Linux 雲端環境手動導入 solutions 模組
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.pose as mp_pose

# 2. 計算關節角度函式
def calculate_angle(a, b, c):
    a = np.array(a)  # 髖關節
    b = np.array(b)  # 膝蓋
    c = np.array(c)  # 腳踝
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 3. WebRTC 影像處理器
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
        
        # 轉換為 RGB 供 MediaPipe 辨識
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 取得左側關鍵點座標
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # 計算膝蓋角度
            angle = calculate_angle(hip, knee, ankle)
            
            # 深蹲動作邏輯判定
            if angle < 90:
                self.stage = "down"
            if angle > 160 and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            # 繪製骨架線條
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # 繪製資訊面板
            cv2.rectangle(img, (0, 0), (250, 80), (245, 117, 16), -1)
            cv2.putText(img, f'REPS: {self.counter}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'STAGE: {self.stage if self.stage else "Ready"}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# 4. Streamlit 介面佈局
st.set_page_config(page_title="AI 深蹲計數器", layout="centered")
st.title("🏋️‍♂️ AI 深蹲計數器")
st.info("目前選擇：深蹲 (Squat) - 雙側追蹤。請確保全身入鏡，完成深蹲時將自動計數。")

# Google STUN 伺服器設定
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="squat-pose-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)