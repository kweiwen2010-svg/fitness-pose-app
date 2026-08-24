import json
import math
import os
import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import streamlit.components.v1 as components

# 設定網頁標題與排版
st.set_page_config(
    page_title="智慧運動姿勢輔助 App", page_icon="💪", layout="centered"
)

# 定義個資儲存的檔案路徑
USER_DATA_FILE = "user_profile.json"


def load_user_data():
  """從本地載入個人資料"""
  if os.path.exists(USER_DATA_FILE):
    try:
      with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return None
  return None


def save_user_data(data):
  """將個人資料儲存到本地"""
  with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


def calculate_angle(a, b, c):
  """計算三個點之間的夾角"""
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
      a[1] - b[1], a[0] - b[0]
  )
  angle = np.abs(radians * 180.0 / np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle


# 初始化 Session State
if "user_data" not in st.session_state:
  st.session_state.user_data = load_user_data()

# 用於計數與狀態控制的全域變數
squat_count = 0
squat_stage = None
last_spoken_count = 0  # 用於避免語音重複觸發

# 網頁端語音播放輔助元件 (透過 JavaScript Web Speech API)
def speak_text(text):
  js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang = 'zh-TW';
        window.speechSynthesis.speak(msg);
    </script>
    """
  components.html(js_code, height=0, width=0)


# 標題區
st.title("💪 智慧運動姿勢輔助系統 (語音教練版)")
st.markdown("歡迎使用！系統已啟用**語音即時播報**與**後鏡頭支援**。")

# --- 步驟一：個人資料管理區 ---
with st.expander("📋 檢視與修改個人基本資料", expanded=False):
  saved_data = st.session_state.user_data
  with st.form("profile_form"):
    default_name = saved_data.get("name", "") if saved_data else ""
    default_age = int(saved_data.get("age", 25)) if saved_data else 25
    default_height = (
        float(saved_data.get("height", 170.0)) if saved_data else 170.0
    )
    default_weight = (
        float(saved_data.get("weight", 65.0)) if saved_data else 65.0
    )

    name_input = st.text_input("您的稱呼", value=default_name)
    age_input = st.number_input(
        "年齡 (歲)", min_value=1, max_value=120, value=default_age
    )
    height_input = st.number_input(
        "身高 (公分)", min_value=50.0, max_value=250.0, value=default_height
    )
    weight_input = st.number_input(
        "體重 (公斤)", min_value=20.0, max_value=300.0, value=default_weight
    )

    submitted = st.form_submit_button("更新個人資料")

    if submitted:
      new_data = {
          "name": name_input,
          "age": age_input,
          "height": height_input,
          "weight": weight_input,
      }
      save_user_data(new_data)
      st.session_state.user_data = new_data
      st.success("✅ 個人資料已更新！")


# --- 步驟二：選擇訓練動作與啟動鏡頭 ---
st.markdown("---")
st.subheader("🏋️ 即時 AI 姿勢矯正與語音教練")

if st.session_state.user_data:
  user_name = st.session_state.user_data.get("name", "運動員")
  st.write(f"嗨！**{user_name}**，請將手機架好後點擊啟動後鏡頭：")

  action_choice = st.selectbox(
      "支援的動作清單",
      ["深蹲 (Squat) - 雙側追蹤與語音播報", "棒式 (Plank) - 核心穩定"],
  )

  st.info(
      f"目前選擇：**{action_choice}**。"
      "請確保全身入鏡，完成深蹲時手機將會透過語音報數。"
  )

  # MediaPipe 初始化設定
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose


  # 定義影像處理回呼函數
  def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global squat_count, squat_stage, last_spoken_count

    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    with mp_pose.Pose(
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) as pose:
      image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      image_rgb.flags.writeable = False
      results = pose.process(image_rgb)

      image_rgb.flags.writeable = True
      image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

      posture_warning = "POSTURE: GOOD"
      warning_color = (0, 255, 0)

      try:
        landmarks = results.pose_landmarks.landmark

        # 左右腳關節提取與計算
        l_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        l_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        l_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        l_angle = calculate_angle(l_hip, l_knee, l_ankle)

        r_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        r_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        r_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        r_angle = calculate_angle(r_hip, r_knee, r_ankle)

        angle = (l_angle + r_angle) / 2

        # 軀幹前傾檢查
        l_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        torso_angle = calculate_angle(l_shoulder, l_hip, [l_hip[0], l_hip[1] + 1])
        if torso_angle > 45:
          posture_warning = "WARNING: BEND FORWARD!"
          warning_color = (0, 0, 255)

        # 深蹲狀態機與計數
        if angle > 160:
          squat_stage = "UP"
        if angle < 95 and squat_stage == "UP":
          squat_stage = "DOWN"
          squat_count += 1

      except Exception:
        pass

      # 當次數增加時，在主畫面非同步觸發語音提示
      if squat_count > last_spoken_count:
        speak_text(str(squat_count))
        last_spoken_count = squat_count

      # 畫面資訊欄繪製
      cv2.rectangle(image_bgr, (0, 0), (240, 80), (40, 40, 40), -1)
      cv2.putText(
          image_bgr, "REPS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
      )
      cv2.putText(
          image_bgr, str(squat_count), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
      )

      cv2.putText(
          image_bgr, "STAGE", (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
      )
      cv2.putText(
          image_bgr, str(squat_stage), (110, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
      )

      # 畫面下方警告列
      cv2.rectangle(image_bgr, (0, h - 45), (w, h), (40, 40, 40), -1)
      cv2.putText(
          image_bgr, posture_warning, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warning_color, 2, cv2.LINE_AA
      )

      # 繪製骨架
      if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(0, 255, 128), thickness=2, circle_radius=2
            ),
            mp_drawing.DrawingSpec(
                color=(255, 0, 128), thickness=2, circle_radius=2
            ),
        )

    return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


  # 啟動 WebRTC 即時串流元件 (已改用後鏡頭 environment)
  webrtc_streamer(
      key="fitness-pose-voice",
      mode=WebRtcMode.SENDRECV,
      video_frame_callback=video_frame_callback,
      rtc_configuration={
          "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
      },
      media_stream_constraints={
          "video": {"facingMode": "environment"},  # 使用後鏡頭
          "audio": False,
      },
  )

else:
  st.warning("⚠️ 請先設定個人資料才能解鎖鏡頭功能！")