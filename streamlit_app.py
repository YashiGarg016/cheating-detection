import streamlit as st
import cv2
import joblib
from ultralytics import YOLO
import mediapipe as mp
from model_utils import (
    run_cheating_prediction,
    TwoDimSmoother,
    HeadPoseSmoother
)

# -------------------- Load models --------------------
phone_model = YOLO("yolov8n.pt")
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# -------------------- UI Layout --------------------
st.title("Cheating Detector 📵👀")

# Centered Start Button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
start_tracking = st.button("▶️ Start Tracking", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Layout for video + stats
col1, col2 = st.columns([2, 1])
FRAME_WINDOW = col1.image([])
with col2:
    prediction_text = st.empty()
    probability_text = st.empty()
    warning_text = st.empty()

# -------------------- Tracking Logic --------------------
if start_tracking:
    clf = joblib.load("gaze_model.pkl")
    kalman_left = TwoDimSmoother()
    kalman_right = TwoDimSmoother()
    head_smoother = HeadPoseSmoother()
    away_start_time = None

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)

        pred, prob, warning, away_start_time = run_cheating_prediction(
            frame, face_mesh, phone_model, clf,
            kalman_left, kalman_right, head_smoother, away_start_time
        )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prediction_text.markdown(f"### Prediction: **{'DISTRACTED' if pred == 1 else 'FOCUSED'}**")
        probability_text.markdown(f"**Probability of distraction:** `{prob:.2f}`")
        if warning:
            warning_text.warning(warning)

    cap.release()