# model_utils.py

import cv2
import numpy as np
import time

# -------------------- Constants --------------------
POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]
LEFT_EYE = [33, 133, 159, 145]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_EYE = [362, 263, 386, 374]
RIGHT_IRIS = [473, 474, 475, 476]
PHONE_CLASSES = ["cell phone", "mobile phone"]
DIR_MAP = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3, "CENTER": 4}
AWAY_THRESHOLD = 0.8

# -------------------- Smoothers --------------------
class TwoDimSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.x = None
        self.y = None
    def apply(self, newx, newy):
        if self.x is None:
            self.x, self.y = newx, newy
        else:
            self.x = (1 - self.alpha) * self.x + self.alpha * newx
            self.y = (1 - self.alpha) * self.y + self.alpha * newy
        return int(self.x), int(self.y)

class HeadPoseSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.pitch = None
        self.yaw = None
        self.roll = None
    def apply(self, pitch, yaw, roll):
        if self.pitch is None:
            self.pitch, self.yaw, self.roll = pitch, yaw, roll
        else:
            self.pitch = (1 - self.alpha) * self.pitch + self.alpha * pitch
            self.yaw   = (1 - self.alpha) * self.yaw   + self.alpha * yaw
            self.roll  = (1 - self.alpha) * self.roll  + self.alpha * roll
        return self.pitch, self.yaw, self.roll

# -------------------- Eye direction --------------------
def get_eye_direction(landmarks, eye_idx, iris_idx, w, h, smoother):
    eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
    iris_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx]
    iris_cx = sum([p[0] for p in iris_pts]) / len(iris_pts)
    iris_cy = sum([p[1] for p in iris_pts]) / len(iris_pts)
    iris_x, iris_y = smoother.apply(iris_cx, iris_cy)

    left, right = min(eye[0][0], eye[1][0]), max(eye[0][0], eye[1][0])
    top, bottom = min(eye[2][1], eye[3][1]), max(eye[2][1], eye[3][1])

    margin_x = int((right - left) * 0.15)
    margin_y = int((bottom - top) * 0.20)

    if iris_x < left + margin_x:
        return "LEFT", iris_x - (left + margin_x), iris_y - (top + margin_y)
    elif iris_x > right - margin_x:
        return "RIGHT", iris_x - (left + margin_x), iris_y - (top + margin_y)
    elif iris_y < top + margin_y:
        return "UP", iris_x - (left + margin_x), iris_y - (top + margin_y)
    elif iris_y > bottom - margin_y:
        return "DOWN", iris_x - (left + margin_x), iris_y - (top + margin_y)
    else:
        return "CENTER", iris_x - (left + margin_x), iris_y - (top + margin_y)

# -------------------- Head pose --------------------
def estimate_head_pose(landmarks, w, h):
    image_points = np.array([(landmarks[idx].x * w, landmarks[idx].y * h) for idx in POSE_LANDMARKS], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, 0.0, -30.0),
        (30.0, 0.0, -30.0),
        (-30.0, 0.0, -90.0),
        (30.0, 0.0, -90.0),
        (0.0, 40.0, -50.0)
    ])
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]])
    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return rotation_vector if success else None

# -------------------- Main prediction wrapper --------------------
def run_cheating_prediction(frame, face_mesh, phone_model, clf, kalman_left, kalman_right, head_smoother, away_start_time):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    left_dir = right_dir = "CENTER"
    left_dx = left_dy = right_dx = right_dy = pitch = yaw = roll = 0
    eyes_away_duration = 0
    phone_detected = 0
    eyes_away = False
    warning = ""

    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        left_dir, left_dx, left_dy = get_eye_direction(landmarks, LEFT_EYE, LEFT_IRIS, w, h, kalman_left)
        right_dir, right_dx, right_dy = get_eye_direction(landmarks, RIGHT_EYE, RIGHT_IRIS, w, h, kalman_right)
        rot_vec = estimate_head_pose(landmarks, w, h)
        if rot_vec is not None:
            pitch, yaw, roll = head_smoother.apply(*rot_vec.ravel())
        eyes_away = left_dir != "CENTER" or right_dir != "CENTER"

    results = phone_model(frame, verbose=False)
    if results and results[0].boxes:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = phone_model.names[cls]
            conf = float(box.conf[0])
            if label in PHONE_CLASSES and conf > 0.5:
                phone_detected = 1
                warning = "⚠️ Mobile Detected!"

    current_time = time.time()
    if eyes_away:
        if away_start_time is None:
            away_start_time = current_time
        eyes_away_duration = current_time - away_start_time
        if eyes_away_duration >= AWAY_THRESHOLD and warning == "":
            warning = "⚠️ Not looking at screen!"
    else:
        away_start_time = None

    features = [[
        DIR_MAP[left_dir],
        DIR_MAP[right_dir],
        left_dx, left_dy,
        right_dx, right_dy,
        pitch, yaw, roll,
        phone_detected,
        eyes_away_duration
    ]]
    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0, 1]

    if pred == 1 and warning == "":
        warning = "⚠️ Predicted distraction"

    return pred, prob, warning, away_start_time