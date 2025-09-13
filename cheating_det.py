import cv2
import mediapipe as mp
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
import time

# ----------------- CONFIG -----------------
YOLO_WEIGHTS = "yolov8s.pt"    # better than nano for accuracy
CONF_THRESHOLD = 0.5
MIN_BOX_AREA = 2000
PERSON_CONF_THRESH = 0.5

FRAME_THRESHOLD = 5            # persistence frames to trigger alert
CALIBRATE_FRAMES = 30          # frames to compute neutral baseline
SMOOTHING_WINDOW = 8           # smoothing window for angles
GAZE_THRESHOLD_DEG = 20        # degrees away from baseline to consider "looking away"
MULTI_PERSON_THRESHOLD = 1
EYE_AWAY_SECONDS = 1.0         # how long eyes can be away before alert (horizontal)
# Eye margin multipliers (tune these)
H_MARGIN = 0.35   # horizontal tolerance (0.35 is fairly wide)
V_MARGIN = 0.40   # vertical tolerance (0.25 avoids small up/down noise)
# ------------------------------------------

# load models
yolo = YOLO(YOLO_WEIGHTS)

# MediaPipe: lightweight face detector (for counting) + face mesh for head pose/eyes
mp_face_det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              refine_landmarks=True,
                                              min_detection_confidence=0.6,
                                              min_tracking_confidence=0.6)

# Head-pose landmark indices
LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# Eye landmarks (MediaPipe FaceMesh) - iris centers 468 & 473
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473
# left/right eye corner pairs (for horizontal)
LEFT_EYE_CORNER_IDX = (33, 133)
RIGHT_EYE_CORNER_IDX = (362, 263)
# Use proper top/bottom eyelid landmarks for vertical bounds
LEFT_EYE_TOP_IDX = 159
LEFT_EYE_BOTTOM_IDX = 145
RIGHT_EYE_TOP_IDX = 386
RIGHT_EYE_BOTTOM_IDX = 374

# 3D model points of the face
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye outer
    (43.3, 32.7, -26.0),    # Right eye outer
    (-28.9, -28.9, -24.1),  # Left mouth corner
    (28.9, -28.9, -24.1)    # Right mouth corner
], dtype=np.float64)

# helpers
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(np.array([x, y, z]))

# smoothing deques & calibration accumulators
yaw_buf = deque(maxlen=SMOOTHING_WINDOW)
pitch_buf = deque(maxlen=SMOOTHING_WINDOW)
calib_yaws = []
calib_pitches = []
calibrated = False
calib_start_time = None
baseline_yaw = 0.0
baseline_pitch = 0.0

# counters for consistency filter
event_counters = defaultdict(int)

# Eye tracking state
last_eyes_on_screen_time = time.time()
last_eyes_vertical_time = time.time()
# We'll also keep short booleans to indicate state
eyes_currently_on_screen = False
eyes_vertical_currently_ok = True

# camera
cap = cv2.VideoCapture(0)
time.sleep(0.5)

print("Auto-calibration will start once a face is detected and run for", CALIBRATE_FRAMES, "frames.")
print("Press 'q' to quit (or ESC). Press 'c' to recalibrate at any time.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- fast face detection (for counting multiple faces) ---
    face_det_res = mp_face_det.process(frame_rgb)
    num_faces = 0
    if face_det_res.detections:
        num_faces = len(face_det_res.detections)
        for det in face_det_res.detections:
            bbox = det.location_data.relative_bounding_box
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

    # --- Face mesh & head pose (primary face only) ---
    mesh_res = mp_face_mesh.process(frame_rgb)
    head_pose_alert = False
    yaw = None; pitch = None; roll = None
    eyes_detected = False
    # Reset per-frame defaults (don't override global last timestamps)
    eyes_on_screen = False

    if mesh_res.multi_face_landmarks and len(mesh_res.multi_face_landmarks) > 0:
        face_lms = mesh_res.multi_face_landmarks[0]
        image_points = []
        for idx in LANDMARK_IDS:
            lm = face_lms.landmark[idx]
            px, py = float(lm.x * w), float(lm.y * h)
            image_points.append((px, py))
        image_points = np.array(image_points, dtype=np.float64)

        # camera intrinsics approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                # smoothing buffers
                yaw_buf.append(yaw)
                pitch_buf.append(pitch)

                if not calibrated:
                    calib_yaws.append(yaw)
                    calib_pitches.append(pitch)
                    if calib_start_time is None:
                        calib_start_time = time.time()
                    cv2.putText(frame, f"Calibrating head pose: {len(calib_yaws)}/{CALIBRATE_FRAMES}", (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if len(calib_yaws) >= CALIBRATE_FRAMES:
                        baseline_yaw = float(np.mean(calib_yaws))
                        baseline_pitch = float(np.mean(calib_pitches))
                        calibrated = True
                        print(f"[Calibration done] baseline_yaw={baseline_yaw:.2f}, baseline_pitch={baseline_pitch:.2f}")
                else:
                    smooth_yaw = float(np.mean(yaw_buf)) if len(yaw_buf) > 0 else yaw
                    smooth_pitch = float(np.mean(pitch_buf)) if len(pitch_buf) > 0 else pitch
                    if abs(smooth_yaw - baseline_yaw) > GAZE_THRESHOLD_DEG or abs(smooth_pitch - baseline_pitch) > GAZE_THRESHOLD_DEG:
                        head_pose_alert = True
                    cv2.putText(frame, f"Yaw:{smooth_yaw:.1f} Pitch:{smooth_pitch:.1f}", (10, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            # --- Eye tracking using iris landmarks (468 & 473) with robust vertical logic ---
            try:
                # get iris coords (pixel)
                left_iris_lm = face_lms.landmark[LEFT_IRIS_IDX]
                right_iris_lm = face_lms.landmark[RIGHT_IRIS_IDX]
                left_iris = (left_iris_lm.x * w, left_iris_lm.y * h)
                right_iris = (right_iris_lm.x * w, right_iris_lm.y * h)

                # get eye-corner coords (pixel) for horizontal range
                l_corner_left = face_lms.landmark[LEFT_EYE_CORNER_IDX[0]]
                l_corner_right = face_lms.landmark[LEFT_EYE_CORNER_IDX[1]]
                r_corner_left = face_lms.landmark[RIGHT_EYE_CORNER_IDX[0]]
                r_corner_right = face_lms.landmark[RIGHT_EYE_CORNER_IDX[1]]

                l_c1 = (l_corner_left.x * w, l_corner_left.y * h)
                l_c2 = (l_corner_right.x * w, l_corner_right.y * h)
                r_c1 = (r_corner_left.x * w, r_corner_left.y * h)
                r_c2 = (r_corner_right.x * w, r_corner_right.y * h)

                # horizontal bounds and widths (pixels)
                l_min_x, l_max_x = min(l_c1[0], l_c2[0]), max(l_c1[0], l_c2[0])
                r_min_x, r_max_x = min(r_c1[0], r_c2[0]), max(r_c1[0], r_c2[0])
                l_width, r_width = max(1.0, l_max_x - l_min_x), max(1.0, r_max_x - r_min_x)

                # --- robust vertical bounds using top/bottom eyelid landmarks (NOT corner Ys) ---
                lt = face_lms.landmark[LEFT_EYE_TOP_IDX]
                lb = face_lms.landmark[LEFT_EYE_BOTTOM_IDX]
                rt = face_lms.landmark[RIGHT_EYE_TOP_IDX]
                rb = face_lms.landmark[RIGHT_EYE_BOTTOM_IDX]

                l_min_y = min(lt.y * h, lb.y * h)
                l_max_y = max(lt.y * h, lb.y * h)
                r_min_y = min(rt.y * h, rb.y * h)
                r_max_y = max(rt.y * h, rb.y * h)

                l_height = max(1.0, l_max_y - l_min_y)
                r_height = max(1.0, r_max_y - r_min_y)

                # horizontal checks (with margin H_MARGIN)
                left_ok_x = (l_min_x + H_MARGIN * l_width) <= left_iris[0] <= (l_max_x - H_MARGIN * l_width)
                right_ok_x = (r_min_x + H_MARGIN * r_width) <= right_iris[0] <= (r_max_x - H_MARGIN * r_width)

                # vertical checks (with margin V_MARGIN), using top/bottom eyelid landmarks
                left_ok_y = (l_min_y + V_MARGIN * l_height) <= left_iris[1] <= (l_max_y - V_MARGIN * l_height)
                right_ok_y = (r_min_y + V_MARGIN * r_height) <= right_iris[1] <= (r_max_y - V_MARGIN * r_height)

                # if eye height is extremely small (unstable landmarks), be forgiving
                if l_height < 3:
                    left_ok_y = True
                if r_height < 3:
                    right_ok_y = True

                # final decisions
                eyes_on_screen = (left_ok_x and right_ok_x and left_ok_y and right_ok_y)
                eyes_vertical_ok = (left_ok_y and right_ok_y)  # vertical-only check

                # update states & timestamps
                if eyes_on_screen:
                    eyes_detected = True
                    eyes_currently_on_screen = True
                    eyes_vertical_currently_ok = True
                    last_eyes_on_screen_time = time.time()
                    last_eyes_vertical_time = time.time()
                    # draw debug dots
                    cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 3, (0, 255, 255), -1)
                    cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 3, (0, 255, 255), -1)
                else:
                    # iris present but outside central band
                    eyes_detected = True
                    # horizontal status:
                    if (left_ok_x and right_ok_x):
                        # horizontally ok but maybe vertical off
                        eyes_currently_on_screen = True
                        last_eyes_on_screen_time = time.time()
                    else:
                        eyes_currently_on_screen = False
                    # vertical status:
                    if eyes_vertical_ok:
                        eyes_vertical_currently_ok = True
                        last_eyes_vertical_time = time.time()
                    else:
                        eyes_vertical_currently_ok = False

                    # small debug markers
                    cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 2, (0, 120, 255), -1)
                    cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 2, (0, 120, 255), -1)

            except Exception:
                # if iris/corners/top/bottom missing, treat as not detected (don't change last_eyes timestamps)
                eyes_detected = False
                eyes_currently_on_screen = False
                eyes_vertical_currently_ok = True  # avoid vertical false alert when landmarks missing

        except Exception:
            pass
    else:
        # no mesh results
        eyes_detected = False
        eyes_currently_on_screen = False
        eyes_vertical_currently_ok = True

    # --- Eye away logic: horizontal & vertical alerts ---
    eye_alert = False
    eye_vertical_alert = False

    # horizontal (left/right) alert
    if not eyes_currently_on_screen:
        if time.time() - last_eyes_on_screen_time > EYE_AWAY_SECONDS:
            eye_alert = True
    else:
        event_counters["eye_away"] = 0  # immediate reset when eyes back

    # vertical (up/down) alert
    if not eyes_vertical_currently_ok:
        if time.time() - last_eyes_vertical_time > EYE_AWAY_SECONDS:
            eye_vertical_alert = True
    else:
        event_counters["eye_vertical"] = 0  # reset vertical counter when okay

    # --- YOLO: persons & suspicious items ---
    yolo_results = yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    person_count = 0
    suspicious_found = []
    for res in yolo_results:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id].lower()
            conf = float(box.conf[0])
            if conf < PERSON_CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            if label == "person":
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if label in {"cell phone", "book", "laptop", "tv", "monitor"}:
                suspicious_found.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Susp:{label}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # --- Build current events & update counters ---
    current_events = []
    if num_faces == 0:
        current_events.append("no_face")
    elif num_faces > 1:
        current_events.append("multi_face")

    if person_count == 0:
        current_events.append("no_person")
    elif person_count > MULTI_PERSON_THRESHOLD:
        current_events.append("multi_person")

    if suspicious_found:
        current_events.append("suspicious")

    if head_pose_alert:
        current_events.append("head_pose")

    if eye_alert:
        current_events.append("eye_away")

    if eye_vertical_alert:
        current_events.append("eye_vertical")

    # update counters
    for ev in ["no_face", "multi_face", "no_person", "multi_person", "suspicious", "head_pose", "eye_away", "eye_vertical"]:
        if ev in current_events:
            event_counters[ev] += 1
        else:
            event_counters[ev] = 0

    # --- Trigger alerts ---
    if event_counters["no_face"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if event_counters["multi_face"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: Multiple faces", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if event_counters["no_person"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: No person", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if event_counters["multi_person"] >= FRAME_THRESHOLD:
        cv2.putText(frame, f"ALERT: Multiple persons ({person_count})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if event_counters["suspicious"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: Suspicious object", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if event_counters["head_pose"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: Looking away (head)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if event_counters["eye_away"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: Eyes not on screen (left/right)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if event_counters["eye_vertical"] >= FRAME_THRESHOLD:
        cv2.putText(frame, "ALERT: Eyes looking up/down", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(frame, f"Faces:{num_faces} Persons:{person_count}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Exam Monitor (press q to quit, c to recalibrate)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        break
    if key == ord('c') or key == ord('C'):
        calibrated = False
        calib_yaws.clear()
        calib_pitches.clear()
        calib_start_time = None
        last_eyes_on_screen_time = time.time()
        last_eyes_vertical_time = time.time()
        eyes_currently_on_screen = False
        eyes_vertical_currently_ok = True
        print("[Manual] Recalibration started...")

cap.release()
cv2.destroyAllWindows()
