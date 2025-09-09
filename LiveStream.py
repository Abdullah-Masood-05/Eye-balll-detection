"""
face_mesh_live.py
Real-time webcam Face Mesh with MediaPipe + OpenCV.
- Draws the face mesh with landmarks.
- Hover mouse over the face: shows the nearest landmark's index and coordinates.
- Press 'q' to quit.

Requirements:
    pip install mediapipe opencv-python numpy
"""

import cv2
import numpy as np
import time
import math
import mediapipe as mp

# ---- Config ----
CAM_INDEX = 0            # change if you have multiple cameras
MAX_HOVER_DIST = 30      # pixels: how close mouse must be to a landmark to be shown
SHOW_EVERY_N_LABEL = 20  # show text label for every N-th landmark
RESIZE_WIDTH = 960       # window resize width

# ---- MediaPipe setup ----
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---- Globals for mouse hover ----
hover_x, hover_y = None, None

def mouse_move(event, x, y, flags, param):
    """Track mouse position over the video window."""
    global hover_x, hover_y
    if event == cv2.EVENT_MOUSEMOVE:
        hover_x, hover_y = x, y

# ---- Helper functions ----
def pixel_coords_from_landmark(landmark, frame_w, frame_h):
    """Convert normalized landmark to pixel coords (x,y) and keep z (relative)."""
    x_pix = int(landmark.x * frame_w)
    y_pix = int(landmark.y * frame_h)
    z_rel = landmark.z
    return x_pix, y_pix, z_rel

def find_closest_landmark(landmarks_px, mx, my):
    """Return (index, (x,y,z), distance) of the landmark closest to mouse (mx,my)."""
    if mx is None or my is None:
        return None, None, None
    best_idx, best_dist = None, float("inf")
    for i, (x, y, z) in enumerate(landmarks_px):
        d = math.hypot(x - mx, y - my)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_idx is None:
        return None, None, None
    return best_idx, landmarks_px[best_idx], best_dist

# ---- Main loop ----
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {CAM_INDEX}. Try changing CAM_INDEX in the script.")
        return

    # Get original capture size
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    scale = RESIZE_WIDTH / orig_w
    target_w, target_h = int(orig_w * scale), int(orig_h * scale)

    cv2.namedWindow("FaceMesh Live", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("FaceMesh Live", mouse_move)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Frame capture failed, skipping...")
                continue

            # resize for performance
            frame = cv2.resize(frame, (target_w, target_h))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)  # ✅ fix for Python 3.12 mediapipe

            results = face_mesh.process(frame_rgb)

            # draw mesh
            if results.multi_face_landmarks:
                landmarks_px = []
                for face_landmarks in results.multi_face_landmarks:
                    # Draw mesh connections
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    try:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )
                    except Exception:
                        pass

                    # Collect landmarks in pixel coords
                    for lm in face_landmarks.landmark:
                        landmarks_px.append(pixel_coords_from_landmark(lm, target_w, target_h))

                    # draw landmark points
                    for i, (x, y, z) in enumerate(landmarks_px):
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        if (i % SHOW_EVERY_N_LABEL) == 0:
                            cv2.putText(frame, str(i), (x+2, y-2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                # hover detection
                idx, lm_px, dist = find_closest_landmark(landmarks_px, hover_x, hover_y)
                if idx is not None and dist <= MAX_HOVER_DIST:
                    x, y, z = lm_px
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    info_lines = [f"idx: {idx}", f"x: {x}px", f"y: {y}px", f"z: {z:.4f}"]
                    # draw info box
                    box_x, box_y = 10, 10
                    line_h, box_w = 18, 170
                    box_h = line_h * len(info_lines) + 8
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
                    for i, line in enumerate(info_lines):
                        cv2.putText(frame, line, (box_x+8, box_y+16 + i*line_h),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # FPS
            cur_time = time.time()
            dt = cur_time - prev_time
            prev_time = cur_time
            if dt > 0:
                fps = 0.9*fps + 0.1*(1.0/dt) if fps else (1.0/dt)
            cv2.putText(frame, f"FPS: {fps:.1f}", (target_w-110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # instructions
            cv2.putText(frame, "Hover mouse over face to inspect values. Press 'q' to quit.",
                        (10, target_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            cv2.imshow("FaceMesh Live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    main()
