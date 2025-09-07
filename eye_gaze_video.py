import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Indices for eyes and iris landmarks (MediaPipe Face Mesh)
LEFT_EYE_IDX = {
    "left_corner": 33,
    "right_corner": 133,
    "top": 159,
    "bottom": 145,
}
RIGHT_EYE_IDX = {
    "left_corner": 362,
    "right_corner": 263,
    "top": 386,
    "bottom": 374,
}
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473


def get_eye_box_and_iris(face_landmarks, w, h, eye_idx_map, iris_idx):
    try:
        left_corner = face_landmarks.landmark[eye_idx_map["left_corner"]]
        right_corner = face_landmarks.landmark[eye_idx_map["right_corner"]]
        top = face_landmarks.landmark[eye_idx_map["top"]]
        bottom = face_landmarks.landmark[eye_idx_map["bottom"]]
        iris = face_landmarks.landmark[iris_idx]
    except Exception:
        return None, None, None, None

    # pixel box (for drawing)
    xs = [left_corner.x, right_corner.x, top.x, bottom.x]
    ys = [left_corner.y, right_corner.y, top.y, bottom.y]
    x_min, x_max = int(min(xs) * w), int(max(xs) * w)
    y_min, y_max = int(min(ys) * h), int(max(ys) * h)
    iris_px = (int(iris.x * w), int(iris.y * h))

    # normalized coordinates by eye corners
    denom_x = right_corner.x - left_corner.x
    denom_y = bottom.y - top.y

    if abs(denom_x) < 1e-6 or abs(denom_y) < 1e-6:
        return (x_min, y_min, x_max, y_max), iris_px, None, None

    rx = (iris.x - left_corner.x) / denom_x
    ry = (iris.y - top.y) / denom_y

    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))

    # Debug print
    print(
        f"DEBUG corners Lx={left_corner.x:.3f}, Rx={right_corner.x:.3f}, "
        f"IrisX={iris.x:.3f} => rx={rx:.3f}, ry={ry:.3f}"
    )

    return (x_min, y_min, x_max, y_max), iris_px, rx, ry


def gaze_from_iris_in_box(iris, box):
    (x_min, y_min, x_max, y_max) = box
    ix, iy = iris
    rx = (ix - x_min) / (x_max - x_min + 1e-6)
    ry = (iy - y_min) / (y_max - y_min + 1e-6)
    return rx, ry


prev_states = {"L": ("Center", "Center"), "R": ("Center", "Center")}


def infer_direction(rx, ry, label, th_x=0.18, th_y=0.15):
    h_dir, v_dir = prev_states[label]

    # Horizontal
    if rx < 0.5 - th_x:
        h_dir = "Left"
    elif rx > 0.5 + th_x:
        h_dir = "Right"
    elif 0.45 <= rx <= 0.55:
        h_dir = "Center"

    # Vertical
    if ry < 0.5 - th_y:
        v_dir = "Up"
    elif ry > 0.5 + th_y:
        v_dir = "Down"
    elif 0.45 <= ry <= 0.55:
        v_dir = "Center"

    prev_states[label] = (h_dir, v_dir)

    print(
        f"Horizontal Direction: {h_dir}, Vertical Direction: {v_dir}\n"
        f"Eye: {label} | rx: {rx:.3f}, ry: {ry:.3f}, "
        f"rx_center: 0.5, ThresholdX: {th_x}, ThresholdY: {th_y}"
    )
    return h_dir, v_dir


def main():
    # Use the video file path here instead of the webcam
    video_path = "eye_ball_2.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    # LEFT EYE
                    box, iris, rx_land, ry_land = get_eye_box_and_iris(
                        lm, w, h, LEFT_EYE_IDX, LEFT_IRIS_IDX
                    )
                    if box and iris:
                        if rx_land is None:
                            rx, ry = gaze_from_iris_in_box(iris, box)
                        else:
                            rx, ry = rx_land, ry_land
                        # Uncomment if mirrored:
                        # rx = 1.0 - rx
                        infer_direction(rx, ry, "L")

                    # RIGHT EYE
                    box, iris, rx_land, ry_land = get_eye_box_and_iris(
                        lm, w, h, RIGHT_EYE_IDX, RIGHT_IRIS_IDX
                    )
                    if box and iris:
                        if rx_land is None:
                            rx, ry = gaze_from_iris_in_box(iris, box)
                        else:
                            rx, ry = rx_land, ry_land
                        # Uncomment if mirrored:
                        # rx = 1.0 - rx
                        infer_direction(rx, ry, "R")

            cv2.imshow("Gaze Tracker", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
