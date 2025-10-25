import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class PupilGazeTracker:
    def __init__(self, buffer_size=5):
        """Initialize MediaPipe Face Mesh for pupil-based gaze tracking."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Eye region landmarks
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                         157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                          388, 387, 386, 385, 384, 398]

        # Iris/Pupil landmarks
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]

        # Smoothing buffer
        self.buffer_size = buffer_size
        self.pupil_buffer = deque(maxlen=buffer_size)

        # Thresholds for direction detection
        self.h_threshold = 0.15  # horizontal sensitivity
        self.v_threshold = 0.20  # vertical sensitivity

    def extract_eye_region(self, frame, landmarks, eye_indices, img_width, img_height):
        """Extract the eye region."""
        points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * img_width)
            y = int(landmarks[idx].y * img_height)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_width - x, w + 2 * padding)
        h = min(img_height - y, h + 2 * padding)
        eye_roi = frame[y:y + h, x:x + w].copy()
        return eye_roi, (x, y, w, h), points

    def get_iris_center(self, landmarks, iris_indices):
        iris_points = np.array([[landmarks[i].x, landmarks[i].y] for i in iris_indices])
        return np.mean(iris_points, axis=0)

    def calculate_pupil_position(self, pupil_center, eye_bbox):
        eye_x, eye_y, eye_w, eye_h = eye_bbox
        eye_center_x = eye_x + eye_w / 2
        eye_center_y = eye_y + eye_h / 2
        rel_x = (pupil_center[0] - eye_center_x) / (eye_w / 2)
        rel_y = (pupil_center[1] - eye_center_y) / (eye_h / 2)
        return rel_x, rel_y

    def determine_gaze_direction(self, left_pos, right_pos):
        avg_x = (left_pos[0] + right_pos[0]) / 2
        avg_y = (left_pos[1] + right_pos[1]) / 2

        # smoothing
        self.pupil_buffer.append((avg_x, avg_y))
        if len(self.pupil_buffer) > 1:
            avg_x, avg_y = np.mean(self.pupil_buffer, axis=0)

        # determine direction
        if avg_x < -self.h_threshold:
            horizontal = "LEFT"
        elif avg_x > self.h_threshold:
            horizontal = "RIGHT"
        else:
            horizontal = "CENTER"

        if avg_y < -self.v_threshold:
            vertical = "UP"
        elif avg_y > self.v_threshold:
            vertical = "DOWN"
        else:
            vertical = "CENTER"

        if horizontal == "CENTER" and vertical == "CENTER":
            return "CENTER", avg_x, avg_y
        elif horizontal == "CENTER":
            return vertical, avg_x, avg_y
        elif vertical == "CENTER":
            return horizontal, avg_x, avg_y
        else:
            return f"{vertical}-{horizontal}", avg_x, avg_y

    def draw_eye_visualization(self, image, eye_bbox, pupil_pixel, rel_pos, eye_label):
        x, y, w, h = eye_bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(image, (center_x, center_y), 3, (255, 0, 0), -1)
        cv2.circle(image, pupil_pixel, 5, (0, 0, 0), -1)
        cv2.circle(image, pupil_pixel, 6, (255, 255, 0), 2)
        cv2.line(image, (center_x, center_y), pupil_pixel, (255, 0, 255), 2)
        text = f"{eye_label}: ({rel_pos[0]:.2f},{rel_pos[1]:.2f})"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    def create_eye_diagram(self, width, height, rel_x, rel_y, label):
        diagram = np.ones((height, width, 3), dtype=np.uint8) * 255
        center = (width // 2, height // 2)
        axes = (width // 2 - 5, height // 2 - 5)
        cv2.ellipse(diagram, center, axes, 0, 0, 360, (0, 0, 0), 2)
        pupil_x = int(center[0] + rel_x * axes[0] * 0.7)
        pupil_y = int(center[1] + rel_y * axes[1] * 0.7)
        cv2.circle(diagram, (pupil_x, pupil_y), 15, (0, 0, 0), -1)
        cv2.putText(diagram, label, (5, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return diagram

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        h, w = frame.shape[:2]

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            try:
                left_center = self.get_iris_center(landmarks, self.LEFT_IRIS)
                right_center = self.get_iris_center(landmarks, self.RIGHT_IRIS)

                left_pixel = (int(left_center[0] * w), int(left_center[1] * h))
                right_pixel = (int(right_center[0] * w), int(right_center[1] * h))

                _, left_bbox, _ = self.extract_eye_region(frame, landmarks, self.LEFT_EYE, w, h)
                _, right_bbox, _ = self.extract_eye_region(frame, landmarks, self.RIGHT_EYE, w, h)

                left_rel = self.calculate_pupil_position(left_pixel, left_bbox)
                right_rel = self.calculate_pupil_position(right_pixel, right_bbox)

                direction, avg_x, avg_y = self.determine_gaze_direction(left_rel, right_rel)

                self.draw_eye_visualization(frame, left_bbox, left_pixel, left_rel, "Left")
                self.draw_eye_visualization(frame, right_bbox, right_pixel, right_rel, "Right")

                left_diag = self.create_eye_diagram(120, 80, left_rel[0], left_rel[1], "LEFT")
                right_diag = self.create_eye_diagram(120, 80, right_rel[0], right_rel[1], "RIGHT")
                frame[10:90, 10:130] = left_diag
                frame[10:90, 140:260] = right_diag

                color = (0, 255, 0) if direction == "CENTER" else (0, 255, 255)
                cv2.putText(frame, f"GAZE: {direction}", (20, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, f"X={avg_x:.2f}, Y={avg_y:.2f}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                return frame, True

            except Exception as e:
                cv2.putText(frame, f"Error: {e}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, False

    def process_live(self):
        """Use webcam for real-time gaze tracking."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("=== Live Pupil Gaze Tracker ===")
        print("Press 'q' to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = self.process_frame(frame)
            cv2.imshow("Live Pupil Gaze Tracker", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    tracker = PupilGazeTracker(buffer_size=5)
    tracker.process_live()


if __name__ == "__main__":
    main()
