import cv2
import mediapipe as mp
import numpy as np
import os
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

        # Eye region landmarks (outline of the eye)
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris/Pupil center landmarks
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Smoothing buffer
        self.buffer_size = buffer_size
        self.pupil_buffer = deque(maxlen=buffer_size)
        
        # Thresholds for direction detection (as percentage of eye width/height)
        self.h_threshold = 0.15  # 15% from center
        self.v_threshold = 0.10  # 20% from center

    def extract_eye_region(self, frame, landmarks, eye_indices, img_width, img_height):
        """Extract the eye region as white background with pupil tracking."""
        # Get eye landmark points
        points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * img_width)
            y = int(landmarks[idx].y * img_height)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(points)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_width - x, w + 2 * padding)
        h = min(img_height - y, h + 2 * padding)
        
        # Extract eye region
        eye_roi = frame[y:y+h, x:x+w].copy()
        
        return eye_roi, (x, y, w, h), points

    def get_iris_center(self, landmarks, iris_indices):
        """Get the center of the iris (pupil approximation)."""
        iris_points = []
        for idx in iris_indices:
            x = landmarks[idx].x
            y = landmarks[idx].y
            iris_points.append([x, y])
        
        iris_array = np.array(iris_points)
        center = np.mean(iris_array, axis=0)
        return center

    def calculate_pupil_position(self, pupil_center, eye_bbox):
        """Calculate pupil position relative to eye white area."""
        eye_x, eye_y, eye_w, eye_h = eye_bbox
        
        # Calculate center of eye region
        eye_center_x = eye_x + eye_w / 2
        eye_center_y = eye_y + eye_h / 2
        
        # Calculate relative position (-1 to 1 range)
        rel_x = (pupil_center[0] - eye_center_x) / (eye_w / 2)
        rel_y = (pupil_center[1] - eye_center_y) / (eye_h / 2)
        
        return rel_x, rel_y

    def determine_gaze_direction(self, left_pos, right_pos):
        """Determine gaze direction based on pupil positions."""
        # Average both eyes
        avg_x = (left_pos[0] + right_pos[0]) / 2
        avg_y = (left_pos[1] + right_pos[1]) / 2
        
        # Apply smoothing
        self.pupil_buffer.append((avg_x, avg_y))
        if len(self.pupil_buffer) > 1:
            smoothed = np.mean(list(self.pupil_buffer), axis=0)
            avg_x, avg_y = smoothed[0], smoothed[1]
        
        # Determine horizontal direction
        if avg_x < -self.h_threshold:
            horizontal = "LEFT"
        elif avg_x > self.h_threshold:
            horizontal = "RIGHT"
        else:
            horizontal = "CENTER"
        
        # Determine vertical direction
        if avg_y < -self.v_threshold:
            vertical = "UP"
        elif avg_y > self.v_threshold:
            vertical = "DOWN"
        else:
            vertical = "CENTER"
        
        # Combine directions
        if horizontal == "CENTER" and vertical == "CENTER":
            return "CENTER", avg_x, avg_y
        elif horizontal == "CENTER":
            return vertical, avg_x, avg_y
        elif vertical == "CENTER":
            return horizontal, avg_x, avg_y
        else:
            return f"{vertical}-{horizontal}", avg_x, avg_y

    def draw_eye_visualization(self, image, eye_bbox, pupil_pixel, rel_pos, eye_label):
        """Draw visualization showing white eye area with black pupil position."""
        x, y, w, h = eye_bbox
        
        # Draw eye bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw center point of eye
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(image, (center_x, center_y), 3, (255, 0, 0), -1)
        
        # Draw pupil position (black dot)
        cv2.circle(image, pupil_pixel, 5, (0, 0, 0), -1)
        cv2.circle(image, pupil_pixel, 6, (255, 255, 0), 2)
        
        # Draw line from center to pupil
        cv2.line(image, (center_x, center_y), pupil_pixel, (255, 0, 255), 2)
        
        # Add position text
        pos_text = f"{eye_label}: ({rel_pos[0]:.2f}, {rel_pos[1]:.2f})"
        cv2.putText(image, pos_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def create_eye_diagram(self, width, height, rel_x, rel_y, label):
        """Create a simple diagram showing eye white and pupil position."""
        diagram = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw eye outline (ellipse)
        center = (width // 2, height // 2)
        axes = (width // 2 - 5, height // 2 - 5)
        cv2.ellipse(diagram, center, axes, 0, 0, 360, (0, 0, 0), 2)
        
        # Calculate pupil position
        pupil_x = int(center[0] + rel_x * axes[0] * 0.7)
        pupil_y = int(center[1] + rel_y * axes[1] * 0.7)
        
        # Draw pupil (black circle)
        cv2.circle(diagram, (pupil_x, pupil_y), 15, (0, 0, 0), -1)
        
        # Add label
        cv2.putText(diagram, label, (5, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return diagram

    def process_frame(self, frame):
        """Process frame and track pupil within eye white."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        h, w = frame.shape[:2]
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            try:
                # Get iris centers (pupils)
                left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS)
                right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS)
                
                # Convert to pixel coordinates
                left_pupil_pixel = (int(left_iris_center[0] * w), 
                                   int(left_iris_center[1] * h))
                right_pupil_pixel = (int(right_iris_center[0] * w), 
                                    int(right_iris_center[1] * h))
                
                # Get eye bounding boxes
                _, left_bbox, left_points = self.extract_eye_region(
                    frame, landmarks, self.LEFT_EYE, w, h)
                _, right_bbox, right_points = self.extract_eye_region(
                    frame, landmarks, self.RIGHT_EYE, w, h)
                
                # Calculate pupil positions relative to eye white
                left_rel_x, left_rel_y = self.calculate_pupil_position(
                    left_pupil_pixel, left_bbox)
                right_rel_x, right_rel_y = self.calculate_pupil_position(
                    right_pupil_pixel, right_bbox)
                
                # Determine gaze direction
                direction, avg_x, avg_y = self.determine_gaze_direction(
                    (left_rel_x, left_rel_y), (right_rel_x, right_rel_y))
                
                # Draw visualizations on main frame
                self.draw_eye_visualization(frame, left_bbox, left_pupil_pixel,
                                           (left_rel_x, left_rel_y), "Left")
                self.draw_eye_visualization(frame, right_bbox, right_pupil_pixel,
                                           (right_rel_x, right_rel_y), "Right")
                
                # Create eye diagrams
                left_diagram = self.create_eye_diagram(120, 80, left_rel_x, left_rel_y, "LEFT")
                right_diagram = self.create_eye_diagram(120, 80, right_rel_x, right_rel_y, "RIGHT")
                
                # Place diagrams on frame
                frame[10:90, 10:130] = left_diagram
                frame[10:90, 140:260] = right_diagram
                
                # Display gaze direction
                direction_color = (0, 255, 0) if direction == "CENTER" else (0, 255, 255)
                cv2.putText(frame, f"GAZE: {direction}", (20, h - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, direction_color, 3)
                
                # Display position values
                cv2.putText(frame, f"Position: X={avg_x:.2f}, Y={avg_y:.2f}",
                           (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255, 255, 255), 2)
                
                return frame, True
                
            except Exception as e:
                print(f"Processing error: {e}")
                cv2.putText(frame, f"Error: {str(e)[:40]}", (20, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return frame, False
        else:
            cv2.putText(frame, "NO FACE DETECTED", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame, False

    def process_video(self, input_path, output_path=None, display=True):
        """Process video file."""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open {input_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frame, detected = self.process_frame(frame)
            
            if detected:
                detection_count += 1
            
            if out:
                out.write(processed_frame)
            
            if display:
                cv2.imshow('Pupil Gaze Tracker', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        detection_rate = (detection_count / frame_count) * 100
        print(f"\n=== Processing Complete ===")
        print(f"Detection rate: {detection_rate:.1f}% ({detection_count}/{frame_count})")
        
        return True


def main():
    """Main function."""
    input_video = "eye_ball_2.mp4"
    output_path = "pupil_gaze_tracked_output.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found.")
        return
    
    tracker = PupilGazeTracker(buffer_size=5)
    
    print("Starting Pupil-Based Gaze Tracking...")
    print(f"Input: {input_video}")
    print(f"Output: {output_path}")
    print("Method: Tracking black pupil on white eye background")
    print("Press 'q' to quit\n")
    
    success = tracker.process_video(input_video, output_path, display=True)
    
    if success:
        print(f"\nOutput saved to: {output_path}")
    else:
        print("\nProcessing failed.")


if __name__ == "__main__":
    main()