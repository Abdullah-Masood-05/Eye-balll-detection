import cv2
import mediapipe as mp
import numpy as np
import argparse
from collections import deque
from scipy.spatial.distance import euclidean


class EnhancedGazeTracker:
    def __init__(self, buffer_size=5, confidence_threshold=0.7):
        """Initialize MediaPipe Face Mesh and advanced gaze tracking components."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh with higher accuracy settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Enhanced eye landmark indices for better coverage
        self.LEFT_EYE_LANDMARKS = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            130, 25, 110, 24, 23, 22, 26, 112, 243
        ]
        
        self.RIGHT_EYE_LANDMARKS = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            359, 255, 339, 254, 253, 252, 256, 341, 463
        ]
        
        # Iris landmarks with all available points
        self.LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
        
        # Eye corner landmarks for better reference points
        self.LEFT_EYE_CORNERS = [33, 133]  # outer, inner
        self.RIGHT_EYE_CORNERS = [362, 263]  # outer, inner
        
        # Pupil center approximation (iris center is close but we can refine)
        self.LEFT_PUPIL_LANDMARKS = [468]  # Main iris center
        self.RIGHT_PUPIL_LANDMARKS = [473]  # Main iris center
        
        # Enhanced thresholds with adaptive capability
        self.base_horizontal_threshold = 0.12
        self.base_vertical_threshold = 0.10
        self.adaptive_threshold_factor = 1.0
        
        # Smoothing buffer for temporal consistency
        self.buffer_size = buffer_size
        self.gaze_buffer = deque(maxlen=buffer_size)
        self.iris_buffer_left = deque(maxlen=buffer_size)
        self.iris_buffer_right = deque(maxlen=buffer_size)
        
        # Head pose estimation for compensation
        self.head_pose_buffer = deque(maxlen=buffer_size)
        
        # Calibration data (can be enhanced with user-specific calibration)
        self.calibration_data = {
            'personal_threshold_multiplier': 1.0,
            'eye_aspect_ratio_baseline': None,
            'head_pose_compensation': True
        }
        
        # Eye aspect ratio for blink detection and quality assessment
        self.ear_threshold = 0.25
        
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for blink detection and quality assessment."""
        if len(eye_landmarks) < 6:
            return 0.0
            
        # Use the most reliable landmarks for EAR calculation
        eye_points = eye_landmarks[:16] if len(eye_landmarks) >= 16 else eye_landmarks
        
        # Calculate vertical distances
        vertical_1 = euclidean(eye_points[1], eye_points[5])
        vertical_2 = euclidean(eye_points[2], eye_points[4])
        
        # Calculate horizontal distance
        horizontal = euclidean(eye_points[0], eye_points[3])
        
        if horizontal == 0:
            return 0.0
            
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def estimate_head_pose(self, landmarks, img_width, img_height):
        """Estimate head pose using facial landmarks."""
        nose_tip = np.array([landmarks[1].x * img_width, landmarks[1].y * img_height])
        nose_bridge = np.array([landmarks[168].x * img_width, landmarks[168].y * img_height])
        left_eye_corner = np.array([landmarks[33].x * img_width, landmarks[33].y * img_height])
        right_eye_corner = np.array([landmarks[263].x * img_width, landmarks[263].y * img_height])
        left_mouth = np.array([landmarks[61].x * img_width, landmarks[61].y * img_height])
        right_mouth = np.array([landmarks[291].x * img_width, landmarks[291].y * img_height])
        chin = np.array([landmarks[18].x * img_width, landmarks[18].y * img_height])
        
        # Calculate head rotation angles
        eye_center = (left_eye_corner + right_eye_corner) / 2
        mouth_center = (left_mouth + right_mouth) / 2
        
        # Horizontal angle (yaw) - based on eye symmetry
        eye_distance_left = euclidean(nose_tip, left_eye_corner)
        eye_distance_right = euclidean(nose_tip, right_eye_corner)
        yaw = np.arctan2(eye_distance_right - eye_distance_left, 
                        (eye_distance_left + eye_distance_right) / 2) * 180 / np.pi
        
        # Vertical angle (pitch) - based on face vertical alignment
        face_height = euclidean(eye_center, mouth_center)
        nose_to_mouth = euclidean(nose_tip, mouth_center)
        pitch = np.arctan2(nose_to_mouth - face_height/2, face_height) * 180 / np.pi
        
        # Roll angle - based on eye level
        eye_angle = np.arctan2(right_eye_corner[1] - left_eye_corner[1],
                              right_eye_corner[0] - left_eye_corner[0]) * 180 / np.pi
        
        return {'yaw': yaw, 'pitch': pitch, 'roll': eye_angle}
    
    def compensate_for_head_pose(self, relative_position, head_pose):
        """Compensate gaze estimation for head pose."""
        if not self.calibration_data['head_pose_compensation']:
            return relative_position
            
        # Apply head pose compensation
        yaw_compensation = -head_pose['yaw'] * 0.02
        pitch_compensation = -head_pose['pitch'] * 0.015
        
        compensated_x = relative_position[0] + yaw_compensation
        compensated_y = relative_position[1] + pitch_compensation
        
        return (compensated_x, compensated_y)
    
    def get_enhanced_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks with enhanced preprocessing."""
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                x = landmarks[idx].x
                y = landmarks[idx].y
                eye_points.append([x, y])
        
        if not eye_points:
            return np.array([])
            
        return np.array(eye_points)
    
    def get_refined_iris_center(self, landmarks, iris_indices):
        """Calculate refined iris center with weighted averaging."""
        iris_points = []
        weights = [2.0, 1.0, 1.0, 1.0, 1.0]
        
        for i, idx in enumerate(iris_indices):
            if idx < len(landmarks):
                x = landmarks[idx].x
                y = landmarks[idx].y
                weight = weights[i] if i < len(weights) else 1.0
                iris_points.extend([[x, y]] * int(weight))
        
        if not iris_points:
            return np.array([0.5, 0.5])
            
        iris_array = np.array(iris_points)
        return np.mean(iris_array, axis=0)
    
    def calculate_advanced_relative_position(self, iris_center, eye_landmarks, eye_corners):
        """Calculate iris position with improved accuracy using eye geometry."""
        if len(eye_landmarks) == 0:
            return (0.0, 0.0), (0.5, 0.5)
            
        x_coords = eye_landmarks[:, 0]
        y_coords = eye_landmarks[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        if len(eye_corners) >= 2:
            eye_center_x = (eye_corners[0][0] + eye_corners[1][0]) / 2
            eye_center_y = (eye_corners[0][1] + eye_corners[1][1]) / 2
            eye_width = abs(eye_corners[0][0] - eye_corners[1][0])
        else:
            eye_center_x = (x_min + x_max) / 2
            eye_center_y = (y_min + y_max) / 2
            eye_width = x_max - x_min
        
        eye_height = y_max - y_min
        
        if eye_width == 0 or eye_height == 0:
            return (0.0, 0.0), (eye_center_x, eye_center_y)
        
        rel_x = (iris_center[0] - eye_center_x) / (eye_width * 0.5)
        rel_y = (iris_center[1] - eye_center_y) / (eye_height * 0.5)
        
        rel_x = np.tanh(rel_x * 1.5) * 0.8
        rel_y = np.tanh(rel_y * 1.5) * 0.8
        
        return (rel_x, rel_y), (eye_center_x, eye_center_y)
    
    def smooth_gaze_data(self, current_position):
        """Apply temporal smoothing to gaze data."""
        self.gaze_buffer.append(current_position)
        
        if len(self.gaze_buffer) < 2:
            return current_position
        
        weights = np.linspace(0.5, 2.0, len(self.gaze_buffer))
        weights = weights / np.sum(weights)
        
        smoothed_x = np.average([pos[0] for pos in self.gaze_buffer], weights=weights)
        smoothed_y = np.average([pos[1] for pos in self.gaze_buffer], weights=weights)
        
        return (smoothed_x, smoothed_y)
    
    def determine_enhanced_gaze_direction(self, left_rel_pos, right_rel_pos, head_pose, ear_left, ear_right):
        """Enhanced gaze direction determination with multiple factors."""
        if ear_left < self.ear_threshold or ear_right < self.ear_threshold:
            return "Blinking", 0.0
        
        avg_rel_x = (left_rel_pos[0] + right_rel_pos[0]) / 2
        avg_rel_y = (left_rel_pos[1] + right_rel_pos[1]) / 2
        
        compensated_pos = self.compensate_for_head_pose((avg_rel_x, avg_rel_y), head_pose)
        
        smoothed_pos = self.smooth_gaze_data(compensated_pos)
        
        h_threshold = self.base_horizontal_threshold * self.adaptive_threshold_factor
        v_threshold = self.base_vertical_threshold * self.adaptive_threshold_factor
        
        if abs(head_pose['yaw']) > 15:
            h_threshold *= 1.2
        if abs(head_pose['pitch']) > 10:
            v_threshold *= 1.2
        
        recent_positions = list(self.gaze_buffer)[-3:] if len(self.gaze_buffer) >= 3 else list(self.gaze_buffer)
        if len(recent_positions) > 1:
            position_variance = np.var([pos[0]**2 + pos[1]**2 for pos in recent_positions])
            confidence = max(0.0, 1.0 - position_variance * 10)
        else:
            confidence = 0.5
        
        if smoothed_pos[0] < -h_threshold:
            horizontal = "Left"
        elif smoothed_pos[0] > h_threshold:
            horizontal = "Right"
        else:
            horizontal = "Center"
        
        if smoothed_pos[1] < -v_threshold:
            vertical = "Up"
        elif smoothed_pos[1] > v_threshold:
            vertical = "Down"
        else:
            vertical = "Center"
        
        if horizontal == "Center" and vertical == "Center":
            direction = "Center"
        elif horizontal == "Center":
            direction = vertical
        elif vertical == "Center":
            direction = horizontal
        else:
            direction = f"{vertical}-{horizontal}"
        
        return direction, confidence
    
    def draw_enhanced_annotations(self, image, left_eye, right_eye, left_iris, right_iris, 
                                left_rel_pos, right_rel_pos, gaze_direction, confidence, 
                                head_pose, img_width, img_height):
        """Draw comprehensive annotations with enhanced visualization."""
        left_ear = self.calculate_eye_aspect_ratio(left_eye) if len(left_eye) > 0 else 0
        right_ear = self.calculate_eye_aspect_ratio(right_eye) if len(right_eye) > 0 else 0
        
        if len(left_eye) > 0:
            x_min, y_min, x_max, y_max = self.get_eye_bounding_box(left_eye, img_width, img_height)
            color = (0, 255, 0) if left_ear > self.ear_threshold else (0, 255, 255)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            iris_pixel = (int(left_iris[0] * img_width), int(left_iris[1] * img_height))
            cv2.circle(image, iris_pixel, 4, (0, 0, 255), -1)
            
            cv2.putText(image, f"L: ({left_rel_pos[0]:.2f}, {left_rel_pos[1]:.2f})", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(image, f"EAR: {left_ear:.2f}", 
                       (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if len(right_eye) > 0:
            x_min, y_min, x_max, y_max = self.get_eye_bounding_box(right_eye, img_width, img_height)
            color = (0, 255, 0) if right_ear > self.ear_threshold else (0, 255, 255)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            iris_pixel = (int(right_iris[0] * img_width), int(right_iris[1] * img_height))
            cv2.circle(image, iris_pixel, 4, (0, 0, 255), -1)
            
            cv2.putText(image, f"R: ({right_rel_pos[0]:.2f}, {right_rel_pos[1]:.2f})", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(image, f"EAR: {right_ear:.2f}", 
                       (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        confidence_color = (0, 255, 255) if confidence > 0.7 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(image, f"Gaze: {gaze_direction} (Conf: {confidence:.2f})", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, confidence_color, 2)
        
        cv2.putText(image, f"Head: Y:{head_pose['yaw']:.1f} P:{head_pose['pitch']:.1f} R:{head_pose['roll']:.1f}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(image, f"Buffer: {len(self.gaze_buffer)}/{self.buffer_size}", 
                   (20, img_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def get_eye_bounding_box(self, eye_landmarks, img_width, img_height):
        """Calculate bounding box for eye region."""
        if len(eye_landmarks) == 0:
            return 0, 0, 0, 0
            
        eye_points = eye_landmarks * np.array([img_width, img_height])
        
        x_min = int(np.min(eye_points[:, 0]))
        x_max = int(np.max(eye_points[:, 0]))
        y_min = int(np.min(eye_points[:, 1]))
        y_max = int(np.max(eye_points[:, 1]))
        
        return x_min, y_min, x_max, y_max
    
    def process_frame(self, frame):
        """Process a single frame with enhanced accuracy."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        h, w = frame.shape[:2]
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            try:
                head_pose = self.estimate_head_pose(landmarks, w, h)
                self.head_pose_buffer.append(head_pose)
                
                left_eye = self.get_enhanced_eye_landmarks(landmarks, self.LEFT_EYE_LANDMARKS)
                right_eye = self.get_enhanced_eye_landmarks(landmarks, self.RIGHT_EYE_LANDMARKS)
                
                left_corners = [[landmarks[idx].x, landmarks[idx].y] for idx in self.LEFT_EYE_CORNERS]
                right_corners = [[landmarks[idx].x, landmarks[idx].y] for idx in self.RIGHT_EYE_CORNERS]
                
                left_iris = self.get_refined_iris_center(landmarks, self.LEFT_IRIS_LANDMARKS)
                right_iris = self.get_refined_iris_center(landmarks, self.RIGHT_IRIS_LANDMARKS)
                
                self.iris_buffer_left.append(left_iris)
                self.iris_buffer_right.append(right_iris)
                
                if len(self.iris_buffer_left) > 1:
                    left_iris = np.mean(list(self.iris_buffer_left), axis=0)
                    right_iris = np.mean(list(self.iris_buffer_right), axis=0)
                
                left_rel_pos, left_center = self.calculate_advanced_relative_position(
                    left_iris, left_eye, left_corners
                )
                right_rel_pos, right_center = self.calculate_advanced_relative_position(
                    right_iris, right_eye, right_corners
                )
                
                left_ear = self.calculate_eye_aspect_ratio(left_eye) if len(left_eye) > 0 else 0
                right_ear = self.calculate_eye_aspect_ratio(right_eye) if len(right_eye) > 0 else 0
                
                gaze_direction, confidence = self.determine_enhanced_gaze_direction(
                    left_rel_pos, right_rel_pos, head_pose, left_ear, right_ear
                )
                
                self.draw_enhanced_annotations(
                    frame, left_eye, right_eye, left_iris, right_iris,
                    left_rel_pos, right_rel_pos, gaze_direction, confidence,
                    head_pose, w, h
                )
                
                status_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.putText(frame, f"Enhanced Face Tracking Active", 
                           (w - 300, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                return frame, True, confidence
                
            except Exception as e:
                print(f"Error in enhanced processing: {e}")
                cv2.putText(frame, f"Processing Error: {str(e)[:30]}", 
                           (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return frame, False, 0.0
        else:
            cv2.putText(frame, "No Face Detected - Adjust lighting/position", 
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame, False, 0.0
    
    def calibrate_personal_thresholds(self, calibration_frames):
        """Calibrate thresholds based on user's eye characteristics (optional)."""
        print("Performing personal calibration...")
        pass
    
    def process_camera(self, camera_index=0, display=True):
        """Process live camera feed with enhanced accuracy tracking."""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera index {camera_index}")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if FPS not available
        
        print(f"Processing live camera feed (index {camera_index})")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Buffer size: {self.buffer_size}, Confidence threshold: 0.7")
        
        frame_count = 0
        detection_count = 0
        high_confidence_count = 0
        total_confidence = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from camera")
                break
            
            frame_count += 1
            
            # Process frame with enhanced accuracy
            processed_frame, face_detected, confidence = self.process_frame(frame)
            
            if face_detected:
                detection_count += 1
                total_confidence += confidence
                if confidence > 0.7:
                    high_confidence_count += 1
            
            # Display
            if display:
                cv2.imshow('Enhanced Gaze Tracking', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                avg_conf = total_confidence / max(detection_count, 1)
                print(f"Frames processed: {frame_count} | Avg Confidence: {avg_conf:.2f}")
        
        # Cleanup
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        # Final statistics
        detection_rate = (detection_count / frame_count) * 100
        high_conf_rate = (high_confidence_count / max(detection_count, 1)) * 100
        avg_confidence = total_confidence / max(detection_count, 1)
        
        print(f"\n=== Enhanced Camera Processing Complete ===")
        print(f"Face detection rate: {detection_rate:.1f}% ({detection_count}/{frame_count})")
        print(f"High confidence rate: {high_conf_rate:.1f}% ({high_confidence_count}/{detection_count})")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Buffer efficiency: {len(self.gaze_buffer)}/{self.buffer_size}")
        
        return True


def main():
    """Main function with enhanced options for camera input."""
    parser = argparse.ArgumentParser(description='Enhanced Gaze Tracking with Live Camera')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--buffer-size', type=int, default=5, help='Temporal smoothing buffer size (default: 5)')
    parser.add_argument('--confidence', type=float, default=0.7, help='Detection confidence threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Create enhanced tracker
    tracker = EnhancedGazeTracker(
        buffer_size=args.buffer_size,
        confidence_threshold=args.confidence
    )
    
    print("Starting Enhanced Gaze Tracking with Live Camera...")
    print("Features: Head pose compensation, temporal smoothing, blink detection")
    print("Press 'q' to quit during playback")
    
    success = tracker.process_camera(
        camera_index=args.camera_index,
        display=not args.no_display
    )
    
    if success:
        print("\nCamera processing completed successfully")
    else:
        print("\nProcessing failed.")


if __name__ == "__main__":
    main()


# Usage examples:
# python enhanced_gaze_tracker.py
# python enhanced_gaze_tracker.py --camera-index 1 --buffer-size 7 --confidence 0.8
# python enhanced_gaze_tracker.py --no-display