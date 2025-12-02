"""
Feature Extraction using MediaPipe

Extracts multi-modal features from video frames:
- RGB eye patches (64x64)
- Facial landmarks (146 dimensions)
- Head pose angles (pitch, yaw, roll)
"""


import cv2
import mediapipe as mp
import numpy as np
import torch


class FeatureExtractor:
    """
    Extract features from frames using MediaPipe Face Mesh.

    Features:
        - RGB eye patches: Left and right eye regions (64x64)
        - Landmarks: 468 face mesh landmarks → 146-dim embedding
        - Head pose: Pitch, yaw, roll angles
    """

    def __init__(
        self,
        eye_patch_size: tuple[int, int] = (64, 64),
        use_refine_landmarks: bool = True,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Args:
            eye_patch_size: Size of eye patches (H, W)
            use_refine_landmarks: Use refined eye/iris landmarks
            static_image_mode: Process each frame independently
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.eye_patch_size = eye_patch_size

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=use_refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Eye landmarks indices (MediaPipe Face Mesh)
        # Left eye: 362, 385, 387, 263, 373, 380
        # Right eye: 33, 160, 158, 133, 153, 144
        self.left_eye_indices = [362, 385, 387, 263, 373, 380, 466, 388, 390, 373, 466, 263]
        self.right_eye_indices = [33, 160, 158, 133, 153, 144, 246, 161, 159, 145, 246, 33]

        # Face contour for head pose estimation
        self.face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def extract_frame(self, frame: np.ndarray) -> dict[str, np.ndarray] | None:
        """
        Extract features from a single frame.

        Args:
            frame: RGB image (H, W, 3)

        Returns:
            Dictionary with:
                - 'left_eye': Left eye patch (H, W, 3)
                - 'right_eye': Right eye patch (H, W, 3)
                - 'landmarks': Landmark features (146,)
                - 'pose': Head pose angles (3,) - [pitch, yaw, roll]
            Or None if face detection fails
        """
        h, w = frame.shape[:2]

        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.dtype == np.uint8 else frame
        else:
            rgb_frame = frame

        # Process frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Convert landmarks to numpy array
        landmarks_array = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks.landmark
        ])

        # Extract eye patches
        left_eye_patch = self._extract_eye_patch(frame, landmarks_array, self.left_eye_indices)
        right_eye_patch = self._extract_eye_patch(frame, landmarks_array, self.right_eye_indices)

        if left_eye_patch is None or right_eye_patch is None:
            return None

        # Extract landmark features (flatten and normalize)
        landmark_features = self._extract_landmark_features(landmarks_array)

        # Extract head pose
        head_pose = self._estimate_head_pose(landmarks_array, (h, w))

        return {
            'left_eye': left_eye_patch,
            'right_eye': right_eye_patch,
            'landmarks': landmark_features,
            'pose': head_pose
        }

    def _extract_eye_patch(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        eye_indices: list
    ) -> np.ndarray | None:
        """
        Extract and crop eye region from frame.

        Args:
            frame: Original frame
            landmarks: All face landmarks
            eye_indices: Indices of eye landmarks

        Returns:
            Eye patch resized to eye_patch_size or None
        """
        # Get eye landmarks
        eye_points = landmarks[eye_indices][:, :2]  # Only x, y

        # Calculate bounding box with padding
        x_min, y_min = eye_points.min(axis=0).astype(int)
        x_max, y_max = eye_points.max(axis=0).astype(int)

        # Add padding (30% on each side)
        width = x_max - x_min
        height = y_max - y_min
        pad_x = int(width * 0.3)
        pad_y = int(height * 0.3)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(frame.shape[1], x_max + pad_x)
        y_max = min(frame.shape[0], y_max + pad_y)

        # Crop eye region
        eye_patch = frame[y_min:y_max, x_min:x_max]

        if eye_patch.size == 0:
            return None

        # Resize to target size
        eye_patch = cv2.resize(eye_patch, self.eye_patch_size)

        return eye_patch

    def _extract_landmark_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract and normalize landmark features.

        Args:
            landmarks: Face landmarks (468, 3)

        Returns:
            Normalized features (146,)
        """
        # Use key landmarks around eyes and face
        # Left eye region: 33-133 (landmark indices)
        # Right eye region: 362-263
        # Mouth region: 61-291
        key_indices = list(range(33, 134)) + list(range(362, 385)) + list(range(61, 92))

        # Extract key landmarks
        key_landmarks = landmarks[key_indices]

        # Normalize by face bounding box
        bbox_min = landmarks.min(axis=0)
        bbox_max = landmarks.max(axis=0)
        bbox_size = bbox_max - bbox_min

        normalized = (key_landmarks - bbox_min) / (bbox_size + 1e-6)

        # Flatten and take first 146 dimensions
        features = normalized.flatten()[:146]

        return features.astype(np.float32)

    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        image_size: tuple[int, int]
    ) -> np.ndarray:
        """
        Estimate head pose angles (pitch, yaw, roll).

        Args:
            landmarks: Face landmarks (468, 3)
            image_size: Image (height, width)

        Returns:
            Head pose angles (3,) - [pitch, yaw, roll] in radians
        """
        h, w = image_size

        # 3D model points (canonical face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float32)

        # 2D image points from landmarks
        image_points = np.array([
            landmarks[1][:2],    # Nose tip
            landmarks[152][:2],  # Chin
            landmarks[226][:2],  # Left eye left corner
            landmarks[446][:2],  # Right eye right corner
            landmarks[57][:2],   # Left mouth corner
            landmarks[287][:2]   # Right mouth corner
        ], dtype=np.float32)

        # Camera matrix (assuming no lens distortion)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return np.zeros(3, dtype=np.float32)

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Calculate Euler angles
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0

        return np.array([pitch, yaw, roll], dtype=np.float32)

    def extract_sequence(
        self,
        frames: list,
        combine_eyes: bool = True
    ) -> dict[str, torch.Tensor] | None:
        """
        Extract features from a sequence of frames.

        Args:
            frames: List of frames (each H, W, 3)
            combine_eyes: Concatenate left and right eye patches horizontally

        Returns:
            Dictionary with:
                - 'rgb': Eye patches (T, 3, H, W) or (T, 3, H, W*2) if combined
                - 'landmarks': Landmark features (T, 146)
                - 'pose': Head pose angles (T, 3)
            Or None if any extraction fails
        """
        rgb_patches = []
        landmarks_list = []
        poses_list = []

        for frame in frames:
            features = self.extract_frame(frame)

            if features is None:
                return None

            # Combine or use one eye
            if combine_eyes:
                eye_patch = np.concatenate([
                    features['left_eye'],
                    features['right_eye']
                ], axis=1)
            else:
                # Use left eye only
                eye_patch = features['left_eye']

            # Convert to tensor format (C, H, W)
            eye_patch = torch.from_numpy(eye_patch).permute(2, 0, 1).float() / 255.0

            rgb_patches.append(eye_patch)
            landmarks_list.append(torch.from_numpy(features['landmarks']))
            poses_list.append(torch.from_numpy(features['pose']))

        return {
            'rgb': torch.stack(rgb_patches),
            'landmarks': torch.stack(landmarks_list),
            'pose': torch.stack(poses_list)
        }

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


if __name__ == '__main__':
    """Test feature extraction."""
    import time

    print("Testing FeatureExtractor...")

    # Initialize extractor
    extractor = FeatureExtractor()

    # Test with webcam or dummy frame
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("No webcam found")

        print("\nTesting with webcam (press 'q' to quit)...")

        frame_times = []
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            features = extractor.extract_frame(frame)
            elapsed = time.time() - start
            frame_times.append(elapsed)

            if features:
                # Visualize
                combined = np.concatenate([
                    features['left_eye'],
                    features['right_eye']
                ], axis=1)

                cv2.imshow('Eye Patches', combined)

                print(f"\rFPS: {1/elapsed:.1f} | "
                      f"Landmarks: {features['landmarks'].shape} | "
                      f"Pose: {features['pose'].round(2)}", end='')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n\nAverage extraction time: {np.mean(frame_times)*1000:.2f}ms")
        print(f"Average FPS: {1/np.mean(frame_times):.1f}")
        print("\nFeature extraction test passed! ✓")

    except Exception as e:
        print(f"\nWebcam test failed: {e}")
        print("Testing with dummy frame instead...")

        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = extractor.extract_frame(dummy_frame)

        if features:
            print("Dummy frame extraction successful!")
            print(f"  Left eye: {features['left_eye'].shape}")
            print(f"  Right eye: {features['right_eye'].shape}")
            print(f"  Landmarks: {features['landmarks'].shape}")
            print(f"  Pose: {features['pose'].shape}")
        else:
            print("No face detected in dummy frame (expected)")

    finally:
        extractor.close()
