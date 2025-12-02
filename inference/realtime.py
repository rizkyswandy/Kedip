"""
Real-time inference for blink detection.

Features:
    - Frame buffering for temporal sequences
    - Optimized for low latency
    - Supports multiple input sources (webcam, video, stream)
"""

import time
from collections import deque

import cv2
import numpy as np
import torch

from data.preprocessing import FeatureExtractor
from models import BlinkTransformer


class RealtimeBlinkDetector:
    """
    Real-time blink detection with frame buffering.

    Maintains a sliding window of frames for temporal analysis.
    """

    def __init__(
        self,
        model: BlinkTransformer,
        device: str = 'cuda',
        sequence_length: int = 16,
        threshold: float = 0.5,
        buffer_size: int | None = None
    ):
        """
        Args:
            model: Trained BlinkTransformer model
            device: Compute device
            sequence_length: Number of frames to analyze
            threshold: Classification threshold
            buffer_size: Size of frame buffer (default: sequence_length)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sequence_length = sequence_length
        self.threshold = threshold

        # Frame buffer (sliding window)
        self.buffer_size = buffer_size or sequence_length
        self.frame_buffer = deque(maxlen=self.buffer_size)

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Statistics
        self.total_blinks = 0
        self.frame_count = 0
        self.last_blink_time = 0
        self.fps_history = deque(maxlen=30)

        # State tracking
        self.last_state = 0  # 0=open, 1=closed
        self.blink_in_progress = False

    def reset(self):
        """Reset detector state."""
        self.frame_buffer.clear()
        self.total_blinks = 0
        self.frame_count = 0
        self.last_blink_time = 0
        self.fps_history.clear()
        self.last_state = 0
        self.blink_in_progress = False

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame.

        Args:
            frame: RGB frame (H, W, 3)

        Returns:
            Dictionary with:
                - 'has_blink': Whether blink detected in current sequence
                - 'confidence': Confidence score
                - 'eye_state': Current eye state (0=open, 1=closed)
                - 'total_blinks': Total blinks detected
                - 'fps': Current FPS
        """
        start_time = time.time()

        # Extract features
        features = self.feature_extractor.extract_frame(frame)

        if features is None:
            # No face detected
            return {
                'has_blink': False,
                'confidence': 0.0,
                'eye_state': 0,
                'total_blinks': self.total_blinks,
                'fps': self.get_fps(),
                'face_detected': False
            }

        # Add to buffer
        self.frame_buffer.append(features)
        self.frame_count += 1

        # Need full buffer for inference
        if len(self.frame_buffer) < self.sequence_length:
            return {
                'has_blink': False,
                'confidence': 0.0,
                'eye_state': 0,
                'total_blinks': self.total_blinks,
                'fps': self.get_fps(),
                'face_detected': True,
                'buffer_full': False
            }

        # Prepare sequence for inference
        sequence = self._prepare_sequence()

        # Run inference
        with torch.no_grad():
            pred_presence, pred_state = self.model(
                sequence['rgb'].unsqueeze(0).to(self.device),
                sequence['landmarks'].unsqueeze(0).to(self.device),
                sequence['pose'].unsqueeze(0).to(self.device)
            )

        # Get predictions
        has_blink = pred_presence.item() > self.threshold
        confidence = pred_presence.item()
        current_state = pred_state[0, -1].item()  # Last frame state

        # Detect blink event (transition from open to closed and back)
        if self._is_blink_event(current_state):
            self.total_blinks += 1
            self.last_blink_time = time.time()

        # Update FPS
        elapsed = time.time() - start_time
        self.fps_history.append(1.0 / elapsed if elapsed > 0 else 0.0)

        return {
            'has_blink': has_blink,
            'confidence': confidence,
            'eye_state': current_state,
            'total_blinks': self.total_blinks,
            'fps': self.get_fps(),
            'face_detected': True,
            'buffer_full': True
        }

    def _prepare_sequence(self) -> dict[str, torch.Tensor]:
        """
        Prepare sequence from buffer for inference.

        Returns:
            Dictionary with tensors
        """
        # Get last sequence_length frames
        frames = list(self.frame_buffer)[-self.sequence_length:]

        # Stack features
        left_eyes = []
        landmarks = []
        poses = []

        for feat in frames:
            # Combine eyes
            eye_combined = np.concatenate([feat['left_eye'], feat['right_eye']], axis=1)
            eye_tensor = torch.from_numpy(eye_combined).permute(2, 0, 1).float() / 255.0

            left_eyes.append(eye_tensor)
            landmarks.append(torch.from_numpy(feat['landmarks']))
            poses.append(torch.from_numpy(feat['pose']))

        return {
            'rgb': torch.stack(left_eyes),
            'landmarks': torch.stack(landmarks),
            'pose': torch.stack(poses)
        }

    def _is_blink_event(self, current_state: float) -> bool:
        """
        Detect if a blink event occurred.

        A blink is: open -> closed -> open

        Args:
            current_state: Current eye state (0-1)

        Returns:
            True if blink detected
        """
        is_closed = current_state > self.threshold

        # Check for transition
        if is_closed and not self.blink_in_progress:
            # Eyes just closed
            self.blink_in_progress = True
            return False

        elif not is_closed and self.blink_in_progress:
            # Eyes just opened - blink complete
            self.blink_in_progress = False
            return True

        return False

    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def close(self):
        """Release resources."""
        self.feature_extractor.close()


class VideoProcessor:
    """
    Process video files or streams with blink detection.
    """

    def __init__(
        self,
        detector: RealtimeBlinkDetector,
        visualize: bool = True
    ):
        """
        Args:
            detector: Blink detector instance
            visualize: Show visualization
        """
        self.detector = detector
        self.visualize = visualize

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None
    ) -> dict:
        """
        Process video file.

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)

        Returns:
            Dictionary with results
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames
        print(f"Processing {total_frames} frames...")
        blink_timestamps = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = self.detector.process_frame(frame)

            # Record blink timestamp
            if result.get('has_blink', False):
                timestamp = frame_idx / fps
                blink_timestamps.append(timestamp)

            # Visualize
            if self.visualize or writer:
                vis_frame = self._draw_results(frame, result)

                if self.visualize:
                    cv2.imshow('Blink Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if writer:
                    writer.write(vis_frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if self.visualize:
            cv2.destroyAllWindows()

        return {
            'total_frames': frame_idx,
            'total_blinks': self.detector.total_blinks,
            'blink_timestamps': blink_timestamps,
            'avg_fps': self.detector.get_fps()
        }

    def process_webcam(self, camera_id: int = 0):
        """
        Process webcam stream.

        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        print("Starting webcam. Press 'q' to quit, 'r' to reset stats...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = self.detector.process_frame(frame)

            # Visualize
            vis_frame = self._draw_results(frame, result)
            cv2.imshow('Blink Detection', vis_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detector.reset()

        cap.release()
        cv2.destroyAllWindows()

    def _draw_results(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input frame
            result: Detection results

        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()

        # Check if face detected
        if not result.get('face_detected', False):
            cv2.putText(
                vis_frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            return vis_frame

        # Eye state
        eye_state = "CLOSED" if result['eye_state'] > 0.5 else "OPEN"
        color = (0, 0, 255) if eye_state == "CLOSED" else (0, 255, 0)

        cv2.putText(
            vis_frame,
            f"Eyes: {eye_state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        # Blink indicator
        if result.get('has_blink', False):
            cv2.putText(
                vis_frame,
                "BLINK!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        # Statistics
        cv2.putText(
            vis_frame,
            f"Blinks: {result['total_blinks']}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis_frame,
            f"Confidence: {result['confidence']:.2f}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis_frame,
            f"FPS: {result['fps']:.1f}",
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return vis_frame


if __name__ == '__main__':
    """Test real-time detector."""
    print("Testing RealtimeBlinkDetector...")
    print("Note: This requires a trained model to test fully.")
    print("See demo/webcam_demo.py for complete usage example.")
    print("\nRealtime inference module created successfully! ✓")
