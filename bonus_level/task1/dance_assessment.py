# Import necessary libraries
import os
import sys
import threading
import time
from pathlib import Path

# Avoid OpenMP runtime conflicts between conda OpenCV and PyTorch on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch must load before cv2 on macOS, or OpenMP runtimes conflict and segfault.
import torch

torch.set_num_threads(1)

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from scoring_module import DanceScorer
from selecting_module import DancerSelector

# Load YOLOv8 pose model
# This will be the starting model to extract poses from videos.
# However, you can replace it with any other pose model compatible with YOLOv8.
model = YOLO("yolov8s-pose.pt")
# Protect the camera model from accidental concurrent calls.
_model_lock = threading.Lock()

# COCO skeleton connections (keypoint pairs)
# This defines the connections between keypoints for drawing the skeleton.
# For better drawing of the skeleton, you can modify the pairs.
skeleton = [
    # head
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    # torso
    (0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12),
    # arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # legs
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# UI refresh interval (ms). Inference runs on a worker thread.
UI_INTERVAL_MS = 33
KEYPOINT_CONFIDENCE = 0.3


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dance Assessment")
        self.root.geometry("1200x600")  # You can adjust the window size as needed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.template_pose_path = ""
        self.template_timestamps = None
        self.template_keypoints = None
        self.template_confidences = None
        self.template_valid = None
        self.dancer_selector = None
        self.dance_scorers = {}
        # Time of the template frame that Tkinter has actually displayed.
        self._template_time = None

        self._frame_lock = threading.Lock()
        self._selector_lock = threading.Lock()
        self._latest_file = None
        self._latest_cam = None
        self._latest_cam_status = "No dancer selected"
        self._latest_score_text = "Score: --"
        self._file_thread = None
        self._cam_thread = None
        self._cam_lost = False

        # Set up frames
        # 让根窗口的权重均分，实现左右对称
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # 1. 上方视频与摄像头区域
        self.left_frame = tk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        self.right_frame = tk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

        # 2. 下方独立评分区域 (横跨两列 columnspan=2)
        self.score_frame = tk.Frame(self.root)
        self.score_frame.grid(row=1, column=0, columnspan=2, pady=20, sticky="ew")
        self.score_frame.columnconfigure(0, weight=1) # 让内部组件居中

        # --- 左侧组件 (视频及控制) ---
        self.label_file = tk.Label(self.left_frame)
        self.label_file.pack()
        self.controls_file = tk.Frame(self.left_frame)
        self.controls_file.pack(pady=5)
        tk.Button(self.controls_file, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Open Pose Data", command=self.load_pose_data).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=5)

        # --- 右侧组件 (摄像头及控制) ---
        self.label_cam = tk.Label(self.right_frame)
        self.label_cam.pack()
        self.controls_cam = tk.Frame(self.right_frame)
        self.controls_cam.pack(pady=5)
        tk.Button(self.controls_cam, text="Start Camera", command=self.start_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Stop Camera", command=self.stop_cam).pack(side=tk.LEFT, padx=5)

        # --- 下方中央组件 (将状态和评分放到 score_frame 中，并设置 anchor="center" 居中) ---
        self.label_cam_status = tk.Label(
            self.score_frame, text="No dancer selected", font=("Arial", 12), anchor="center"
        )
        self.label_cam_status.pack(fill=tk.X, pady=2)
        
        self.label_score = tk.Label(
            self.score_frame, text="Score: --", font=("Arial", 14, "bold"), fg="green", anchor="center", justify=tk.CENTER
        )
        self.label_score.pack(fill=tk.X, pady=2)

    def on_close(self):
        self.stop_video()
        self.stop_cam()
        self.root.destroy()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            if self.running_file:
                self.stop_video()
            self.video_path = path
            with self._selector_lock:
                self.template_pose_path = ""
                self.template_timestamps = None
                self.template_keypoints = None
                self.template_confidences = None
                self.template_valid = None
                self.dancer_selector = None
                self.dance_scorers = {}

            video_path = Path(path)
            automatic_candidates = [
                Path(__file__).resolve().parent
                / "dance_pose_data"
                / video_path.stem
                / "poses.npz",
                video_path.parent / "dance_pose_data" / video_path.stem / "poses.npz",
                video_path.parent / video_path.stem / "poses.npz",
            ]
            loaded_automatically = False
            for candidate in automatic_candidates:
                if candidate.is_file():
                    loaded_automatically = self._load_pose_file(candidate, show_message=False)
                    if loaded_automatically:
                        break

            pose_message = (
                "\nPose cache loaded automatically."
                if loaded_automatically
                else "\nUse Open Pose Data before starting the template."
            )
            messagebox.showinfo(
                "Video Selected", os.path.basename(path) + pose_message
            )

    def load_pose_data(self):
        path = filedialog.askopenfilename(
            title="Select preprocessed template poses",
            filetypes=[("Pose cache", "*.npz"), ("All files", "*.*")],
        )
        if path:
            if self.running_file:
                self.stop_video()
            self._load_pose_file(Path(path), show_message=True)

    def _load_pose_file(self, path, show_message):
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "timestamps",
                    "keypoints",
                    "confidences",
                    "valid",
                }
                if not required.issubset(data.files):
                    missing = ", ".join(sorted(required.difference(data.files)))
                    raise ValueError(f"Missing arrays: {missing}")
                timestamps = data["timestamps"].astype(np.float32)
                keypoints = data["keypoints"].astype(np.float32)
                confidences = data["confidences"].astype(np.float32)
                valid = data["valid"].astype(bool)

            if (
                keypoints.ndim != 3
                or keypoints.shape[1:] != (17, 2)
                or confidences.shape != keypoints.shape[:2]
                or len(timestamps) != len(keypoints)
                or len(valid) != len(keypoints)
            ):
                raise ValueError("Unexpected pose-cache array shapes")
            if not np.any(valid):
                raise ValueError("The pose cache contains no valid template poses")

            selector = DancerSelector(timestamps, keypoints, confidences, valid)
            with self._selector_lock:
                self.template_pose_path = str(path)
                self.template_timestamps = timestamps
                self.template_keypoints = keypoints
                self.template_confidences = confidences
                self.template_valid = valid
                self.dancer_selector = selector
                self.dance_scorers = {}
            if show_message:
                messagebox.showinfo(
                    "Pose Data Loaded",
                    f"{Path(path).name}: {int(np.sum(valid))}/{len(valid)} valid poses",
                )
            return True
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Pose Data Error", str(exc))
            return False

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        if self.dancer_selector is None:
            messagebox.showwarning(
                "No Pose Data",
                "Preprocess the template first, then select its poses.npz file.",
            )
            return
        if self.running_file:
            return

        self.cap_file = cv2.VideoCapture(self.video_path)
        if not self.cap_file.isOpened():
            messagebox.showerror("Video Error", "Could not open the selected video.")
            self.cap_file = None
            return

        with self._frame_lock:
            self._latest_file = None
            self._template_time = None
        with self._selector_lock:
            self.dancer_selector.reset()
            self.dance_scorers.clear()
        self.running_file = True
        self._file_thread = threading.Thread(target=self._video_worker, daemon=True)
        self._file_thread.start()
        self._tick_video_ui()

    def stop_video(self):
        self.running_file = False
        thread = self._file_thread
        self._file_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if self.cap_file is not None:
            self.cap_file.release()
            self.cap_file = None
        with self._frame_lock:
            self._latest_file = None
            self._template_time = None
        with self._selector_lock:
            for scorer in self.dance_scorers.values():
                scorer.pause()

    def _open_webcam(self):
        if sys.platform == "darwin":
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(0)

    def start_cam(self):
        if self.running_cam:
            return

        self.cap_cam = self._open_webcam()
        if not self.cap_cam.isOpened():
            self.cap_cam = None
            messagebox.showerror(
                "Camera Error",
                "Could not open the webcam.\n\n"
                "On macOS, allow camera access in:\n"
                "System Settings → Privacy & Security → Camera",
            )
            return

        # Prefer the newest frame; avoid building up stale buffered frames.
        try:
            self.cap_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        with self._frame_lock:
            self._latest_cam = None
            self._latest_cam_status = "Searching for dancer..."
            self._latest_score_text = "Score: --"
        self._cam_lost = False
        if self.dancer_selector is not None:
            with self._selector_lock:
                self.dancer_selector.reset()
                self.dance_scorers.clear()
        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) if predictor else []:
            reset = getattr(tracker, "reset", None)
            if reset is not None:
                reset()
        self.running_cam = True
        self._cam_thread = threading.Thread(target=self._cam_worker, daemon=True)
        self._cam_thread.start()
        self._tick_cam_ui()

    def stop_cam(self):
        self.running_cam = False
        thread = self._cam_thread
        self._cam_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if self.cap_cam is not None:
            self.cap_cam.release()
            self.cap_cam = None
        with self._frame_lock:
            self._latest_cam = None
            self._latest_cam_status = "No dancer selected"
            self._latest_score_text = "Score: --"
        with self._selector_lock:
            for scorer in self.dance_scorers.values():
                scorer.pause()
        self.label_cam_status.configure(text="No dancer selected")
        self.label_score.configure(text="Score: --")

    def _video_worker(self):
        cap = self.cap_file
        if cap is None:
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
            
        started_at = time.monotonic()

        while self.running_file:
            ret, frame = cap.read()
            if not ret:
                self.running_file = False
                break

            # 计算这一帧原本应该在什么绝对时间点播放
            current_time = time.monotonic()
            elapsed = current_time - started_at
            
            # 读取当前帧在视频中的实际时间戳（毫秒转秒）
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # 如果视频跑得太快了，就等等它
            wait_seconds = video_time - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            
            # 如果处理太慢导致视频滞后了（wait_seconds < -0.1），可以选择跳帧，这里保持正常渲染
            processed = self._draw_template_pose(frame, video_time)
            with self._frame_lock:
                self._latest_file = (processed, video_time)

    def _cam_worker(self):
        while self.running_cam:
            cap = self.cap_cam
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                self._cam_lost = True
                self.running_cam = False
                break
            capture_clock = time.monotonic()
            with self._frame_lock:
                template_time_at_capture = self._template_time

            with _model_lock:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.3,
                    imgsz=320,
                    verbose=False,
                )
            result = results[0]
            detections = []
            if (
                result.boxes is not None
                and result.boxes.id is not None
                and result.keypoints is not None
            ):
                track_ids = result.boxes.id.int().cpu().numpy()
                boxes = result.boxes.xyxyn.cpu().numpy()
                keypoints = result.keypoints.xyn.cpu().numpy()
                keypoint_conf = result.keypoints.conf
                if keypoint_conf is None:
                    confidences = np.ones(keypoints.shape[:2], dtype=np.float32)
                else:
                    confidences = keypoint_conf.cpu().numpy()
                for person_index, track_id in enumerate(track_ids):
                    detections.append(
                        {
                            "track_id": int(track_id),
                            "bbox": boxes[person_index].astype(np.float32),
                            "keypoints": keypoints[person_index].astype(np.float32),
                            "confidence": confidences[person_index].astype(np.float32),
                        }
                    )

            selector = self.dancer_selector
            score_results = {}
            if selector is not None:
                with self._selector_lock:
                    selected_ids, scores = selector.update(
                        detections, capture_clock, template_time_at_capture
                    )
                    template_active = selector.template_active
                    detections_by_id = {
                        detection["track_id"]: detection
                        for detection in detections
                    }
                    if template_active:
                        for track_id in list(self.dance_scorers):
                            if track_id not in selected_ids:
                                self.dance_scorers.pop(track_id, None)
                        for track_id in selected_ids:
                            detection = detections_by_id.get(track_id)
                            if detection is None:
                                continue
                            scorer = self.dance_scorers.get(track_id)
                            if scorer is None:
                                scorer = DanceScorer(
                                    self.template_timestamps,
                                    self.template_keypoints,
                                    self.template_confidences,
                                    self.template_valid,
                                )
                                self.dance_scorers[track_id] = scorer
                            result = scorer.update(
                                track_id,
                                detection["keypoints"],
                                detection["confidence"],
                                capture_clock,
                                template_time_at_capture,
                            )
                            if result is not None:
                                score_results[track_id] = result
                    else:
                        for scorer in self.dance_scorers.values():
                            scorer.pause()
            else:
                selected_ids, scores = set(), {}
                template_active = False

            processed = frame.copy()
            height, width = processed.shape[:2]
            for detection in detections:
                track_id = detection["track_id"]
                is_selected = track_id in selected_ids
                # OpenCV uses BGR: unselected people use bright cyan/blue.
                line_color = (0, 255, 0) if is_selected else (255, 220, 0)
                point_color = (0, 0, 255) if is_selected else (255, 80, 0)
                self._draw_skeleton(
                    processed,
                    detection["keypoints"],
                    detection["confidence"],
                    line_color,
                    point_color,
                    thickness=3 if is_selected else 1,
                )

                x1, y1, x2, y2 = detection["bbox"]
                box_start = (int(x1 * width), int(y1 * height))
                box_end = (int(x2 * width), int(y2 * height))
                box_color = (0, 255, 0) if is_selected else (255, 220, 0)
                cv2.rectangle(processed, box_start, box_end, box_color, 2)
                total = scores.get(track_id, {}).get("total", 0.0)
                label = f"ID {track_id}  {total:.2f}"
                if is_selected:
                    current_score = score_results.get(track_id, {}).get("total")
                    score_suffix = (
                        f" SCORE {current_score:.0f}"
                        if current_score is not None
                        else ""
                    )
                    label = f"DANCER {label}{score_suffix}"
                cv2.putText(
                    processed,
                    label,
                    (box_start[0], max(20, box_start[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

            if selected_ids:
                cv2.putText(
                    processed,
                    f"DANCERS {len(selected_ids)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if selector is None:
                status = "Load preprocessed pose data"
                score_text = "Score: --"
            elif template_time_at_capture is None:
                status = "Start the template video"
                score_text = "Score: --"
            elif not template_active:
                status = "Waiting for template dancer..."
                score_text = "Score paused: no template pose"
            elif not selected_ids:
                status = "Searching for dancers..."
                score_text = "Score: waiting for dancers"
            else:
                status_parts = []
                score_lines = []
                for track_id in sorted(selected_ids):
                    selected_score = scores.get(track_id, {})
                    status_parts.append(
                        f"ID {track_id} "
                        f"(template {selected_score.get('template', 0.0):.2f}, "
                        f"motion {selected_score.get('activity', 0.0):.2f})"
                    )
                    score_result = score_results.get(track_id)
                    if score_result is None:
                        score_lines.append(f"ID {track_id}: calibrating...")
                        continue
                    motion_text = (
                        f"{score_result['motion']:.0f}"
                        if score_result["motion_informative"]
                        else "--"
                    )
                    score_lines.append(
                        f"ID {track_id}: Current {score_result['total']:.0f} | "
                        f"Overall {score_result['overall']:.0f} | "
                        f"Pose {score_result['pose']:.0f} | "
                        f"Motion {motion_text} | "
                        f"Timing {score_result['timing']:.0f} | "
                        f"Delay {score_result['lag']:.2f}s"
                    )
                status = "Dancers: " + "; ".join(status_parts)
                score_text = "\n".join(score_lines)

            with self._frame_lock:
                self._latest_cam = processed
                self._latest_cam_status = status
                self._latest_score_text = score_text

    def _tick_video_ui(self):
        with self._frame_lock:
            packet = self._latest_file
            self._latest_file = None

        if packet is not None:
            frame, video_time = packet
            self.update_label(self.label_file, frame)
            # Publish time only after this exact frame is visible to the player.
            with self._frame_lock:
                self._template_time = video_time

        if self.running_file:
            self.root.after(UI_INTERVAL_MS, self._tick_video_ui)
        elif self.cap_file is not None:
            self.stop_video()

    def _tick_cam_ui(self):
        if self._cam_lost:
            self._cam_lost = False
            self.stop_cam()
            messagebox.showwarning("Camera Error", "Lost connection to the webcam.")
            return

        if not self.running_cam:
            return

        with self._frame_lock:
            frame = self._latest_cam
            self._latest_cam = None
            status = self._latest_cam_status
            score_text = self._latest_score_text

        # Only refresh when a new inferred frame arrives; keep last image otherwise.
        if frame is not None:
            self.update_label(self.label_cam, frame)
        self.label_cam_status.configure(text=status)
        self.label_score.configure(text=score_text)

        self.root.after(UI_INTERVAL_MS, self._tick_cam_ui)

    def _draw_template_pose(self, frame, video_time):
        # 默认总是显示视频画面，直接复制输入的视频帧
        overlay = frame.copy()

        selector = self.dancer_selector
        if selector is None or not len(selector.valid_indices):
            return overlay
            
        valid_times = self.template_timestamps[selector.valid_indices]
        position = int(np.searchsorted(valid_times, video_time))
        choices = []
        if position < len(valid_times):
            choices.append(position)
        if position > 0:
            choices.append(position - 1)
        if not choices:
            return overlay
            
        nearest = min(
            choices, key=lambda index: abs(float(valid_times[index]) - video_time)
        )
        pose_index = int(selector.valid_indices[nearest])
        
        if abs(float(self.template_timestamps[pose_index]) - video_time) <= 0.15:
            self._draw_skeleton(
                overlay,
                self.template_keypoints[pose_index],
                self.template_confidences[pose_index],
                (0, 255, 0),
                (0, 0, 255),
                thickness=2,
            )
        return overlay

    @staticmethod
    def _draw_skeleton(
        overlay, keypoints_xyn, confidence, line_color, point_color, thickness
    ):
        height, width = overlay.shape[:2]
        points = np.asarray(keypoints_xyn)
        confidence = np.asarray(confidence)
        visible = (
            (confidence >= KEYPOINT_CONFIDENCE)
            & np.all(np.isfinite(points), axis=1)
        )
        safe_points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        pixel_points = np.column_stack(
            (safe_points[:, 0] * width, safe_points[:, 1] * height)
        ).astype(np.int32)

        for point_a, point_b in skeleton:
            if visible[point_a] and visible[point_b]:
                cv2.line(
                    overlay,
                    tuple(pixel_points[point_a]),
                    tuple(pixel_points[point_b]),
                    line_color,
                    thickness,
                )
        for index, point in enumerate(pixel_points):
            if visible[index]:
                cv2.circle(overlay, tuple(point), 4, point_color, -1)

    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 384), Image.BILINEAR)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()