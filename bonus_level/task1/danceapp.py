# Import necessary libraries
import os
import sys
import threading

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

# Load YOLOv8 pose model
# This will be the starting model to extract poses from videos.
# However, you can replace it with any other pose model compatible with YOLOv8.
model = YOLO("yolov8s-pose.pt")
# Shared YOLO weights are not safe for concurrent predicts; serialize inference.
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


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stickman Dance GUI")
        self.root.geometry("1200x600")  # You can adjust the window size as needed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self._frame_lock = threading.Lock()
        self._latest_file = None
        self._latest_cam = None
        self._file_thread = None
        self._cam_thread = None
        self._cam_lost = False

        # Set up frames
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10)

        # Video File Window
        self.label_file = tk.Label(self.left_frame)
        self.label_file.pack()
        self.controls_file = tk.Frame(self.left_frame)
        self.controls_file.pack()
        tk.Button(self.controls_file, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_file, text="Show/Hide Video", command=self.toggle_video_display).pack(side=tk.LEFT, padx=5)

        # Webcam Window
        self.label_cam = tk.Label(self.right_frame)
        self.label_cam.pack()
        self.controls_cam = tk.Frame(self.right_frame)
        self.controls_cam.pack()
        tk.Button(self.controls_cam, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=5)

    def on_close(self):
        self.stop_video()
        self.stop_cam()
        self.root.destroy()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
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

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

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
        self._cam_lost = False
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

    def _video_worker(self):
        while self.running_file:
            cap = self.cap_file
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                self.running_file = False
                break
            processed = self.process_pose(frame, show_background=self.show_video_frame)
            with self._frame_lock:
                self._latest_file = processed

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
            # Webcam always shows the live frame (toggle only applies to left video).
            processed = self.process_pose(frame, show_background=True)
            with self._frame_lock:
                self._latest_cam = processed

    def _tick_video_ui(self):
        with self._frame_lock:
            frame = self._latest_file
            self._latest_file = None

        if frame is not None:
            self.update_label(self.label_file, frame)

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

        # Only refresh when a new inferred frame arrives; keep last image otherwise.
        if frame is not None:
            self.update_label(self.label_cam, frame)

        self.root.after(UI_INTERVAL_MS, self._tick_cam_ui)

    def process_pose(self, frame, show_background=True):
        with _model_lock:
            results = model(frame, conf=0.3, verbose=False, imgsz=320)
        height, width = frame.shape[:2]

        if show_background:
            overlay = frame.copy()
        else:
            overlay = np.ones_like(frame) * 255  # white background

        for result in results:
            if result.keypoints is not None:
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for person_kpts in keypoints_xyn:
                    keypoints = [
                        (int(x * width), int(y * height)) for x, y in person_kpts
                    ]

                    for pt1, pt2 in skeleton:
                        if pt1 < len(keypoints) and pt2 < len(keypoints):
                            x1, y1 = keypoints[pt1]
                            x2, y2 = keypoints[pt2]
                            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green lines for skeleton

                    for x, y in keypoints:
                        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)  # red circles for keypoints

        return overlay

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
