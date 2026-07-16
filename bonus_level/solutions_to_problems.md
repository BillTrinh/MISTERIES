# Problems and Solutions

## 1. Pose model selection

### Problem

The original application needed a pose model that was accurate enough for dance
recognition while still being fast enough for real-time video and webcam use.

`danceapp_wholebody.py` was created to test the RTMW COCO-WholeBody model. It
provides 133 keypoints, including body, feet, face, and hands. However, this
model was too large for the current application:

- It required more computation and had higher inference latency.
- Most face and hand keypoints were unnecessary for the current stickman and
  dance-comparison requirements.
- It introduced additional `rtmlib` and `onnxruntime` dependencies and
  compatibility requirements.

### Solution

The main application keeps the Ultralytics YOLOv8 small pose model:

```python
model = YOLO("yolov8s-pose.pt")
```

`yolov8s-pose.pt` outputs the 17 COCO body keypoints. This is sufficient for
tracking the main arm, leg, shoulder, hip, and torso movements used in dance
analysis, while providing a better balance between accuracy, model size, and
real-time speed.

`danceapp_wholebody.py` is retained as an experimental WholeBody version, but
`danceapp.py` and `yolov8s-pose.pt` are used for the main application.

## 2. Webcam frame dropping and flashing

### Problem

The webcam image appeared to flash or update unevenly. The camera itself was
not necessarily losing frames. The main cause was that frame capture, YOLO
inference, image conversion, and Tkinter display updates all ran sequentially
on Tkinter's main thread.

YOLO inference could block the main event loop for tens or hundreds of
milliseconds. The previous `root.after(1, ...)` call only requested another
update after inference completed; it did not guarantee a stable frame rate.
This caused irregular display updates that looked like dropped frames or
flashing.

### Solution

The video and webcam processing were moved to background worker threads:

- Worker threads read frames and run pose inference.
- The Tkinter main thread only updates the interface.
- Thread-safe latest-frame buffers store the newest processed video and webcam
  frames.
- The interface checks for a new frame every 33 ms, targeting approximately
  30 FPS.
- If inference has not produced a new frame, the last valid image remains on
  screen instead of displaying an empty frame.
- The webcam buffer size is set to 1 when supported, reducing delayed,
  accumulated frames.

Because the video and webcam share one YOLO model instance, inference is
protected by `_model_lock`. This allows both sides of the interface to run
without performing unsafe concurrent predictions on the same model.

The relevant structure in `danceapp.py` is:

```text
Video worker thread  ─┐
                      ├─ serialized YOLO inference ─ latest-frame buffers
Webcam worker thread ─┘                                  │
                                                        ▼
                                              Tkinter UI at ~30 FPS
```

This separates expensive pose inference from interface rendering and removes
the main cause of the flashing behavior.


Here is the documented issue and solution written in the same Markdown format and style as your example:

---

## 3. Reference video playback speed distortion

### Problem

The template/reference video did not play at its original, native speed. It either suffered from cumulative slowdowns or general pacing mismatches.

The primary cause was the video worker thread's frame timing implementation. It relied on a sequential relative delay calculation (`started_at + video_time - time.monotonic()`) inside the loop. If certain frames took longer to process or render due to system overhead, the loop accumulated lag over time. Additionally, relying purely on the computed frame index and a generic FPS estimation from OpenCV could lead to playback drift if the video possessed variable frame rates or minor encoding inconsistencies.

### Solution

The video worker's loop was refactored to enforce a strict synchronization mechanism based on the video’s absolute internal clock:

* **Absolute Clock Alignment:** Instead of calculating time by incrementing frame counts, the system queries the exact hardware-level playback timestamp from the video file container using `cv2.CAP_PROP_POS_MSEC`.
* **Dynamic Waiting:** The worker calculates the real elapsed time since the video started and compares it directly against the video's absolute timestamp. If the processing thread gets ahead of the video, it dynamically sleeps for the precise remainder (`wait_seconds = video_time - elapsed`).
* **Lag Recovery:** If the thread falls behind due to heavy inference overhead, the negative wait duration is ignored, preventing the loop from queueing up "stale" sleep delays and causing an ongoing slowdown.

The updated pipeline workflow in `danceapp.py` is structured as follows:

```text
Video Worker Loop ──> Fetch cv2.CAP_PROP_POS_MSEC ──> Compare with elapsed monotonic time
                                                                   │
    ┌───────────────── Catch up / Skip Sleep <─────────────────────┤ (If system lags behind video)
    ▼                                                              ▼
Draw Template Pose ──> Update thread-safe buffer ──> Sleep precisely for remaining time delta

```

This absolute time alignment ensures that even if individual frames fluctuate in processing demands, the visual pacing of the reference template remains perfectly synchronized with real-world time.
