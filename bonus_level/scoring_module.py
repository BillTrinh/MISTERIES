"""Time-aligned scoring for a player already selected by DancerSelector."""

from collections import deque

import numpy as np

from selecting_module import (
    _joint_angles,
    _normalise_pose,
)


KEYPOINT_WEIGHTS = np.asarray(
    [
        0.2, 0.1, 0.1, 0.1, 0.1,   # face
        1.0, 1.0,                  # shoulders
        1.2, 1.2,                  # elbows
        1.4, 1.4,                  # wrists
        1.0, 1.0,                  # hips
        1.3, 1.3,                  # knees
        1.4, 1.4,                  # ankles
    ],
    dtype=np.float32,
)
ANGLE_WEIGHTS = np.asarray(
    [1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.3, 1.3], dtype=np.float32
)
LAG_CANDIDATES = np.arange(0.0, 0.81, 0.1, dtype=np.float32)


class DanceScorer:
    """Score pose, motion, and timing without charging inference latency."""

    def __init__(self, timestamps, keypoints, confidences, valid):
        self.timestamps = np.asarray(timestamps, dtype=np.float32)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.confidences = np.asarray(confidences, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.valid_indices = np.flatnonzero(self.valid)
        self.valid_times = self.timestamps[self.valid_indices]
        self.history = deque()
        self.track_id = None
        self.mirror_mode = None
        self.smoothed = None
        self.last_score_clock = None
        self.weighted_total = 0.0
        self.scored_seconds = 0.0

    def reset_session(self):
        self.history.clear()
        self.track_id = None
        self.mirror_mode = None
        self.smoothed = None
        self.last_score_clock = None
        self.weighted_total = 0.0
        self.scored_seconds = 0.0

    def pause(self):
        """Pause scoring while keeping the accumulated session result."""
        self.history.clear()
        self.track_id = None
        self.mirror_mode = None
        self.smoothed = None
        self.last_score_clock = None

    def _nearest_template_index(self, timestamp):
        if not len(self.valid_indices):
            return None
        position = int(np.searchsorted(self.valid_times, timestamp))
        choices = []
        if position < len(self.valid_times):
            choices.append(position)
        if position > 0:
            choices.append(position - 1)
        if not choices:
            return None
        nearest = min(
            choices,
            key=lambda index: abs(float(self.valid_times[index]) - timestamp),
        )
        if abs(float(self.valid_times[nearest]) - timestamp) > 0.12:
            return None
        return int(self.valid_indices[nearest])

    @staticmethod
    def _prepare_candidate(keypoints, confidence, mirrored):
        return _normalise_pose(keypoints, confidence, mirror=mirrored)

    def _pose_components(
        self, candidate_points, candidate_conf, template_index, mirrored
    ):
        candidate, candidate_visible = self._prepare_candidate(
            candidate_points, candidate_conf, mirrored
        )
        template, template_visible = _normalise_pose(
            self.keypoints[template_index],
            self.confidences[template_index],
            mirror=False,
        )
        if candidate is None or template is None:
            return None

        common = candidate_visible & template_visible
        common_weight = float(np.sum(KEYPOINT_WEIGHTS[common]))
        template_weight = float(np.sum(KEYPOINT_WEIGHTS[template_visible]))
        if np.sum(common) < 6 or template_weight <= 0:
            return None
        coverage = common_weight / template_weight

        distances = np.linalg.norm(candidate[common] - template[common], axis=1)
        coordinate_error = float(
            np.average(distances, weights=KEYPOINT_WEIGHTS[common])
        )
        coordinate_score = float(np.exp(-((coordinate_error / 0.65) ** 2)))

        candidate_angles, candidate_angle_valid = _joint_angles(
            candidate, candidate_visible
        )
        template_angles, template_angle_valid = _joint_angles(
            template, template_visible
        )
        common_angles = candidate_angle_valid & template_angle_valid
        if np.any(common_angles):
            differences = np.abs(
                candidate_angles[common_angles] - template_angles[common_angles]
            )
            per_angle = np.exp(-((differences / np.deg2rad(35.0)) ** 2))
            angle_score = float(
                np.average(per_angle, weights=ANGLE_WEIGHTS[common_angles])
            )
            pose_score = 0.60 * angle_score + 0.40 * coordinate_score
        else:
            angle_score = coordinate_score
            pose_score = coordinate_score

        return {
            "pose": pose_score,
            "angle": angle_score,
            "coordinate": coordinate_score,
            "coverage": coverage,
            "candidate": candidate,
            "candidate_visible": candidate_visible,
            "template": template,
            "template_visible": template_visible,
        }

    @staticmethod
    def _motion_pair_score(previous, current):
        candidate_common = (
            previous["candidate_visible"] & current["candidate_visible"]
        )
        template_common = (
            previous["template_visible"] & current["template_visible"]
        )
        common = candidate_common & template_common
        if np.sum(common) < 4:
            return None

        candidate_delta = (
            current["candidate"][common] - previous["candidate"][common]
        )
        template_delta = current["template"][common] - previous["template"][common]
        candidate_magnitude = np.linalg.norm(candidate_delta, axis=1)
        template_magnitude = np.linalg.norm(template_delta, axis=1)

        moving = template_magnitude >= 0.025
        if np.sum(moving) < 2:
            return None
        candidate_delta = candidate_delta[moving]
        template_delta = template_delta[moving]
        candidate_magnitude = candidate_magnitude[moving]
        template_magnitude = template_magnitude[moving]

        denominator = np.maximum(
            candidate_magnitude * template_magnitude, 1e-6
        )
        cosine = np.sum(candidate_delta * template_delta, axis=1) / denominator
        direction_score = (np.clip(cosine, -1.0, 1.0) + 1.0) * 0.5
        magnitude_score = np.minimum(
            candidate_magnitude, template_magnitude
        ) / np.maximum(candidate_magnitude, template_magnitude)
        joint_scores = 0.70 * direction_score + 0.30 * magnitude_score
        return float(np.mean(joint_scores))

    def _evaluate_alignment(self, observations, lag, mirrored):
        aligned = []
        pose_scores = []
        coverages = []
        for observation in observations:
            template_index = self._nearest_template_index(
                observation["template_time"] - lag
            )
            if template_index is None:
                continue
            components = self._pose_components(
                observation["keypoints"],
                observation["confidence"],
                template_index,
                mirrored,
            )
            if components is None:
                continue
            aligned.append(components)
            pose_scores.append(components["pose"])
            coverages.append(components["coverage"])

        if len(pose_scores) < 2:
            return None

        motion_scores = []
        for previous, current in zip(aligned, aligned[1:]):
            motion_score = self._motion_pair_score(previous, current)
            if motion_score is not None:
                motion_scores.append(motion_score)

        pose_score = float(np.mean(pose_scores))
        coverage = float(np.mean(coverages))
        motion_informative = bool(motion_scores)
        motion_score = float(np.mean(motion_scores)) if motion_scores else pose_score
        alignment_score = (
            0.75 * pose_score + 0.25 * motion_score
            if motion_informative
            else pose_score
        )
        return {
            "alignment": alignment_score,
            "pose": pose_score,
            "motion": motion_score,
            "motion_informative": motion_informative,
            "coverage": coverage,
            "lag": float(lag),
            "mirrored": bool(mirrored),
        }

    def _best_alignment(self, observations):
        mirror_options = (
            (self.mirror_mode,) if self.mirror_mode is not None else (False, True)
        )
        candidates = []
        for mirrored in mirror_options:
            for lag in LAG_CANDIDATES:
                result = self._evaluate_alignment(
                    observations, float(lag), mirrored
                )
                if result is not None:
                    candidates.append(result)
        if not candidates:
            return None
        return max(candidates, key=lambda result: result["alignment"])

    @staticmethod
    def _timing_score(lag):
        if lag <= 0.15:
            return 1.0
        return float(max(0.0, 1.0 - (lag - 0.15) / 0.65))

    def update(
        self,
        track_id,
        keypoints,
        confidence,
        capture_clock,
        template_time_at_capture,
    ):
        if template_time_at_capture is None:
            self.pause()
            return None

        if self.track_id != track_id:
            self.history.clear()
            self.track_id = track_id
            self.mirror_mode = None
            self.smoothed = None
            self.last_score_clock = None

        self.history.append(
            {
                "clock": float(capture_clock),
                "template_time": float(template_time_at_capture),
                "keypoints": np.asarray(keypoints, dtype=np.float32).copy(),
                "confidence": np.asarray(confidence, dtype=np.float32).copy(),
            }
        )
        while (
            self.history
            and capture_clock - self.history[0]["clock"] > 1.2
        ):
            self.history.popleft()
        if len(self.history) < 3:
            return None

        observations = list(self.history)
        result = self._best_alignment(observations)
        if result is None:
            return None

        history_span = observations[-1]["clock"] - observations[0]["clock"]
        if self.mirror_mode is None and history_span >= 1.0:
            self.mirror_mode = result["mirrored"]

        timing = self._timing_score(result["lag"])
        if result["motion_informative"]:
            raw_total = (
                0.60 * result["pose"]
                + 0.25 * result["motion"]
                + 0.15 * timing
            )
        else:
            raw_total = 0.85 * result["pose"] + 0.15 * timing

        coverage_factor = min(1.0, result["coverage"] / 0.75)
        values = {
            "pose": 100.0 * result["pose"],
            "motion": 100.0 * result["motion"],
            "timing": 100.0 * timing,
            "coverage": 100.0 * result["coverage"],
            "total": 100.0 * raw_total * coverage_factor,
            "lag": result["lag"],
            "mirrored": result["mirrored"],
            "motion_informative": result["motion_informative"],
        }

        if self.smoothed is None:
            self.smoothed = values.copy()
        else:
            elapsed = max(
                0.0,
                capture_clock
                - (
                    self.last_score_clock
                    if self.last_score_clock is not None
                    else capture_clock
                ),
            )
            alpha = 1.0 - np.exp(-elapsed / 0.35)
            for key in ("pose", "motion", "timing", "coverage", "total", "lag"):
                self.smoothed[key] = (
                    (1.0 - alpha) * self.smoothed[key] + alpha * values[key]
                )
            self.smoothed["mirrored"] = values["mirrored"]
            self.smoothed["motion_informative"] = values["motion_informative"]

        if self.last_score_clock is not None:
            elapsed = float(
                np.clip(capture_clock - self.last_score_clock, 0.0, 0.5)
            )
            self.weighted_total += self.smoothed["total"] * elapsed
            self.scored_seconds += elapsed
        self.last_score_clock = float(capture_clock)

        output = self.smoothed.copy()
        output["overall"] = (
            self.weighted_total / self.scored_seconds
            if self.scored_seconds > 0
            else output["total"]
        )
        output["scored_seconds"] = self.scored_seconds
        return output
