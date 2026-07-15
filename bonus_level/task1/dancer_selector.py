"""Temporal multi-person dancer selection for the webcam stream."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


LEFT_RIGHT_MAP = np.asarray(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
)
ANGLE_TRIPLETS = (
    (5, 7, 9),
    (6, 8, 10),
    (7, 5, 11),
    (8, 6, 12),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)
UPPER_BODY = np.asarray([5, 6, 7, 8, 9, 10])
LOWER_BODY = np.asarray([11, 12, 13, 14, 15, 16])


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=deque)


def _normalise_pose(keypoints, confidence, mirror=False):
    points = np.asarray(keypoints, dtype=np.float32).copy()
    confidence = np.asarray(confidence, dtype=np.float32).copy()
    if mirror:
        points = points[LEFT_RIGHT_MAP]
        confidence = confidence[LEFT_RIGHT_MAP]

    visible = (confidence >= 0.3) & np.all(np.isfinite(points), axis=1)
    if np.sum(visible) < 4:
        return None, visible

    if visible[11] and visible[12]:
        center = (points[11] + points[12]) * 0.5
    elif visible[5] and visible[6]:
        center = (points[5] + points[6]) * 0.5
    else:
        center = np.mean(points[visible], axis=0)

    scales = []
    if visible[5] and visible[6]:
        scales.append(np.linalg.norm(points[5] - points[6]))
    if visible[11] and visible[12]:
        scales.append(np.linalg.norm(points[11] - points[12]))
    if visible[5] and visible[6] and visible[11] and visible[12]:
        shoulder_center = (points[5] + points[6]) * 0.5
        hip_center = (points[11] + points[12]) * 0.5
        scales.append(np.linalg.norm(shoulder_center - hip_center))

    scale = max(scales, default=0.0)
    if scale < 1e-4:
        return None, visible

    points = (points - center) / scale
    if mirror:
        points[:, 0] *= -1.0
    points[~visible] = np.nan
    return points, visible


def _joint_angles(points, visible):
    angles = []
    valid_angles = []
    for point_a, vertex, point_c in ANGLE_TRIPLETS:
        if visible[point_a] and visible[vertex] and visible[point_c]:
            vector_a = points[point_a] - points[vertex]
            vector_c = points[point_c] - points[vertex]
            denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_c)
            if denominator > 1e-6:
                cosine = np.clip(np.dot(vector_a, vector_c) / denominator, -1.0, 1.0)
                angles.append(np.arccos(cosine))
                valid_angles.append(True)
                continue
        angles.append(0.0)
        valid_angles.append(False)
    return np.asarray(angles), np.asarray(valid_angles)


def _pose_similarity(candidate_points, candidate_conf, template_points, template_conf):
    best = 0.0
    for mirrored in (False, True):
        candidate, candidate_visible = _normalise_pose(
            candidate_points, candidate_conf, mirror=mirrored
        )
        template, template_visible = _normalise_pose(
            template_points, template_conf, mirror=False
        )
        if candidate is None or template is None:
            continue

        common = candidate_visible & template_visible
        if np.sum(common) < 6:
            continue

        distances = np.linalg.norm(candidate[common] - template[common], axis=1)
        coordinate_score = float(np.exp(-np.mean(distances) / 0.65))

        candidate_angles, candidate_angle_valid = _joint_angles(
            candidate, candidate_visible
        )
        template_angles, template_angle_valid = _joint_angles(
            template, template_visible
        )
        common_angles = candidate_angle_valid & template_angle_valid
        if np.any(common_angles):
            angle_error = np.mean(
                np.abs(candidate_angles[common_angles] - template_angles[common_angles])
            )
            angle_score = float(max(0.0, 1.0 - angle_error / np.pi))
            similarity = 0.60 * coordinate_score + 0.40 * angle_score
        else:
            similarity = coordinate_score
        best = max(best, similarity)
    return best


class DancerSelector:
    """Select the tracked person whose recent motion best follows the template."""

    def __init__(self, timestamps, keypoints, confidences, valid):
        self.timestamps = np.asarray(timestamps, dtype=np.float32)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.confidences = np.asarray(confidences, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.valid_indices = np.flatnonzero(self.valid)
        self.tracks = {}
        self.selected_id = None
        self._selected_missing_since = None
        self._challenger_id = None
        self._challenger_since = None
        self.scores = {}
        self.template_active = False

    def reset(self):
        self.tracks.clear()
        self.selected_id = None
        self._selected_missing_since = None
        self._challenger_id = None
        self._challenger_since = None
        self.scores = {}
        self.template_active = False

    def _has_template_pose(self, timestamp):
        if timestamp is None or timestamp < 0 or not len(self.valid_indices):
            return False
        valid_times = self.timestamps[self.valid_indices]
        position = int(np.searchsorted(valid_times, timestamp))
        choices = []
        if position < len(valid_times):
            choices.append(position)
        if position > 0:
            choices.append(position - 1)
        return bool(
            choices
            and min(abs(float(valid_times[index]) - timestamp) for index in choices)
            <= 0.15
        )

    def _nearest_template_index(self, timestamp):
        if not len(self.valid_indices):
            return None
        valid_times = self.timestamps[self.valid_indices]
        position = int(np.searchsorted(valid_times, timestamp))
        choices = []
        if position < len(valid_times):
            choices.append(position)
        if position > 0:
            choices.append(position - 1)
        if not choices:
            return None
        nearest = min(choices, key=lambda index: abs(float(valid_times[index]) - timestamp))
        if abs(float(valid_times[nearest]) - timestamp) > 0.15:
            return None
        return int(self.valid_indices[nearest])

    def _template_similarity(self, state, now, template_time):
        if template_time is None or template_time < 0:
            return 0.0
        recent = [record for record in state.history if now - record[0] <= 1.0]
        if len(recent) < 2:
            return 0.0

        best_lag_score = 0.0
        # Positive lag means the webcam dancer is behind the template.
        for lag in np.linspace(-0.2, 0.8, 6):
            similarities = []
            for _, points, confidence, _, observed_template_time in recent:
                if observed_template_time is None:
                    continue
                target_time = observed_template_time - float(lag)
                template_index = self._nearest_template_index(target_time)
                if template_index is None:
                    continue
                similarities.append(
                    _pose_similarity(
                        points,
                        confidence,
                        self.keypoints[template_index],
                        self.confidences[template_index],
                    )
                )
            if similarities:
                best_lag_score = max(best_lag_score, float(np.mean(similarities)))
        return best_lag_score

    @staticmethod
    def _activity_score(state, now):
        recent = [record for record in state.history if now - record[0] <= 1.2]
        if len(recent) < 3:
            return 0.0

        upper_speeds = []
        lower_speeds = []
        for previous, current in zip(recent, recent[1:]):
            elapsed = current[0] - previous[0]
            if elapsed <= 0 or elapsed > 0.5:
                continue
            previous_points, previous_visible = _normalise_pose(
                previous[1], previous[2]
            )
            current_points, current_visible = _normalise_pose(current[1], current[2])
            if previous_points is None or current_points is None:
                continue
            common = previous_visible & current_visible

            for group, destination in (
                (UPPER_BODY, upper_speeds),
                (LOWER_BODY, lower_speeds),
            ):
                group_common = group[common[group]]
                if len(group_common) >= 2:
                    displacement = np.linalg.norm(
                        current_points[group_common] - previous_points[group_common],
                        axis=1,
                    )
                    destination.append(float(np.median(displacement) / elapsed))

        def scale_speed(values):
            if not values:
                return 0.0
            speed = float(np.median(values))
            return float(np.clip((speed - 0.04) / 0.70, 0.0, 1.0))

        upper = scale_speed(upper_speeds)
        lower = scale_speed(lower_speeds)
        if max(upper, lower) <= 0:
            return 0.0
        balance = min(upper, lower) / max(upper, lower)
        return 0.5 * (upper + lower) * (0.6 + 0.4 * balance)

    @staticmethod
    def _reliability_score(state, now):
        age_score = min(1.0, (now - state.first_seen) / 1.5)
        recent = [record for record in state.history if now - record[0] <= 1.0]
        if not recent:
            return 0.0
        visibility = float(
            np.mean([np.mean(record[2] >= 0.3) for record in recent])
        )
        return 0.55 * age_score + 0.45 * visibility

    def update(self, detections, now, template_time):
        """Update tracks and return (selected_track_id, component_scores)."""
        seen_ids = set()
        for detection in detections:
            track_id = int(detection["track_id"])
            seen_ids.add(track_id)
            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(first_seen=now, last_seen=now)
                self.tracks[track_id] = state
            state.last_seen = now
            state.history.append(
                (
                    now,
                    detection["keypoints"].copy(),
                    detection["confidence"].copy(),
                    detection["bbox"].copy(),
                    template_time,
                )
            )
            while state.history and now - state.history[0][0] > 2.0:
                state.history.popleft()

        for track_id in list(self.tracks):
            if now - self.tracks[track_id].last_seen > 2.0:
                del self.tracks[track_id]

        self.template_active = self._has_template_pose(template_time)
        if not self.template_active:
            # Keep tracking histories warm, but do not select anyone while the
            # template contains an intro, outro, or another pose-free section.
            self.selected_id = None
            self._selected_missing_since = None
            self._challenger_id = None
            self._challenger_since = None
            self.scores = {}
            return None, {}

        scores = {}
        for track_id, state in self.tracks.items():
            if now - state.last_seen > 0.4:
                continue
            template_score = self._template_similarity(state, now, template_time)
            activity_score = self._activity_score(state, now)
            reliability_score = self._reliability_score(state, now)
            visible = float(np.mean(state.history[-1][2] >= 0.3))
            eligible = now - state.first_seen >= 0.4 and visible >= 0.45
            total = (
                0.55 * template_score
                + 0.30 * activity_score
                + 0.15 * reliability_score
            )
            scores[track_id] = {
                "total": total,
                "template": template_score,
                "activity": activity_score,
                "reliability": reliability_score,
                "eligible": eligible,
            }
        self.scores = scores

        if self.selected_id is not None:
            if self.selected_id in seen_ids:
                self._selected_missing_since = None
            elif self._selected_missing_since is None:
                self._selected_missing_since = now
            elif now - self._selected_missing_since >= 1.2:
                self.selected_id = None
                self._selected_missing_since = None

        eligible_scores = {
            track_id: score
            for track_id, score in scores.items()
            if score["eligible"]
        }
        if not eligible_scores:
            self._challenger_id = None
            self._challenger_since = None
            return self.selected_id, scores

        best_id = max(eligible_scores, key=lambda track_id: eligible_scores[track_id]["total"])
        best_total = eligible_scores[best_id]["total"]

        if self.selected_id is None:
            should_challenge = best_total >= 0.32
        elif best_id == self.selected_id:
            self._challenger_id = None
            self._challenger_since = None
            return self.selected_id, scores
        else:
            current_total = scores.get(self.selected_id, {}).get("total", 0.0)
            should_challenge = best_total >= current_total + 0.12

        if not should_challenge:
            self._challenger_id = None
            self._challenger_since = None
            return self.selected_id, scores

        if self._challenger_id != best_id:
            self._challenger_id = best_id
            self._challenger_since = now
        elif self._challenger_since is not None and now - self._challenger_since >= 0.7:
            self.selected_id = best_id
            self._selected_missing_since = None
            self._challenger_id = None
            self._challenger_since = None

        return self.selected_id, scores
