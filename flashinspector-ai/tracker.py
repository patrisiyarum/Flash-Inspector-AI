"""
Simple IoU-based object tracker for deduplicating detections across video frames.

Each tracked object gets a unique ID. Detections in consecutive frames are matched
by IoU overlap, so the same physical object is reported once rather than per-frame.
"""

from __future__ import annotations


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackedObject:
    __slots__ = ("track_id", "class_name", "bbox", "confidence", "first_seen",
                 "last_seen", "hit_count", "frames_since_seen", "violations")

    def __init__(self, track_id: int, class_name: str, bbox: list[float],
                 confidence: float, timestamp: float):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.hit_count = 1
        self.frames_since_seen = 0
        self.violations: list[dict] = []

    def update(self, bbox: list[float], confidence: float, timestamp: float):
        self.bbox = bbox
        self.confidence = max(self.confidence, confidence)
        self.last_seen = timestamp
        self.hit_count += 1
        self.frames_since_seen = 0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "best_confidence": round(self.confidence, 3),
            "first_seen_sec": self.first_seen,
            "last_seen_sec": self.last_seen,
            "hit_count": self.hit_count,
            "violations": self.violations,
        }


class SimpleTracker:
    """Match detections across frames using IoU overlap."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self.active: list[TrackedObject] = []
        self.finished: list[TrackedObject] = []

    def update(self, detections: list[dict], timestamp: float) -> list[tuple[dict, TrackedObject]]:
        """Match new detections to existing tracks.

        Args:
            detections: list of dicts with keys "class", "confidence", "bbox"
            timestamp: current timestamp in seconds

        Returns:
            list of (detection_dict, TrackedObject) pairs
        """
        matched_pairs: list[tuple[dict, TrackedObject]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        # Greedy matching by highest IoU
        scores = []
        for d_idx, det in enumerate(detections):
            for t_idx, track in enumerate(self.active):
                if det["class"] == track.class_name:
                    s = iou(det["bbox"], track.bbox)
                    if s >= self.iou_threshold:
                        scores.append((s, d_idx, t_idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        for _, d_idx, t_idx in scores:
            if d_idx in used_dets or t_idx in used_tracks:
                continue
            det = detections[d_idx]
            track = self.active[t_idx]
            track.update(det["bbox"], det["confidence"], timestamp)
            matched_pairs.append((det, track))
            used_dets.add(d_idx)
            used_tracks.add(t_idx)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue
            track = TrackedObject(
                self._next_id, det["class"], det["bbox"], det["confidence"], timestamp
            )
            self._next_id += 1
            self.active.append(track)
            matched_pairs.append((det, track))

        # Age unmatched tracks
        new_active = []
        for t_idx, track in enumerate(self.active):
            if t_idx not in used_tracks:
                track.frames_since_seen += 1
            if track.frames_since_seen <= self.max_age:
                new_active.append(track)
            else:
                self.finished.append(track)
        self.active = new_active

        return matched_pairs

    def get_all_tracks(self) -> list[TrackedObject]:
        return self.finished + self.active
