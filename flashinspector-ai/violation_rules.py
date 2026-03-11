"""
FlashInspector AI - Violation Detection Rules

Converts raw YOLO detections into fire inspection violations.
Ported from the logic in test_model_video_colab.ipynb.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Class sets that indicate specific violation types
# ---------------------------------------------------------------------------

MISSING_CLASSES = {"empty_mount", "extinguisher_cabinet_empty", "bracket_empty"}
NONCOMPLIANT_CLASSES = {"non_compliant_tag", "noncompliant_tag", "yellow_tag", "red_tag"}
EXIT_DARK_CLASSES = {"exit_sign_dark", "exit_dark", "unlit_exit"}
SMOKE_MISSING_CLASSES = {"smoke_detector_missing", "detector_missing"}
BLOCKED_EXIT_CLASSES = {"blocked_exit", "exit_blocked"}
EXTINGUISHER_CLASSES = {"fire_extinguisher"}
HEIGHT_CHECK_CLASSES = {"notification_appliance", "pull_station"}

ALL_VIOLATION_CLASSES = (
    MISSING_CLASSES | NONCOMPLIANT_CLASSES | EXIT_DARK_CLASSES
    | SMOKE_MISSING_CLASSES | BLOCKED_EXIT_CLASSES
)

# Canonical names for class consolidation
CLASS_MAP = {
    "right exit": "emergency_exit", "left exit": "emergency_exit",
    "Right Exit": "emergency_exit", "Left Exit": "emergency_exit",
    "Straight Exit": "emergency_exit", "straight exit": "emergency_exit",
    "Left-Right Exit": "emergency_exit", "left-right exit": "emergency_exit",
    "emergency exit": "emergency_exit", "Emergency Exit": "emergency_exit",
    "fire-extinguisher": "fire_extinguisher",
    "fire extinguisher": "fire_extinguisher",
    "Fire_Extinguisher": "fire_extinguisher",
    "Fire-Extinguisher": "fire_extinguisher",
    "yellow tag": "yellow_tag",
    "red tag": "red_tag",
    "white tag": "white_tag",
    "Emergency-Light": "emergency_light",
    "emergency-light": "emergency_light",
    "Alarm": "alarm",
    "Alarm(Bell)": "alarm_bell",
    "Alarm(Lever)": "alarm_lever",
    "Alarm (Lever)": "alarm_lever",
}

# Per-class confidence thresholds: violations use lower threshold because
# it's better to flag and let a human review than to miss a real violation.
DEFAULT_CONF = 0.30
CLASS_CONFIDENCE = {
    # Equipment — high confidence
    "fire_extinguisher": 0.40,
    "emergency_exit": 0.40,
    "smoke_detector": 0.40,
    "fire_blanket": 0.40,
    "manual_call_point": 0.40,
    # Violations — lower confidence (better to over-flag)
    "empty_mount": 0.20,
    "extinguisher_cabinet_empty": 0.20,
    "non_compliant_tag": 0.25,
    "yellow_tag": 0.25,
    "red_tag": 0.25,
    "exit_sign_dark": 0.25,
    "smoke_detector_missing": 0.25,
    "blocked_exit": 0.25,
    # Tags
    "white_tag": 0.30,
    # Equipment for height checks
    "notification_appliance": 0.30,
    "pull_station": 0.30,
    "fire_alarm_panel": 0.30,
}

# Height zone: if the center of a detection is in the top fraction of the
# frame, it may be mounted too high (ADA / NFPA compliance).
HEIGHT_WARN_ZONE = 0.25


def consolidate_class(name: str) -> str:
    return CLASS_MAP.get(name, name)


def get_confidence_threshold(class_name: str) -> float:
    return CLASS_CONFIDENCE.get(class_name, DEFAULT_CONF)


def _boxes_nearby(bbox_a: list[float], bbox_b: list[float], margin: float = 80) -> bool:
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b
    return not (
        x2a + margin < x1b or x2b + margin < x1a
        or y2a + margin < y1b or y2b + margin < y1a
    )


# ---------------------------------------------------------------------------
# Violation: labeled types
# ---------------------------------------------------------------------------

VIOLATION_TYPES = {
    "missing": {
        "label": "EXTINGUISHER MISSING",
        "description": "Empty mount/cabinet with no nearby extinguisher",
        "severity": "critical",
    },
    "non_compliant": {
        "label": "NON-COMPLIANT TAG",
        "description": "Red/yellow tag indicating failed inspection",
        "severity": "critical",
    },
    "exit_dark": {
        "label": "EXIT SIGN NOT ILLUMINATED",
        "description": "Dark or non-functioning exit sign",
        "severity": "critical",
    },
    "smoke_missing": {
        "label": "SMOKE DETECTOR MISSING",
        "description": "Detector base on ceiling without unit",
        "severity": "critical",
    },
    "blocked_exit": {
        "label": "EXIT BLOCKED",
        "description": "Emergency exit obstructed by objects",
        "severity": "critical",
    },
    "height_warning": {
        "label": "MOUNTED TOO HIGH",
        "description": "Notification appliance or pull station above normal mounting height",
        "severity": "warning",
    },
}


def check_violations(
    detections: list[dict],
    frame_height: int,
    timestamp: float,
    missing_margin: float = 80,
) -> list[dict]:
    """Analyze a frame's detections and return a list of violations.

    Each detection dict must have: class, confidence, bbox [x1,y1,x2,y2].
    Returns list of violation dicts.
    """
    violations: list[dict] = []

    extinguisher_bboxes = [
        d["bbox"] for d in detections
        if d["class"] in EXTINGUISHER_CLASSES
    ]

    for det in detections:
        cls = det["class"]
        bbox = det["bbox"]
        conf = det["confidence"]

        # Non-compliant tag
        if cls in NONCOMPLIANT_CLASSES:
            violations.append(_make_violation(
                "non_compliant", det, timestamp,
            ))

        # Exit sign dark
        elif cls in EXIT_DARK_CLASSES:
            violations.append(_make_violation(
                "exit_dark", det, timestamp,
            ))

        # Smoke detector missing
        elif cls in SMOKE_MISSING_CLASSES:
            violations.append(_make_violation(
                "smoke_missing", det, timestamp,
            ))

        # Blocked exit
        elif cls in BLOCKED_EXIT_CLASSES:
            violations.append(_make_violation(
                "blocked_exit", det, timestamp,
            ))

        # Empty mount — only a violation if no extinguisher nearby
        elif cls in MISSING_CLASSES:
            has_nearby = any(
                _boxes_nearby(bbox, ext_bb, margin=missing_margin)
                for ext_bb in extinguisher_bboxes
            )
            if not has_nearby:
                violations.append(_make_violation(
                    "missing", det, timestamp,
                ))

        # Height check for notification appliances and pull stations
        if cls in HEIGHT_CHECK_CLASSES:
            center_y = (bbox[1] + bbox[3]) / 2
            if center_y < frame_height * HEIGHT_WARN_ZONE:
                v = _make_violation("height_warning", det, timestamp)
                v["equipment_type"] = cls
                violations.append(v)

    return violations


def _make_violation(vtype: str, detection: dict, timestamp: float) -> dict:
    info = VIOLATION_TYPES[vtype]
    return {
        "type": vtype,
        "label": info["label"],
        "severity": info["severity"],
        "class": detection["class"],
        "confidence": detection["confidence"],
        "bbox": detection["bbox"],
        "timestamp_sec": timestamp,
    }
