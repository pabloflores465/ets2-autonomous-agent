"""
Clasificador de color de semáforos usando CV clásico.
Para cada bbox de traffic_light, determina el estado (RED/YELLOW/GREEN).
"""

from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from src.perception.detector import Detection


class LightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class TrafficLightClassifier:
    """Clasifica el color de un semáforo dentro de su bbox."""

    def __init__(self, config: dict):
        cfg = config["perception"]["traffic_light"]
        self.red_ranges = [
            (np.array([h[0], cfg["saturation_min"], cfg["value_min"]]),
             np.array([h[1], 255, 255]))
            for h in cfg["red_hue_ranges"]
        ]
        self.yellow_range = (
            np.array([cfg["yellow_hue_range"][0], cfg["saturation_min"], cfg["value_min"]]),
            np.array([cfg["yellow_hue_range"][1], 255, 255]),
        )
        self.green_range = (
            np.array([cfg["green_hue_range"][0], cfg["saturation_min"], cfg["value_min"]]),
            np.array([cfg["green_hue_range"][1], 255, 255]),
        )
        self.min_pixels = 25  # mínimo píxeles para considerar un color

    def classify(self, frame: np.ndarray, detection: Detection) -> LightState:
        """
        Determina el estado del semáforo.
        Args:
            frame: imagen completa (H, W, 3) en BGR
            detection: bbox de clase traffic_light
        Returns:
            LightState
        """
        x1, y1, x2, y2 = detection.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return LightState.UNKNOWN

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return LightState.UNKNOWN

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Contar píxeles de cada color
        red_pixels = 0
        for lower, upper in self.red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_pixels += cv2.countNonZero(mask)

        yellow_mask = cv2.inRange(hsv, self.yellow_range[0], self.yellow_range[1])
        yellow_pixels = cv2.countNonZero(yellow_mask)

        green_mask = cv2.inRange(hsv, self.green_range[0], self.green_range[1])
        green_pixels = cv2.countNonZero(green_mask)

        # Determinar estado por mayoría
        counts = {
            LightState.RED: red_pixels,
            LightState.YELLOW: yellow_pixels,
            LightState.GREEN: green_pixels,
        }
        best = max(counts, key=counts.get)

        if counts[best] < self.min_pixels:
            return LightState.UNKNOWN

        return best
