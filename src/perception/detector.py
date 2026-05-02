"""
Detector de objetos basado en YOLO11n (Ultralytics).
Usa Metal Performance Shaders (MPS) en Apple Silicon.
"""

import time

import numpy as np
from ultralytics import YOLO


class Detection:
    """Una detección individual."""

    def __init__(self, class_id: int, class_name: str, confidence: float, bbox: np.ndarray):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        # bbox: [x1, y1, x2, y2] en píxeles
        self.bbox = bbox
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height

    def __repr__(self):
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox.astype(int).tolist()})"
        )


class YOLODetector:
    """Detector YOLO para percepción visual."""

    CLASSES_OF_INTEREST = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic_light",
        11: "stop_sign",
    }

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "mps",
        confidence: float = 0.35,
        iou: float = 0.45,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.iou = iou
        self._device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Ejecuta inferencia sobre un frame.
        Args:
            frame: np.ndarray (H, W, 3) RGB
        Returns:
            Lista de Detection filtradas por clases de interés.
        """
        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)
        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            if class_id not in self.CLASSES_OF_INTEREST:
                continue
            confidence = boxes.conf[i].item()
            bbox = boxes.xyxy[i].cpu().numpy()
            class_name = self.CLASSES_OF_INTEREST[class_id]
            detections.append(Detection(class_id, class_name, confidence, bbox))

        return detections

    def benchmark(self, frame: np.ndarray, n_runs: int = 50) -> dict:
        """Mide tiempo de inferencia."""
        # warmup
        for _ in range(5):
            self.detect(frame)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.detect(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
        return {
            "n_runs": n_runs,
            "avg_ms": np.mean(times),
            "max_ms": np.max(times),
            "min_ms": np.min(times),
            "device": self._device,
        }

    @property
    def classes(self) -> dict:
        return self.CLASSES_OF_INTEREST
