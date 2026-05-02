"""
Logging y registro de métricas.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class CycleMetrics:
    """Métricas de un ciclo de percepción-decisión-actuación."""

    timestamp: float
    frame_id: int
    fps: float
    capture_ms: float
    inference_ms: float
    decision_ms: float
    actuation_ms: float
    total_ms: float
    detections_count: int
    active_behavior: str
    action: str
    confidence_sum: float
    over_budget: bool


class SessionLogger:
    """Registra eventos y métricas de una sesión de conducción."""

    def __init__(self, log_dir: str = "data/logs", level: str = "INFO"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.jsonl")
        self.event_file = os.path.join(log_dir, f"events_{timestamp}.log")

        # Logger de eventos
        self.logger = logging.getLogger(f"ets2_agent.{timestamp}")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # File handler para eventos
        fh = logging.FileHandler(self.event_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.metrics: list[CycleMetrics] = []
        self.start_time: float = 0.0

    def start_session(self):
        self.start_time = time.monotonic()
        self.logger.info(f"Session {self.session_id} started")

    def log_cycle(self, metrics: CycleMetrics):
        """Registra métricas de un ciclo."""
        self.metrics.append(metrics)

        # Escribir a JSONL
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

        # Log resumido cada 30 ciclos
        if metrics.frame_id % 30 == 0:
            self.logger.info(
                f"Frame {metrics.frame_id:5d} | "
                f"FPS: {metrics.fps:.1f} | "
                f"Total: {metrics.total_ms:.0f}ms | "
                f"Detections: {metrics.detections_count:2d} | "
                f"Behavior: {metrics.active_behavior:20s} | "
                f"{'⚠ OVER' if metrics.over_budget else '       '}"
            )

    def log_event(self, level: str, message: str):
        """Registra un evento."""
        getattr(self.logger, level.lower())(message)

    def end_session(self) -> dict:
        """Finaliza sesión y devuelve resumen."""
        elapsed = time.monotonic() - self.start_time
        total_frames = len(self.metrics)

        if total_frames == 0:
            self.logger.warning("No metrics recorded")
            return {"error": "no data"}

        # Calcular promedios
        avg_total_ms = sum(m.total_ms for m in self.metrics) / total_frames
        avg_fps = sum(m.fps for m in self.metrics) / total_frames
        over_budget_pct = sum(1 for m in self.metrics if m.over_budget) / total_frames * 100

        # Distribución de comportamientos
        behaviors = {}
        for m in self.metrics:
            b = m.active_behavior
            behaviors[b] = behaviors.get(b, 0) + 1

        summary = {
            "session_id": self.session_id,
            "duration_s": elapsed,
            "total_frames": total_frames,
            "avg_fps": round(avg_fps, 1),
            "avg_total_ms": round(avg_total_ms, 1),
            "over_budget_pct": round(over_budget_pct, 1),
            "behavior_distribution": {
                k: round(v / total_frames * 100, 1) for k, v in behaviors.items()
            },
        }

        self.logger.info(f"Session ended. Summary: {json.dumps(summary, indent=2)}")
        return summary
