.PHONY: run lint format test clean install

# ── Ejecución ──
run:
	python src/main.py

# ── Calidad de código ──
lint:
	ruff check src/

format:
	ruff format src/

typecheck:
	mypy src/ --ignore-missing-imports

# ── Tests ──
test:
	python -m pytest tests/ -v

test-capture:
	python -m pytest tests/test_capture.py -v

test-detector:
	python -m pytest tests/test_detector.py -v

# ── Setup ──
install:
	pip install -r requirements.txt
	pip install ruff mypy pytest

install-dev:
	pip install -r requirements.txt
	pip install ruff mypy pytest pytest-cov

# ── Limpieza ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache

clean-logs:
	rm -rf data/logs/*.log data/logs/*.jsonl

# ── Benchmark ──
bench:
	python -c "\
import yaml; \
from src.capture.screen_grabber import ScreenGrabber; \
from src.perception.detector import YOLODetector; \
with open('config.yaml') as f: config = yaml.safe_load(f); \
g = ScreenGrabber(config); \
d = YOLODetector(device='mps'); \
print('MSS:', g.benchmark(30)); \
frame = g.capture(); \
print('YOLO:', d.benchmark(frame, 30))"

# ── Descarga de modelo ──
download-model:
	python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
