# IMPLEMENTATION.md — Arquitectura y Detalles Técnicos

## Stack Tecnológico

| Componente | Tecnología | Versión | Justificación |
|-----------|-----------|---------|---------------|
| Lenguaje | Python | 3.11+ | Ecosistema CV/ML maduro |
| Detección | Ultralytics YOLO | 11n (nano) | ~2.6M params, ~15-25ms en M1 MPS, 80 clases COCO |
| Captura | python-mss | latest | Cross-platform, puro Python, ~3-5ms por frame 720p |
| Emulación | pyautogui | latest | Mouse steering analógico, teclado W/S, compatible Parsec |
| CV Clásico | OpenCV | 4.x | Procesamiento minimapa, clasificador color semáforos |
| Decisión | py_trees | latest | Behavior Tree para lógica de conducción |
| ML Backend | PyTorch | 2.x con MPS | Aceleración GPU Apple Silicon |
| Logging | logging stdlib + structlog | — | Métricas y eventos |
| Config | PyYAML | — | `config.yaml` externo |
| Métricas | NumPy + Pandas | — | Análisis post-prueba |
| Visualización | OpenCV | — | Preview con bboxes overlay |

### Stack Descartado y Por Qué
- **DXcam**: solo Windows, no disponible en macOS → **MSS** en su lugar
- **pydirectinput**: API DirectInput de Windows → **pyautogui** vía Parsec
- **vgamepad**: driver XInput virtual Windows → no aplica en este setup
- **ROS2**: overkill para un agente single-machine → arquitectura plana con BT

---

## Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────────┐
│                      Mac M3 (16GB RAM)                            │
│  ┌────────────────────┐    ┌────────────────┐                    │
│  │  ETS2 Demo         │    │  Parsec Host   │                    │
│  │  (1280×720, 1raP)  │───▶│  (encode+net)  │                    │
│  └────────────────────┘    └───────┬────────┘                    │
└────────────────────────────────────┼──────────────────────────────┘
                                     │  WiFi local
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Mac M1 (8GB RAM)                             │
│  ┌────────────────────┐                                           │
│  │  Parsec Client     │──── frame 720p ─────────────────────┐    │
│  │  (decode+display)  │                                      │    │
│  └────────┬───────────┘                                      ▼    │
│           │ keyboard/mouse ← pyautogui            ┌──────────────┐│
│           │                                        │  MSS Capture ││
│           │                                        │  (ventana    ││
│           │                                        │  Parsec)     ││
│           │                                        └──────┬───────┘│
│           │                                               │ frame  │
│           │                                        ┌──────▼───────┐│
│           │                                        │  YOLO11n     ││
│           │                                        │  (MPS inf.)  ││
│           │                                        └──────┬───────┘│
│           │                                               │ bboxes │
│           │                                        ┌──────▼───────┐│
│           │                                        │  Zones       ││
│           │                                        │  (post-proc) ││
│           │                                        └──────┬───────┘│
│           │                                               │        │
│                                                   ┌──────┬───────┐│
│                                                   │  Lane        ││
│                                                   │  Detector    ││
│                                                   │  (Canny+     ││
│                                                   │  HoughLinesP)││
│                                                   └──────┬───────┘│
│                                                   │ offset/lane  ││
│  ┌────────┴───────────┐                           ┌──────▼───────┐│
│  │  pyautogui          │◀──── mouse coords ───────│  Behavior    ││
│  │  (mouse + W/S/esc)  │      + keys              │  Tree        ││
│  └────────────────────┘                           │  (py_trees)  ││
│                                                   └──────┬───────┘│
│                                                          │ vector │
│                                                   ┌──────▼───────┐│
│                                                   │  Minimap CV  ││
│                                                   │  (red line + ││
│                                                   │   blue arrow)││
│                                                   └──────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
ets2-autonomous/
├── README.md
├── DOMAIN.md
├── PLAN.md
├── IMPLEMENTATION.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── main.py                      # Loop principal 15Hz
│   ├── capture/
│   │   └── screen_grabber.py        # MSS wrapper, captura región Parsec
│   ├── perception/
│   │   ├── detector.py              # YOLO11n cargo + inferencia
│   │   ├── zones.py                 # Asignación bbox → zona lógica
│   │   ├── traffic_light_classifier.py  # CV color dentro del bbox
│   │   └── minimap.py              # ROI + filtro rojo + skeletonize
│   ├── decision/
│   │   ├── behavior_tree.py         # Árbol principal py_trees
│   │   ├── behaviors/
│   │   │   ├── emergency_stop.py    # Peligro inminente
│   │   │   ├── obstacle_avoid.py    # Vehículo/peatón cercano
│   │   │   ├── traffic_light.py     # Rojo/Amarillo/Verde
│   │   │   ├── stop_sign.py         # Señal de alto
│   │   │   ├── yield_pedestrian.py  # Ceder a peatón
│   │   │   ├── lane_follow.py       # Seguir GPS
│   │   │   └── cruise.py            # Avance normal
│   │   └── context.py               # Estado del mundo (world model)
│   ├── actuation/
│   │   └── controller.py            # Mouse steering + keyboard via pyautogui
│   └── utils/
│       ├── logger.py                # Logging de eventos y métricas
│       └── metrics.py               # Cálculo de métricas en runtime
├── tests/
│   ├── test_capture.py
│   ├── test_detector.py
│   ├── test_minimap.py
│   ├── test_behavior_tree.py
│   └── test_controller.py
├── configs/
│   ├── ets2_demo.yaml               # Config para ETS2 Demo
│   └── ets2_full.yaml               # Config para ETS2 Full
├── models/
│   └── yolo11n.pt                    # Pesos YOLO descargados
├── data/
│   ├── captures/                     # Screenshots de prueba
│   └── logs/                         # Logs de sesiones
└── docs/
    ├── reporte_tecnico.pdf
    └── video_demostrativo.mp4
```

---

## Configuración de ETS2

Para reproducibilidad, fijar en el juego:

| Parámetro | Valor |
|-----------|-------|
| Resolución | 1280×720 (ventana) |
| Modo pantalla | Ventana sin bordes |
| FOV | 70° (default) |
| Cámara | Cabina (1ra persona) |
| Control steering | Mouse X-axis |
| Acelerar | W |
| Frenar | S |
| Freno de mano | Espacio |
| Sensibilidad mouse | 0.40 (configurable) |
| Hora del día | Mediodía (máxima iluminación) |
| Clima | Despejado |

---

## Árbol de Comportamiento Detallado

```
Root (Selector: prioridad de arriba hacia abajo)
│
├── [1] EmergencyStop (Sequence)
│   ├── Condición: obstacle_ttc < 1.0s (tiempo a colisión)
│   ├── Condición: obstacle_in_frontal_zone AND confidence > 60%
│   └── Acción: full_brake() + handbrake()
│
├── [2] TrafficLightStop (Sequence)
│   ├── Condición: traffic_light in frontal_zone
│   ├── Condición: light_state == RED
│   └── Acción: progressive_brake() hasta STOP
│
├── [3] StopSignStop (Sequence)
│   ├── Condición: stop_sign in frontal_zone AND confidence > 50%
│   ├── Condición: vehicle_not_stopped() OR approach_zone
│   └── Acción: stop_and_wait(3s)
│
├── [4] ObstacleAvoid (Sequence)
│   ├── Condición: vehicle/pedestrian in frontal_zone
│   ├── Condición: distance_estimate < safe_distance
│   └── Acción: reduce_speed() OR steer_away()
│
├── [5] YieldPedestrian (Sequence)
│   ├── Condición: pedestrian in frontal/capo_zone
│   └── Acción: full_stop() hasta pedestrian_clear
│
├── [6] LaneFollow (Sequence)
│   ├── Condición: gps_vector available OR lane_info present
│   ├── Acción: GPS direction (60%) + lane centering offset (40%)
│   └── Fallback: dirt road → GPS only, no lane correction
│
└── [7] Cruise (Action)
    └── Acción: accelerate(W) + minor_steering_corrections()
```

### Arquitecturas de Decisión Alternativas (Candidatas)

#### Árbol de Comportamiento (Seleccionado) ✅
- **Pros**: Prioridades explícitas, componible, testeable por nodo, maduro (py_trees)
- **Contras**: Lógica de transición entre comportamientos puede requerir states previos
- **Ideal para**: Sistemas con comportamientos jerárquicos bien definidos

#### Sistema de Reglas con Prioridad (Utility-based)
- **Pros**: Scores continuos → transiciones suaves, no binarias. Fácil tunear pesos
- **Contras**: Más difícil de depurar (por qué score X ganó a Y?), menos explicable
- **Ideal para**: Comportamientos con solapamiento (ej: frenar un poco por curva + frenar más por obstáculo)

#### Subsumption (Brooks)
- **Pros**: Extremadamente robusto, capas independientes, usado en robótica real
- **Contras**: Requiere suprimir/inhibir señales entre capas, complejo de sintonizar
- **Ideal para**: Sistemas reactivos puros con sensores ruidosos

---

## Pipeline de Percepción

### Captura (screen_grabber.py)
```
MSS captura monitor primario a 720p
  → Recorte a región de ventana Parsec (configurable en config.yaml)
  → Conversión BGR → RGB (MSS da BGRA, OpenCV usa BGR)
  → Frame numpy array (720, 1280, 3)
```

### Detección YOLO (detector.py)
```python
model = YOLO("models/yolo11n.pt")
model.to("mps")  # Metal Performance Shaders
results = model(frame, conf=0.35, iou=0.45, verbose=False)
# → List[bbox(x1,y1,x2,y2), class_id, confidence]
```

**Clases COCO relevantes (mapeo YOLO11 → dominio):**
| COCO ID | Clase COCO | Dominio |
|---------|-----------|---------|
| 2 | car | automóvil |
| 7 | truck | camión |
| 5 | bus | autobús |
| 3 | motorcycle | motocicleta |
| 0 | person | peatón |
| 9 | traffic light | semáforo |
| 11 | stop sign | señal de alto |

### Zonas (zones.py)
Post-procesamiento puro (sin segunda inferencia). Cada bbox se asigna a zona por coordenadas relativas:
```
frame_width, frame_height = 1280, 720

ZONAS = {
    "frontal":       (0.15, 0.30, 0.85, 0.85),   # x1%, y1%, x2%, y2%
    "capo":          (0.15, 0.78, 0.85, 1.00),
    "espejo_izq":    (0.00, 0.00, 0.15, 0.35),
    "espejo_der":    (0.85, 0.00, 1.00, 0.35),
}
```

### Clasificador de Semáforo (traffic_light_classifier.py)
Para cada bbox de clase `traffic light`:
```
1. Recortar ROI del bbox
2. Convertir a HSV
3. Contar píxeles en rangos:
   - Rojo:   (0-10, 100-255, 100-255) + (160-180, 100-255, 100-255)
   - Amarillo:(15-35, 100-255, 100-255)
   - Verde:  (45-85, 100-255, 100-255)
4. Estado = color con más píxeles > umbral
```

### Minimapa (minimap.py)
```
1. ROI fija: esquina inferior derecha (configurable)
2. Filtro de color rojo en HSV → máscara binaria de ruta GPS
3. Skeletonize (morphological thinning) → línea de 1px
4. Calcular vector de dirección:
   - PCA sobre el contorno más grande de la línea roja → ángulo principal
   - Output: GPSDirection (STRAIGHT|TURN_LEFT|TURN_RIGHT) + intensidad
5. Detección de flecha azul del camión:
   - Filtro HSV azul (H: 100-130)
   - Buscar contorno más grande (2-8% del área del minimapa)
   - Output: coordenadas normalizadas del centro del camión
```

### Detector de Carriles (lane_detector.py)
```
1. ROI: mitad inferior del frame (55%-95% altura)
2. Preprocesamiento: BGR → Gray → GaussianBlur → Canny
3. HoughLinesP → segmentos de línea
4. Clasificar pendiente: negativa → izq, positiva → der
5. Promediar líneas por lado, extender a todo el alto de ROI
6. Offset: distancia centro carril a centro frame
7. Terracería (DIRT): < 2 líneas → GPS puro sin corrección
8. Fusión: steering = GPS_weight * angle_gps + lane_weight * offset_correction
```

---

## Pipeline de Actuación

### Mouse Steering (controller.py)
```python
import pyautogui

# Configuración
STEERING_SENSITIVITY = 0.40   # px de movimiento por grado
STEERING_CENTER = (640, 360)  # Centro de la ventana de juego

def steer(angle_degrees: float, duration_ms: int):
    """angle_degrees: negativo = izquierda, positivo = derecha"""
    dx = int(angle_degrees * STEERING_SENSITIVITY)
    pyautogui.moveRel(dx, 0, duration=duration_ms / 1000)

def center_steering():
    pyautogui.moveTo(*STEERING_CENTER)

def press_key(key: str, duration_ms: int):
    pyautogui.keyDown(key)
    time.sleep(duration_ms / 1000)
    pyautogui.keyUp(key)

def hold_key(key: str):
    pyautogui.keyDown(key)

def release_key(key: str):
    pyautogui.keyUp(key)
```

### Comandos vs Teclas
| Comando | Tecla | Tipo |
|---------|-------|------|
| Acelerar | W | Hold/Release |
| Frenar | S | Hold/Release |
| Freno mano | Space | Pulse |
| Steering izq | Mouse -X | MoveRel |
| Steering der | Mouse +X | MoveRel |
| Centrar | Mouse ToCenter | MoveTo |
| Paro manual | Q | Pulse (detectado en main loop) |

---

## Loop Principal (main.py)

```python
# Pseudocódigo
TARGET_HZ = 15
FRAME_BUDGET_MS = 1000 / TARGET_HZ  # 66ms

bt_root = build_behavior_tree()
grabber = ScreenGrabber(config)
detector = YOLODetector("models/yolo11n.pt")
controller = Controller()

while not emergency_stop:
    t_start = time.monotonic()

    # Captura
    frame = grabber.capture()

    # Percepción
    detections = detector.infer(frame)
    zones = assign_zones(detections)
    gps_vector = process_minimap(frame)

    # Contexto (world model)
    world_state = build_context(zones, gps_vector)

    # Decisión (tick del BT)
    bt_root.tick_once()
    action = bt_root.blackboard.action

    # Actuación
    controller.execute(action)

    # Logging
    log_cycle(t_start, detections, action)

    # Timeout adaptativo
    elapsed = (time.monotonic() - t_start) * 1000
    if elapsed < FRAME_BUDGET_MS:
        time.sleep((FRAME_BUDGET_MS - elapsed) / 1000)
    else:
        logger.warn(f"Frame over budget: {elapsed:.1f}ms > {FRAME_BUDGET_MS}ms")
```

---

## Plan de Pruebas

### Escenario 1: Autopista sin obstáculos
- GPS recto, sin tráfico
- **Esperado**: CRUISE continuo, 5+ minutos sin intervención

### Escenario 2: Vehículo delante
- GPS recto, un camión lento delante
- **Esperado**: ObstacleAvoid reduce velocidad, mantiene distancia segura

### Escenario 3: Semáforo
- Semáforo rojo → verde
- **Esperado**: STOP en rojo, CRUISE cuando verde

### Escenario 4: Señal de alto
- Intersección con stop
- **Esperado**: detención total, pausa 3s, avance si vía libre

### Escenario 5: Peatón
- Peatón cruzando o en banquina
- **Esperado**: STOP o reducción de velocidad

### Escenario 6: Pérdida de detección
- Simular tapando cámara o bajando confianza
- **Esperado**: RecoveryMode → reducir velocidad → STOP si persiste

---

## Métricas a Registrar por Ciclo

```python
@dataclass
class CycleMetrics:
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
    over_budget: bool
```

---

## Manejo de Intersecciones (intersection.py)

### Detección
- GPS indica `TURN_LEFT` o `TURN_RIGHT` con intensidad > 0.5
- Activación del nodo `IntersectionHandler` (prioridad 3 en BT)

### Pipeline de Cruce
```
Fase 0: Aproximación → reducir a 20 km/h (0.5s)
Fase 1: Look → mirar izquierda (1.5s) + mirar derecha (1.5s)
         Verificar espejos + zonas laterales con YOLO
Fase 2: Wait  → si tráfico detectado, esperar hasta 5s
Fase 3: Cross → steering GPS, acelerar a 15 km/h
Fase 4: Done  → GPS intensity < 0.3, volver cámara al frente
```

### Camera Look (controller.py)
```python
# Mirar 60° a la izquierda
controller.execute(action, camera_look_angle=60.0)
# Mirar 60° a la derecha  
controller.execute(action, camera_look_angle=-60.0)
# Volver al frente
controller.execute(action, camera_look_angle=0.0)
```

---

## Detección de Velocidad (speed_detector.py)

### Método primario: OCR del tablero
1. ROI: 40-60% X, 82-92% Y (display LCD inferior en cabina)
2. Binarización adaptativa (Gaussian, 15x15)
3. Encontrar contornos con forma de dígito (aspect ratio 1.2-4.0)
4. Clasificar dígitos por Hu Moments + template matching sintético
5. Concatenar dígitos → velocidad en km/h

### Método fallback: Optical Flow
1. Farneback optical flow entre frames consecutivos
2. Magnitud promedio del flujo
3. Calibración: `speed_kmh = flow_mag * flow_scale` (default 50.0)
4. Ventana de suavizado: 5 frames

### Conducción Conservadora
| Contexto | Velocidad objetivo | Acción |
|----------|-------------------|--------|
| Recta, vía libre | 70 km/h | W presionada |
| Curva (GPS > 0.3) | 45 km/h | Soltar W |
| Aproximación intersección | 20 km/h | Brake suave |
| Máximo absoluto | 85 km/h | Soltar W |
| Arranque (< 10 km/h) | — | W a fondo |
