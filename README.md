# 🚛 Sistema de Percepción y Conducción Autónoma Visual en Euro Truck Simulator 2

Agente de conducción autónoma que observa la salida visual de Euro Truck Simulator 2 en primera persona y controla el camión mediante emulación externa, usando YOLO11 y Árboles de Comportamiento.

---

## 🎯 Objetivo

Desarrollar un sistema capaz de conducir un camión en ETS2 utilizando **exclusivamente percepción visual** (sin modificar el juego ni acceder a APIs internas), tomando decisiones mediante un Árbol de Comportamiento y emulando teclado/mouse vía Parsec.

---

## 🏗️ Arquitectura

```
┌─────────────────────┐     WiFi     ┌──────────────────────────────────┐
│  Mac M3 (16GB RAM)   │◄──Parsec───►│  Mac M1 (8GB RAM)                │
│  ├─ ETS2 Demo         │              │  ├─ MSS Captura (720p@15Hz)      │
│  └─ Parsec Host       │              │  ├─ YOLO11n (MPS) → Detección    │
└─────────────────────┘              │  ├─ Minmapa CV → GPS vector     │
                                       │  ├─ Behavior Tree → Decisión   │
                                       │  └─ pyautogui → Mouse+Teclado  │
                                       └──────────────────────────────────┘
```

- **Percepción**: YOLO11n sobre frames completos + CV clásico para minimapa y semáforos
- **Decisión**: Árbol de Comportamiento con 7 prioridades
- **Actuación**: Mouse steering + teclado vía pyautogui → Parsec → ETS2
- **Latencia total**: ~36-75ms en WiFi local

---

## 🚀 Instalación Rápida

### Requisitos
- macOS 14+ (Apple Silicon M1 o superior)
- Python 3.11+
- ETS2 (Demo o Full) corriendo en otra Mac con Parsec Host
- Ambas máquinas en misma red WiFi/Ethernet

### Setup

```bash
# 1. Clonar
git clone <repo-url>
cd ets2-autonomous

# 2. Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar pesos YOLO11n
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
mv yolo11n.pt models/

# 5. Configurar
cp configs/ets2_demo.yaml config.yaml
# Editar región de captura, sensibilidad, etc.

# 6. En la Mac con ETS2: iniciar Parsec Host
# 7. En la Mac M1: iniciar Parsec Client y conectar

# 8. Ejecutar agente
python src/main.py
# Presionar Q para paro de emergencia
```

---

## 📁 Estructura del Proyecto

```
ets2-autonomous/
├── README.md                   ← Este archivo
├── DOMAIN.md                   ← Lenguaje ubicuo del dominio
├── PLAN.md                     ← Plan de implementación y fases
├── IMPLEMENTATION.md           ← Arquitectura y detalles técnicos
├── requirements.txt            ← Dependencias
├── config.yaml                 ← Configuración local
├── src/
│   ├── main.py                 ← Loop principal 15Hz
│   ├── capture/                ← Captura de pantalla (MSS)
│   ├── perception/             ← YOLO + CV clásico
│   ├── decision/               ← Behavior Tree + comportamientos
│   ├── actuation/              ← pyautogui mouse/teclado
│   └── utils/                  ← Logging y métricas
├── configs/                    ← Configuraciones predefinidas
├── models/                     ← Pesos de modelos ML
├── data/                       ← Capturas y logs
└── docs/                       ← Reportes y video
```

---

## 🧠 Toma de Decisiones (Behavior Tree)

```
Root (Selector: prioridad ↓)
├── [1] EmergencyStop     ← Peligro inminente
├── [2] ObstacleAvoid     ← Vehículo/peatón cercano
├── [3] TrafficLightStop  ← Semáforo rojo
├── [4] StopSignStop      ← Señal de alto
├── [5] YieldPedestrian   ← Ceder paso a peatón
├── [6] LaneFollow        ← Seguir GPS
└── [7] Cruise            ← Avance normal
```

---

## 📊 Escenarios de Validación

| # | Escenario | Estado |
|---|-----------|--------|
| 1 | Autopista sin obstáculos | ⬜ Pendiente |
| 2 | Vehículo delante (frenado) | ⬜ Pendiente |
| 3 | Semáforo rojo → verde | ⬜ Pendiente |
| 4 | Señal de alto | ⬜ Pendiente |
| 5 | Peatón en zona de conflicto | ⬜ Pendiente |
| 6 | Pérdida de detección | ⬜ Pendiente |
| 7 | Maniobra de rebase | ⬜ Fase 4 |
| 8 | Cruce de intersección | ⬜ Fase 2 |

---

## 🔧 Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.11+ |
| Detección | YOLO11n (Ultralytics) |
| Captura | python-mss |
| Emulación | pyautogui |
| CV Clásico | OpenCV |
| Decisión | py_trees |
| ML Backend | PyTorch 2.x + MPS |
| Streaming | Parsec |

---

## ⚠️ Limitaciones

- **macOS solo**: captura por MSS (sin GPU acceleration), pyautogui para inputs
- **Latencia Parsec**: 15-40ms round-trip en WiFi local
- **ETS2 Demo**: 60 min límite por sesión, sin saves, mapa limitado
- **8GB RAM M1**: justo para YOLO11n + Parsec client + Python
- **Steering por mouse**: menos preciso que volante físico pero funcional
- **Sin telemetría para decisión**: solo visión + GPS del minimapa

---

## 📝 Licencia

Proyecto académico. Ver documento de requerimientos para restricciones de uso.

---

## 📚 Documentación Adicional

- [DOMAIN.md](./DOMAIN.md) — Lenguaje ubicuo y glosario
- [PLAN.md](./PLAN.md) — Fases y plan de implementación
- [IMPLEMENTATION.md](./IMPLEMENTATION.md) — Arquitectura detallada y APIs
- [Documento de Requerimientos](./docs/Proyecto_ETS2_YOLO26_Requerimientos.pdf)
