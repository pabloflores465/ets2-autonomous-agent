# PLAN.md — Plan de Implementación

## Fases del Proyecto

### Fase 1: Pipeline Mínimo Funcional (captura + detección + control básico)
**Objetivo**: Acelerar, frenar, girar y detenerse sin chocar. Sin cruces ni rebases.

| Tarea | Descripción | Output |
|-------|-------------|--------|
| 1.1 | Configurar entorno Python 3.11+ con dependencias | `requirements.txt`, venv |
| 1.2 | Implementar captura MSS de ventana Parsec a 720p | `src/capture/screen_grabber.py` |
| 1.3 | Cargar YOLO11n y ejecutar inferencia sobre frames | `src/perception/detector.py` |
| 1.4 | Post-procesamiento: asignar detecciones a zonas | `src/perception/zones.py` |
| 1.5 | Clasificador de color para semáforos (CV clásico) | `src/perception/traffic_light_classifier.py` |
| 1.6 | Procesamiento de minimapa: ROI + filtro rojo + skeletonize | `src/perception/minimap.py` |
| 1.7 | Árbol de Comportamiento con 4 estados: Cruise, Brake, Stop, Idle | `src/decision/behavior_tree.py` |
| 1.8 | Mouse steering + teclado vía pyautogui | `src/actuation/controller.py` |
| 1.9 | Loop principal a 15Hz con timeout adaptativo | `src/main.py` |
| 1.10 | Paro manual de emergencia (tecla Q) | Integrado en main loop |
| 1.11 | Logger de eventos y métricas básicas | `src/utils/logger.py` |
| 1.12 | Prueba: autopista recta sin tráfico | Video + log |
| 1.13 | Prueba: autopista con vehículo delante | Video + log |

**Métrica de éxito Fase 1**: 5 minutos de conducción sin colisión en autopista recta con tráfico ligero.

---

### Fase 2: Intersecciones y Señalización
**Objetivo**: Detenerse ante semáforos y señales de alto. Cruce básico.

| Tarea | Descripción |
|-------|-------------|
| 2.1 | Agregar nodos TrafficLightStop, StopSignStop al BT |
| 2.2 | Verificación lateral para cruces (cuando GPS indica giro) |
| 2.3 | Nodo CrossIntersection en BT |
| 2.4 | Lógica de prioridad de paso (ceder a vehículos en intersección) |
| 2.5 | Prueba: semáforo rojo → detención → verde → avance |
| 2.6 | Prueba: señal de alto → detención → avance |
| 2.7 | Prueba: cruce con verificación lateral |

**Métrica de éxito Fase 2**: 80% de semáforos y stops obedecidos en circuito urbano simple.

---

### Fase 3: Peatones y Robustez
**Objetivo**: Reaccionar a peatones y recuperarse de fallos de percepción.

| Tarea | Descripción |
|-------|-------------|
| 3.1 | Nodo YieldPedestrian en BT |
| 3.2 | Nodo RecoveryMode (detecciones inconsistentes → reducir velocidad) |
| 3.3 | Suavizado de steering (filtro de media móvil sobre comandos mouse) |
| 3.4 | Prueba: peatón en zona de conflicto |
| 3.5 | Prueba: pérdida temporal de detección |

---

### Fase 4: Rebases y Maniobras Complejas
**Objetivo**: Completar maniobra de rebase segura.

| Tarea | Descripción |
|-------|-------------|
| 4.1 | Nodo Overtake en BT |
| 4.2 | Verificación lateral + trasera para rebase |
| 4.3 | Secuencia: señalizar → cambiar carril → acelerar → retornar |
| 4.4 | Prueba: rebase de vehículo lento |

---

### Fase 5: Evaluación Final
**Objetivo**: Métricas completas, video demostrativo, reporte.

| Tarea | Descripción |
|-------|-------------|
| 5.1 | Dashboard de métricas en tiempo real |
| 5.2 | Pruebas de circuito completo (30+ min) |
| 5.3 | Ablation study: YOLO11n vs YOLO11s |
| 5.4 | Video demostrativo de todos los escenarios |
| 5.5 | Reporte técnico final |

---

## Reglas del Proyecto

### Fuentes de Información Permitidas
- ✅ Percepción visual (YOLO + CV clásico)
- ✅ Minimapa (GPS visual)
- ❌ Telemetría SCS para decisión (solo para evaluación/métricas)
- ❌ Memoria del proceso del juego
- ❌ Mods de automatización

### Restricciones Técnicas
- No modificar archivos del juego
- No inyectar código en el proceso de ETS2
- Interacción externa únicamente vía Parsec + pyautogui
- Vista en primera persona (cabina)
- Rate de decisión: 15Hz fijo

### Prioridades de Conducción
1. **Seguridad primero**: evitar colisiones > obedecer señales > seguir ruta > velocidad
2. **Frenar ante duda**: si confianza de detección < 40%, reducir velocidad
3. **Recuperación segura**: 3 frames sin detecciones → reducir a 30 km/h, 10 frames → STOP

### Principios de Arquitectura
- **Separación clara**: Captura → Percepción → Decisión → Actuación → Registro
- **Cada módulo testeable en aislamiento**
- **Configuración externalizada**: resolución, rutas, umbrales en `config.yaml`
- **Reproducibilidad**: misma resolución, FOV, cámara y horario en pruebas
