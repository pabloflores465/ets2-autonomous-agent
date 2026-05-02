# DOMAIN.md — Lenguaje Ubicuo del Proyecto

## Proyecto
**Sistema de percepción y toma de decisiones para conducción autónoma visual en Euro Truck Simulator 2**

---

## Entidades del Dominio

### Agente (Agente de Conducción)
Sistema de software que observa la salida visual de ETS2 y emite comandos de control. Corre en la Mac M1 como cliente Parsec. También llamado "conductor virtual" o "el bot".

### Vehículo Ego
El camión controlado por el agente dentro del simulador.

### Escena
Un frame capturado de la ventana de Parsec que contiene la vista completa del juego: cabina, parabrisas, espejos retrovisores y minimapa.

### Zona de Percepción
Región lógica dentro de la escena donde se clasifican las detecciones de YOLO:

| Zona | Descripción | Coordenadas relativas |
|------|-------------|----------------------|
| **Frontal** | Carril propio, obstáculos delante | Tercio central inferior |
| **Espejo Izquierdo** | Vehículos aproximándose por izquierda | Cuadrante superior izquierdo (espejo virtual) |
| **Espejo Derecho** | Vehículos aproximándose por derecha | Cuadrante superior derecho (espejo virtual) |
| **Capó** | Peatones/vehículos inmediatamente al frente | Franja inferior central |
| **Lateral** | Verificación de cruces (cuando gira cámara) | Mitad izquierda/derecha del frame |

### Objeto de Interés
Cualquier entidad detectada por YOLO11n relevante para la conducción: camión, automóvil, autobús, motocicleta, peatón, semáforo, señal de alto.

### Estado de Semáforo
Clasificación del color del semáforo detectado: `RED`, `YELLOW`, `GREEN`. Si YOLO detecta un `traffic light`, un clasificador auxiliar (CV clásico por color en la ROI del bbox) determina el estado.

### Minimapa
Elemento de UI del juego ubicado en la esquina inferior derecha. Contiene:
- **Ruta GPS**: línea roja que indica el camino a seguir
- **Flecha del camión**: indicador verde de posición y orientación del vehículo ego

### Ruta GPS
Trayectoria deseada extraída del minimapa mediante esqueletización de la línea roja. Se reduce a un vector de dirección: `STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`.

### Señal de Alto
Señal de tráfico octagonal roja con texto "STOP". YOLO la detecta como clase `stop sign`. Obliga a detención total.

### Parsec
Software de streaming remoto. Corre en dos roles:
- **Parsec Host**: en la Mac M3, captura ETS2 y lo stremea
- **Parsec Client**: en la Mac M1, recibe el stream y forwardea inputs

### Ciclo de Percepción-Decisión-Actuación
Pipeline que se ejecuta a 15Hz:
1. MSS captura frame de la ventana Parsec (720p)
2. YOLO11n infiere sobre el frame → bounding boxes + clases + confianza
3. Post-procesamiento asigna detecciones a zonas
4. CV clásico procesa ROI del minimapa → vector GPS
5. Árbol de Comportamiento evalúa prioridades y selecciona acción
6. pyautogui ejecuta mouse steering + teclas (W/S/espacio)

### Árbol de Comportamiento (Behavior Tree)
Estructura jerárquica de decisión donde nodos de mayor prioridad interrumpen a los de menor prioridad. Organizado como:

```
Root (Selector de Prioridad)
├── EmergencyStop       ← Peligro inminente
├── CollisionRecovery   ← Recuperación post-choque (reverse + corrección)
├── IntersectionHandler ← Cruce de intersecciones (mirar laterales + espejos)
├── TrafficLightStop    ← Semáforo rojo
├── StopSignStop        ← Señal de alto
├── ObstacleAvoid       ← Vehículo/peatón cercano
├── YieldPedestrian     ← Peatón en zona de conflicto
├── LaneFollow          ← Seguir GPS + centrado de carril
└── Cruise              ← Avanzar libre (70 km/h crucero)
```

### Tick
Una ejecución completa del árbol de comportamiento (15 veces por segundo).

### Latencia Sistémica
Tiempo total desde que el frame se renderiza en ETS2 hasta que el input llega al juego. Incluye: encode Parsec → red → decode → MSS → YOLO → decisión → pyautogui → Parsec → red → input. Estimado: 36-75ms en WiFi local.

---

## Acciones de Conducción

| Acción | Comando | Descripción |
|--------|---------|-------------|
| **Acelerar** | Tecla W presionada | Aceleración constante |
| **Desacelerar** | Soltar W, presionar S | Frenado progresivo |
| **Frenado de Emergencia** | S sostenido + Espacio (freno de mano) | Detención inmediata |
| **Girar Izquierda** | Mouse move negativo en X | Steering proporcional izquierdo |
| **Girar Derecha** | Mouse move positivo en X | Steering proporcional derecho |
| **Centrar Volante** | Mouse moveTo centro | Re-centrar dirección |
| **Mantener Carril** | Micro-correcciones de mouse según GPS | Seguimiento de ruta |
| **Paro Manual** | Tecla Q o botón en ventana | Toma de control humana |
| **Mirar Izquierda** | Camera look mouse left | Girar cámara lateral para cruce |
| **Mirar Derecha** | Camera look mouse right | Girar cámara lateral para cruce |
| **Reverse** | Flecha abajo | Marcha atrás para corrección |

---

## Métricas

| Métrica | Definición | Unidad |
|---------|-----------|--------|
| **FPS Efectivo** | Frames por segundo procesados exitosamente | Hz |
| **Latencia Total** | Tiempo de ciclo completo percepción→actuación | ms |
| **mAP** | Mean Average Precision del detector por clase | % |
| **Eventos de Seguridad** | Colisiones, frenadas tardías, cruces inseguros | conteo |
| **Cumplimiento de Señales** | % de semáforos rojos y señales de alto obedecidos | % |
| **Continuidad** | Tiempo/distancia sin intervención humana | min / km |
| **Falsos Positivos Críticos** | Frenadas o detenciones innecesarias | conteo |

---

## Glosario Técnico

- **SCS**: Software Czech Solutions, desarrollador de ETS2
- **.scs**: Formato de archivo del juego (zip renombrado con assets)
- **MPS**: Metal Performance Shaders, backend de PyTorch para Apple Silicon
- **COCO**: Common Objects in Context, dataset de 80 clases usado para pre-entrenar YOLO
- **ROI**: Region of Interest, sub-rectángulo del frame
- **bbox**: Bounding box, rectángulo que encierra un objeto detectado
- **MSS**: Multiple ScreenShots, librería Python cross-platform de captura
- **PWM**: Pulse Width Modulation aplicada a inputs discretos

### Velocímetro (Speed Detection)
Estimación de velocidad del camión por dos métodos:
- **OCR del tablero**: ROI en display LCD inferior, binarización adaptativa, template matching de dígitos
- **Optical Flow** (fallback): magnitud promedio de movimiento entre frames → calibración a km/h

### Camera Look
Control de cámara lateral para verificar tráfico en intersecciones. El agente gira la vista horizontalmente (mouse lateral) para inspeccionar izquierda y derecha antes de cruzar.

### Intersección
Cruce de vías donde el GPS indica giro fuerte (>50% intensidad). Pipeline:
1. Reducir velocidad a 20 km/h
2. Mirar lateral (cámara + espejo correspondiente)
3. Verificar YOLO en zona lateral y espejo  
4. Cruza si vía libre, espera hasta 5s si hay tráfico
5. Timeout: avanzar con precaución

### Velocidad de Crucero
70 km/h en recta, 45 km/h en curvas, 85 km/h máximo absoluto. Conducción conservadora priorizando estabilidad.

### Colisión (Collision Detection)
Detección visual sin telemetría usando:
- **Optical Flow**: colapso de movimiento (>70% reducción) = impacto
- **Red Flash**: píxeles rojos súbitos en bordes de pantalla = daño

### Recuperación de Colisión (Collision Recovery)
Secuencia post-choque: frenar → esperar 1s → reverse 2s corrigiendo dirección según GPS → retomar.
