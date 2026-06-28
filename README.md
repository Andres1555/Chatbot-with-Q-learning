# UNEG - Chatbot con Q-Learning (Contextual Bandit)

Chatbot para la Universidad Nacional Experimental de Guayana que aprende a responder mediante refuerzo: el usuario califica cada respuesta con Like (+1) o Dislike (-1) y el bot mejora progresivamente.

## Requisitos

- Python 3.8+
- Tkinter (incluido con Python en Windows)

## Instalacion

```bash
git clone <repo>
cd "Chatbot recompensa y castigo"
python main.py
```

## Estructura del proyecto

```
├── main.py              # Punto de entrada
├── src/
│   ├── dataset.py       # Datos: intenciones, respuestas y patrones
│   ├── classifier.py    # Clasificador de intenciones por regex
│   └── agent.py         # Agente Q-Learning (Contextual Bandit)
└── gui/
    └── app.py           # Interfaz grafica Tkinter
```

## Funcionamiento

### 1. El usuario escribe un mensaje
```
"hola" → classifier.classify() → "saludo"
```

### 2. El bot elige una respuesta
```python
agent.select_action("saludo")
# 30% azar (explorar), 70% la de mayor Q (explotar)
```

### 3. El usuario califica
- **Like (+1)**: la respuesta le gusto → sube el Q-valor
- **Dislike (-1)**: no le gusto → baja el Q-valor

### 4. El agente aprende
```
Q(s,a) ← Q(s,a) + α · [R - Q(s,a)]
```

| Parametro | Valor | Significado |
|---|---|---|
| α (alpha) | 0.3 | Tasa de aprendizaje |
| ε (epsilon) | 0.3 → 0.01 | Exploracion inicial que se reduce |
| ε decay | 0.96 | Factor de reduccion por feedback |

## Dataset

6 estados (intenciones), 5 acciones (respuestas globales A0-A4):

| Estado | Accion | Respuesta |
|---|---|---|
| saludo | A0 | Saludo institucional |
| despedida | A1 | Despedida |
| carrera | A2 | Programas academicos |
| inscripcion | A3 | Proceso de inscripcion |
| requisitos | A4 | Documentos requeridos |
| default | - | Fallback (sin respuesta propia) |

## Como probar

1. Ejecutar `python main.py`
2. Escribir "hola" y enviar
3. Si la respuesta es incorrecta (ej: habla de carreras), clickear **Dislike**
4. Repetir con "hola" hasta que responda con el saludo
5. Clickear **Like** para reforzar
6. Repetir con otras preguntas: "carreras", "inscripciones", "requisitos"

Despues de varias interacciones, el bot aprende a dar la respuesta correcta para cada pregunta.
