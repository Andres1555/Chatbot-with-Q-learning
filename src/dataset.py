# Datos del chatbot: intenciones, respuestas y patrones de clasificacion
# Cada estado (intencion) representa lo que el usuario pregunta
# Todas las respuestas forman un pool global (A0-A4) para que el bot aprenda

# -- Respuestas del bot organizadas por intencion --
# Solo las intenciones con respuestas generan acciones en el pool global
_INTENT_RESPONSES = {
    "saludo": ["Hola! Soy el asistente de la UNEG. En que puedo ayudarte?"],
    "despedida": ["Gracias por comunicarte con la UNEG! Que tengas un excelente dia."],
    "carrera": ["Ing. Informatica, Ing. Industrial, Administracion, Contaduria, Educacion, Idiomas y mas."],
    "inscripcion": ["Inscripciones abiertas en julio. Registro en www.uneg.edu.ve y entrega de recaudos."],
    "requisitos": ["Titulo de bachiller, notas certificadas, cedula, fotos carnet, partida de nacimiento."],

    "default": []
}

# -- Patrones regex para clasificar la intencion --
_INTENT_PATTERNS = {
    "saludo": [
        r"\bhola\b", r"\bbuenos di[ai]s\b", r"\bbuenas tardes\b",
        r"\bbuenas noches\b", r"\bqu[eé] tal\b", r"\bhey\b",
        r"\bhi\b", r"\bhello\b", r"\bbuenas\b", r"\bsaludos\b",
        r"\bbuen dia\b", r"\bbrother\b"
    ],
    "despedida": [
        r"\badi[oó]s\b", r"\bchao\b", r"\bhasta luego\b",
        r"\bhasta pronto\b", r"\bbye\b", r"\bnos vemos\b",
        r"\bme voy\b", r"\bterminamos\b", r"\bchau\b"
    ],
    "carrera": [
        r"\bcarrera\b", r"\bcarreras\b", r"\bprogramas?\b",
        r"\bqu[eé] ofrecen\b", r"\boferta académica\b",
        r"\bpensum\b", r"\bestudio\b", r"\bestudiar\b",
        r"\bingenier[aí]a\b", r"\badministración\b",
        r"\bcontadur[aí]a\b", r"\beducación\b",
        r"\bidiomas\b", r"\blicenciatura\b",
        r"\bpregrado\b", r"\bpostgrado\b",
        r"\bmaestr[aí]a\b", r"\bespecialización\b"
    ],
    "inscripcion": [
        r"\binscripci[oó]n\b", r"\binscripciones\b",
        r"\binscribirme\b", r"\bmatr[ií]cula\b",
        r"\bregistro\b", r"\binscribir\b",
        r"\bc[oó]mo me inscribo\b", r"\bproceso de admisión\b",
        r"\badmisión\b", r"\bprueba interna\b",
        r"\bprueba de admisión\b", r"\bingreso\b",
        r"\bentrar a la u\b", r"\bentrar a la universidad\b"
    ],
    "requisitos": [
        r"\brequisitos\b", r"\brequisito\b", r"\bdocumentos\b",
        r"\brecaudos\b", r"\bnecesito para\b",
        r"\bqu[eé] necesito\b", r"\brequerimientos\b",
        r"\bnotas certificadas\b", r"\btítulo de bachiller\b",
        r"\bc[oá]mo entro\b"
    ],

    "default": []
}

STATE_KEYS = list(_INTENT_RESPONSES.keys())

STATE_NAMES = {k: f"S{i}: {k.replace('_', ' ').title()}" for i, k in enumerate(STATE_KEYS)}

# -- Pool global de respuestas (todas las acciones) --
def build_response_pool():
    pool = []
    for state_key in STATE_KEYS:
        for resp in _INTENT_RESPONSES[state_key]:
            pool.append({"text": resp, "origin_intent": state_key})
    return pool

ALL_RESPONSES = build_response_pool()
TOTAL_ACTIONS = len(ALL_RESPONSES)

# -- Dataset compilado para el clasificador --
DATASET = {}
for key in STATE_KEYS:
    DATASET[key] = {
        "patterns": _INTENT_PATTERNS[key],
        "responses": _INTENT_RESPONSES[key]
    }
