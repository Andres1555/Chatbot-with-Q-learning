# Clasificador de intenciones: determina que quiere el usuario
# usando expresiones regulares sobre el mensaje de entrada
import re
from src.dataset import DATASET, STATE_KEYS


class IntentClassifier:
    def __init__(self, dataset=None):
        # Permite inyectar un dataset alternativo
        self.dataset = dataset if dataset is not None else DATASET

    # Convierte un mensaje del usuario en una intencion (estado)
    # Recorre cada intencion y busca coincidencias con sus patrones regex
    def classify(self, message):
        msg_lower = message.lower().strip()
        if not msg_lower:
            return "default"
        # Itera sobre todas las intenciones excepto "default"
        for intent, data in self.dataset.items():
            if intent == "default":
                continue
            for pattern in data["patterns"]:
                if re.search(pattern, msg_lower):
                    return intent
        # Si ningun patron coincide, devuelve default
        return "default"

    # Devuelve el indice numerico de una intencion
    def get_state_index(self, intent):
        return STATE_KEYS.index(intent) if intent in STATE_KEYS else len(STATE_KEYS) - 1

    # Devuelve la clave de intencion dado un indice numerico
    def get_state_key(self, index):
        return STATE_KEYS[index] if index < len(STATE_KEYS) else "default"

    # Devuelve las acciones disponibles para una intencion
    def get_available_actions(self, intent):
        return list(range(len(self.dataset[intent]["responses"])))
