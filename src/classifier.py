import re
from src.dataset import DATASET, STATE_KEYS


class IntentClassifier:
    def __init__(self, dataset=None):
        self.dataset = dataset if dataset is not None else DATASET

    def classify(self, message):
        msg_lower = message.lower().strip()
        if not msg_lower:
            return "default"
        for intent, data in self.dataset.items():
            if intent == "default":
                continue
            for pattern in data["patterns"]:
                if re.search(pattern, msg_lower):
                    return intent
        return "default"

    def get_state_index(self, intent):
        return STATE_KEYS.index(intent) if intent in STATE_KEYS else len(STATE_KEYS) - 1

    def get_state_key(self, index):
        return STATE_KEYS[index] if index < len(STATE_KEYS) else "default"

    def get_available_actions(self, intent):
        return list(range(len(self.dataset[intent]["responses"])))
