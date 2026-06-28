# Agente de Q-Learning (Contextual Bandit)
# Aprende que accion (respuesta) es mejor para cada estado (intencion)
# mediante retroalimentacion del usuario (Like=+1, Dislike=-1)
import random
from collections import defaultdict
from src.dataset import STATE_KEYS, ALL_RESPONSES, TOTAL_ACTIONS


# Inicializa los Q-valores de cada accion en 0.0
def _init_q():
    return defaultdict(lambda: 0.0)


class QLearningAgent:
    def __init__(self, alpha=0.3, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.96):
        # alpha: tasa de aprendizaje (0=nada, 1=reemplazo total)
        self.alpha = alpha
        # epsilon: probabilidad de explorar vs explotar
        self.epsilon = epsilon
        # epsilon_min: valor minimo al que puede llegar epsilon
        self.epsilon_min = epsilon_min
        # epsilon_decay: factor de reduccion de epsilon por actualizacion
        self.epsilon_decay = epsilon_decay
        # Tabla Q: Q[estado][accion] = valor aprendido
        self.q_table = defaultdict(_init_q)
        # Datos de la ultima actualizacion 
        self.last_update = None
        # Contadores de interaccion
        self.total_interactions = 0
        self.positive_rewards = 0
        self.negative_rewards = 0
        # Conjunto de acciones que han sido calificadas por estado
        self.rated_actions = defaultdict(set)

    # Elige una accion para el estado dado usando politica epsilon-greedy
    # Si numero aleatorio < epsilon -> explora (elige al azar)
    # Si no -> explota (elige la accion con mayor Q)
    def select_action(self, state_key):
        actions = list(range(TOTAL_ACTIONS))
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.q_table[state_key][a] for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    # Actualiza la tabla Q con la ecuacion de Bellman (Contextual Bandit):
    # Q(s,a) <- Q(s,a) + alpha * (R - Q(s,a))
    # Donde R es la recompensa (+1 Like, -1 Dislike)
    def update(self, state_key, action_idx, reward):
        old_q = self.q_table[state_key][action_idx]
        q_update = self.alpha * (reward - old_q)
        self.q_table[state_key][action_idx] = old_q + q_update
        new_q = self.q_table[state_key][action_idx]
        # Registra que esta accion fue calificada para este estado
        self.rated_actions[state_key].add(action_idx)
        origin = ALL_RESPONSES[action_idx]["origin_intent"]
        # Guarda datos para mostrar en la UI
        self.last_update = {
            "state": state_key,
            "action": action_idx,
            "origin": origin,
            "old_q": round(old_q, 4),
            "new_q": round(new_q, 4),
            "reward": reward,
            "alpha": self.alpha,
            "q_update": round(q_update, 4)
        }
        self.total_interactions += 1
        if reward > 0:
            self.positive_rewards += 1
        else:
            self.negative_rewards += 1
        # Reduce epsilon progresivamente
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return new_q

    # Devuelve las lineas de la formula de Bellman para la UI
    def get_bellman_formula(self):
        return [
            "Ecuacion de Bellman (Contextual Bandit):",
            "Q(s,a) <- Q(s,a) + alpha * [R - Q(s,a)]",
            "",
            f"alpha (tasa de aprendizaje) = {self.alpha}",
            f"epsilon (exploracion) = {self.epsilon:.4f}",
            f"epsilon_min = {self.epsilon_min} | Decaimiento = {self.epsilon_decay}",
            f"",
            f"Total acciones disponibles: {TOTAL_ACTIONS}"
        ]

    # Devuelve el texto de la ultima actualizacion para la UI
    def get_last_update_str(self):
        if not self.last_update:
            return "Aun no hay actualizaciones"
        u = self.last_update
        lines = [
            f"Estado: {u['state']}",
            f"Accion elegida: A{u['action']} (original de: {u['origin']})",
            f"Q_anterior = {u['old_q']:.4f}",
            f"Recompensa (R) = {u['reward']:+d}",
            f"alpha * (R - Q_anterior) = {u['alpha']} * ({u['reward']:+d} - {u['old_q']:.4f}) = {u['q_update']:.4f}",
            f"Q_nueva = {u['old_q']:.4f} + ({u['q_update']:.4f}) = {u['new_q']:.4f}"
        ]
        return "\n".join(lines)

    # Devuelve los datos para dibujar la tabla Q en la UI
    # Columnas:acciones calificadas adicionales 
    def get_q_table_display(self):
        cols = set(range(min(TOTAL_ACTIONS, 8)))
        for rated_set in self.rated_actions.values():
            cols.update(rated_set)
        col_list = sorted(cols)[:14]

        rows = []
        for state_key in STATE_KEYS:
            row = {"state": state_key}
            for a in col_list:
                row[a] = round(self.q_table[state_key][a], 3)
            rows.append(row)
        return col_list, rows

    # Devuelve la intencion de origen de una accion
    def get_action_origin(self, action_idx):
        if 0 <= action_idx < len(ALL_RESPONSES):
            return ALL_RESPONSES[action_idx]["origin_intent"]
        return "?"

    # Devuelve estadisticas de la sesion
    def get_stats(self):
        total = self.positive_rewards + self.negative_rewards
        pct_pos = (self.positive_rewards / total * 100) if total > 0 else 0
        return {
            "total": self.total_interactions,
            "likes": self.positive_rewards,
            "dislikes": self.negative_rewards,
            "acierto": round(pct_pos, 1)
        }
