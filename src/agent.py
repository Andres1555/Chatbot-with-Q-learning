import random
from collections import defaultdict
from src.dataset import STATE_KEYS, ALL_RESPONSES, TOTAL_ACTIONS


def _init_q():
    return defaultdict(lambda: 0.0)


class QLearningAgent:
    def __init__(self, alpha=0.3, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.96):
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(_init_q)
        self.last_update = None
        self.total_interactions = 0
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.rated_actions = defaultdict(set)

    def select_action(self, state_key):
        actions = list(range(TOTAL_ACTIONS))
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.q_table[state_key][a] for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state_key, action_idx, reward):
        old_q = self.q_table[state_key][action_idx]
        q_update = self.alpha * (reward - old_q)
        self.q_table[state_key][action_idx] = old_q + q_update
        new_q = self.q_table[state_key][action_idx]
        self.rated_actions[state_key].add(action_idx)
        origin = ALL_RESPONSES[action_idx]["origin_intent"]
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
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return new_q

    def get_bellman_formula(self):
        return [
            "Ecuación de Bellman (Contextual Bandit):",
            "Q(s,a) ← Q(s,a) + α · [R - Q(s,a)]",
            "",
            f"α (tasa de aprendizaje) = {self.alpha}",
            f"ε (exploración) = {self.epsilon:.4f}",
            f"ε_min = {self.epsilon_min} | Decaimiento = {self.epsilon_decay}",
            f"",
            f"Total acciones disponibles: {TOTAL_ACTIONS}"
        ]

    def get_last_update_str(self):
        if not self.last_update:
            return "Aún no hay actualizaciones"
        u = self.last_update
        lines = [
            f"Estado: {u['state']}",
            f"Acción elegida: A{u['action']} (original de: {u['origin']})",
            f"Q_anterior = {u['old_q']:.4f}",
            f"Recompensa (R) = {u['reward']:+d}",
            f"α · (R - Q_anterior) = {u['alpha']} × ({u['reward']:+d} - {u['old_q']:.4f}) = {u['q_update']:.4f}",
            f"Q_nueva = {u['old_q']:.4f} + ({u['q_update']:.4f}) = {u['new_q']:.4f}"
        ]
        return "\n".join(lines)

    def get_q_table_display(self):
        """Returns (column_indices, rows) for Q-table display.
        Shows A0-A7 plus any rated actions beyond A7 (max 14 cols)."""
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

    def get_action_origin(self, action_idx):
        if 0 <= action_idx < len(ALL_RESPONSES):
            return ALL_RESPONSES[action_idx]["origin_intent"]
        return "?"

    def get_stats(self):
        total = self.positive_rewards + self.negative_rewards
        pct_pos = (self.positive_rewards / total * 100) if total > 0 else 0
        return {
            "total": self.total_interactions,
            "likes": self.positive_rewards,
            "dislikes": self.negative_rewards,
            "acierto": round(pct_pos, 1)
        }
