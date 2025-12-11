from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Agente Q-Learning discreto para Flappy Bird.
    Utiliza estados discretizados y una Q-table.
    """
    def __init__(self, actions, game=None, learning_rate=0.2, discount_factor=0.95,
                 epsilon=0, epsilon_decay=0.995, min_epsilon=0.05,
                 load_q_table_path=None):
        super().__init__(actions, game)

        # Parámetros de aprendizaje
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = list(actions)

        # Inicialización de la Q-table
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print("No se encontró Q-table. Se inicia vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            print("Iniciando Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        # Límites usados para discretizar observaciones continuas
        self.x_bins   = np.linspace(-150, 150, 25)
        self.y_bins   = np.linspace(-300, 300, 70)
        self.v_bins   = np.linspace(-20, 20, 35)
        self.gap_bins = np.linspace(-200, 200, 20)

    def digitize(self, value, bins):
        """Devuelve el índice del bin correspondiente a un valor."""
        return int(np.digitize([value], bins)[0])

    def discretize_state(self, state):
        """Convierte el estado continuo del juego en una tupla discretizada."""
        player_y       = state["player_y"]
        player_vel     = state["player_vel"]
        next_pipe_dist = state["next_pipe_dist_to_player"]
        next_pipe_top  = state["next_pipe_top_y"]
        next_pipe_bot  = state["next_pipe_bottom_y"]

        # Distancia vertical al centro del hueco del tubo
        gap_center = (next_pipe_top + next_pipe_bot) / 2.0
        vertical_center_dist = player_y - gap_center

        # Tamaño real del hueco
        actual_gap = next_pipe_bot - next_pipe_top

        return (
            self.digitize(next_pipe_dist, self.x_bins),
            self.digitize(vertical_center_dist, self.y_bins),
            self.digitize(player_vel, self.v_bins),
            self.digitize(actual_gap, self.gap_bins)
        )

    def act(self, state):
        """Selecciona una acción usando epsilon-greedy."""
        discrete_state = self.discretize_state(state)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        q_values = self.q_table[discrete_state]
        return self.actions[int(np.argmax(q_values))]

    def update(self, state, action, reward, next_state, done):
        """Actualiza la Q-table usando la ecuación de Q-learning."""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        a_idx = self.actions.index(action)
        current_q = self.q_table[discrete_state][a_idx]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next_state])

        self.q_table[discrete_state][a_idx] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self):
        """Reduce epsilon según el factor de decaimiento."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """Guarda la Q-table en un archivo."""
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, path):
        """Carga una Q-table desde archivo."""
        with open(path, 'rb') as f:
            q_dict = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
