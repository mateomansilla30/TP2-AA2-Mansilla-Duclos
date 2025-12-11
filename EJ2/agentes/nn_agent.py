from agentes.base import Agent
import numpy as np
import tensorflow as tf
import pickle
import os

class NNAgent(Agent):
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        self.actions = list(actions)

        # Carga del modelo neuronal previamente entrenado
        self.model = tf.keras.models.load_model(model_path)

        # Carga de parámetros de normalización
        norm_path = 'normalization_params.pkl'
        if os.path.exists(norm_path):
            with open(norm_path, 'rb') as f:
                norm_params = pickle.load(f)
            self.mean = norm_params['mean'].astype(np.float32)
            self.std  = norm_params['std'].astype(np.float32)
        else:
            # Valores por defecto si no hay normalización guardada
            self.mean = np.zeros(4, dtype=np.float32)
            self.std  = np.ones(4, dtype=np.float32)

        # Bins precasteados para discretización rápida
        self.x_bins   = np.linspace(-150, 150, 25).astype(np.float32)
        self.y_bins   = np.linspace(-300, 300, 70).astype(np.float32)
        self.v_bins   = np.linspace(-20, 20, 35).astype(np.float32)
        self.gap_bins = np.linspace(-200, 200, 20).astype(np.float32)

        # Buffer para evitar creación repetida de arrays
        self.state_buf = np.zeros(4, dtype=np.float32)

    @staticmethod
    def fast_digitize(value, bins):
        """Versión optimizada de digitize usando búsqueda binaria."""
        return np.searchsorted(bins, value, side='right')

    def discretize_state(self, state):
        """Convierte el estado continuo a índices discretizados."""
        player_y       = state["player_y"]
        player_vel     = state["player_vel"]
        next_pipe_dist = state["next_pipe_dist_to_player"]
        next_pipe_top  = state["next_pipe_top_y"]
        next_pipe_bot  = state["next_pipe_bottom_y"]

        gap_center = (next_pipe_top + next_pipe_bot) * 0.5
        vertical_center_dist = player_y - gap_center
        actual_gap = next_pipe_bot - next_pipe_top

        return (
            self.fast_digitize(next_pipe_dist, self.x_bins),
            self.fast_digitize(vertical_center_dist, self.y_bins),
            self.fast_digitize(player_vel, self.v_bins),
            self.fast_digitize(actual_gap, self.gap_bins)
        )

    def act(self, state):
        """Predice Q-values y selecciona la acción óptima."""
        d0, d1, d2, d3 = self.discretize_state(state)

        # Normalización sin crear nuevos arrays
        self.state_buf[0] = d0
        self.state_buf[1] = d1
        self.state_buf[2] = d2
        self.state_buf[3] = d3

        norm = (self.state_buf - self.mean) / self.std

        # Inferencia directa sobre el modelo
        q_values = self.model(norm.reshape(1, 4), training=False)

        action_index = int(tf.argmax(q_values[0]))
        return self.actions[action_index]

    def get_q_values(self, state):
        """Devuelve los Q-values del modelo para un estado dado."""
        d0, d1, d2, d3 = self.discretize_state(state)

        self.state_buf[:] = (d0, d1, d2, d3)
        norm = (self.state_buf - self.mean) / self.std

        q_values = self.model(norm.reshape(1, 4), training=False)
        return q_values.numpy()[0]
