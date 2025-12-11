from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import time
from agentes.dq_agent import QAgent
import os
import csv

# Inicialización del juego y del entorno
game = FlappyBird()
env = PLE(game, display_screen=False, fps=30)
env.init()
actions = env.getActionSet()

# Eliminación de Q-table previa para comenzar desde cero
if os.path.exists("flappy_birds_q_table.pkl"):
    os.remove("flappy_birds_q_table.pkl")
    print("Archivo flappy_birds_q_table.pkl eliminado")

# Configuración del agente Q-Learning
agent = QAgent(
    actions,
    game,
    epsilon=1.0,
    min_epsilon=0.05,
    epsilon_decay=0.999762,
    learning_rate=0.2,
    discount_factor=0.95
)

num_episodes = 15000
max_steps_per_episode = 20000
rewards_all_episodes = []
episode_records = []

print(f"Acciones disponibles: {actions}")
print(f"Game Height: {game.height}, Game Width: {game.width}")

# Ciclo de entrenamiento
for episode in range(num_episodes):
    env.reset_game()
    state = env.getGameState()
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = agent.act(state)
        reward = env.act(action)
        next_state = env.getGameState()
        done = env.game_over()

        # Actualiza la Q-table
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_all_episodes.append(total_reward)
    agent.decay_epsilon()

    episode_records.append({
        "episode": episode + 1,
        "reward": total_reward,
        "epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table)
    })

    # Registro periódico del progreso
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_all_episodes[-100:])
        print(f"Episodio: {episode+1}/{num_episodes}, Recompensa Promedio: {avg_reward:.2f}")
        print("Estados visitados:", len(agent.q_table))
        agent.save_q_table("flappy_birds_q_table.pkl")

# Guardado final de la Q-table
agent.save_q_table("flappy_birds_q_table_final.pkl")

# Exportación del log de entrenamiento
with open("flappy_birds_training_log.csv", mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["episode", "reward", "epsilon", "q_table_size"])
    writer.writeheader()
    writer.writerows(episode_records)

print("Entrenamiento finalizado.")
