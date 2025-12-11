from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import argparse
import importlib
import sys
import inspect

# Inicializa el juego y el entorno (PLE)
game = FlappyBird()
env = PLE(game, display_screen=True, fps=30)
env.init()

actions = env.getActionSet()

# Parser simple para indicar qué agente cargar dinámicamente
parser = argparse.ArgumentParser(description="Test de agentes para FlappyBird (PLE)")
parser.add_argument('--agent', type=str, required=True,
                    help='Ruta completa del agente, ej: agentes.random_agent.RandomAgent')
args = parser.parse_args()

# Carga dinámica del módulo y la clase del agente
try:
    module_path, class_name = args.agent.rsplit('.', 1)
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
except Exception as e:
    print(f"No se pudo cargar el agente: {e}")
    sys.exit(1)

# Revisa la firma del __init__ para saber si el agente usa Q-table
agent_init_params = inspect.signature(AgentClass.__init__).parameters

if "load_q_table_path" in agent_init_params:
    print("Agente compatible con Q-table, cargando archivo")
    agent = AgentClass(
        actions,
        game,
        load_q_table_path="flappy_birds_q_table_final.pkl"
    )
else:
    print("Agente sin Q-table. Se inicializa normalmente.")
    agent = AgentClass(actions, game)

# Loop principal: ejecuta episodios indefinidamente
while True:
    env.reset_game()
    agent.reset()

    state = env.getGameState()
    done = False
    total_reward = 0

    print("\n--- Ejecutando agente ---")

    # Ciclo de interacción agente-entorno
    while not done:
        action = agent.act(state)
        reward = env.act(action)
        state = env.getGameState()
        done = env.game_over()
        total_reward += reward

        time.sleep(0.03)   # Control de velocidad para visualización

    print(f"Recompensa episodio: {total_reward}")
