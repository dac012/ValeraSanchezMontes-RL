import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# RED NEURONAL 
class DQN_Network(nn.Module):
    """
    Sustituye a la tabla Q. Recibe un estado y devuelve el valor Q
    estimado para cada acción posible: Q(s, a1), Q(s, a2)...
    """
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()

        # Definimos una arquitectura sencilla de 3 capas
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12), # Capa de entrada (dimensión del estado)
            nn.ReLU(inplace=True),    # Activación no lineal
            nn.Linear(12, 8),          # Capa oculta
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions) # Capa de salida (una neurona por acción)
        )

        # Inicialización técnica de pesos (Kaiming) para mejorar la convergencia
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        # Propagación hacia adelante: devuelve los valores Q
        return self.FC(x)


# REPLAY BUFFER 
class ReplayBuffer:
    """
    Guarda experiencias pasadas (S, A, R, S', done).
    Rompe la correlación temporal entrenando con muestras aleatorias.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # Si se llena, borra lo más antiguo

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Extrae un grupo aleatorio para el entrenamiento
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# AGENTE DQN 
class AgentDeepQLearning:
    def __init__(self, env: gym.Env,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 discount_factor: float = 0.99,
                 lr: float = 1e-3,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,
                 batch_size: int = 64,
                 buffer_capacity: int = 10000,
                 target_update_freq: int = 100):

        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.shape[0]

        # Parámetros de exploración y aprendizaje
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.gamma = float(discount_factor)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Inicialización de redes: q_network (la que se entrena) y target_network (el objetivo)
        self.q_network = DQN_Network(self.nA, self.nS)
        self.target_network = DQN_Network(self.nA, self.nS)
        
        # Sincronización inicial de parámetros y desactivación de gradientes en target
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.MSELoss() # Error Cuadrático Medio

        # Estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.list_stats_success = []
        self.step_count = 0
        self.global_step = 0
        self.t = 0
        self.episode_return = 0.0

    def get_action(self, state):
        # Epsilon-greedy con PyTorch
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        done = terminated or truncated
        
        # Guardar la experiencia en la memoria
        self.memory.push(state, action, reward, next_state, terminated)

        self.step_count += 1
        self.global_step += 1
        self.episode_return += reward

        #  Aprender si tenemos suficiente memoria
        if len(self.memory) >= self.batch_size:
            self._learn()

        # Actualizar la Red Objetivo (cada C pasos)
        if self.global_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Gestión de fin de episodio
        if done:
            self.episode_lengths.append(self.step_count)
            self.list_stats_success.append(1 if (terminated and not truncated) else 0)
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

    def _learn(self):
        # Muestrear el batch de la memoria
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convertir a Tensores de PyTorch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Cálculo de Q(s, a) actual (Predicción)
        current_q_values = self.q_network(states).gather(1, actions)

        # Cálculo del Target (Etiqueta real esperada)
        with torch.no_grad():
            # Usamos la target_network para calcular el valor máximo del siguiente estado
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            # Ecuación de Bellman: R + gamma * max Q(s', a')
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Optimización
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward() # Backpropagation
        self.optimizer.step() # Actualizar pesos

    def get_stats(self):
        return self.q_network, self.list_stats, self.episode_lengths, self.list_stats_success