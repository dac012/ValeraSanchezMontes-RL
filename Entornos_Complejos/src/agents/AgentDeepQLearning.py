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
    Esta red aproxima Q(s,a).
    En lugar de tener una tabla como en SARSA,
    ahora usamos una red neuronal que recibe el estado
    y devuelve el valor estimado de TODAS las acciones.
    """

    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()

        # Creamos una red totalmente conectada:
        # estado -> 12 neuronas -> 8 neuronas -> num_actions
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )

        # Inicializamos pesos (mejor estabilidad con ReLU)
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        # Pasamos el estado por la red
        # Devuelve un vector: [Q(s,a1), Q(s,a2), ..., Q(s,aN)]
        Q = self.FC(x)
        return Q


# REPLAY BUFFER

class ReplayBuffer:
    """
    En DQN no entrenamos con la transición actual directamente.
    Guardamos experiencias en memoria y luego entrenamos con
    muestras aleatorias para evitar correlación temporal.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Guardamos transición (S, A, R, S', done)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Sacamos un minibatch aleatorio
    def sample(self, batch_size):
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

        # Información del entorno
        self.env = env
        self.nA = env.action_space.n
        self.nS = env.observation_space.shape[0]

        # Parámetros principales
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.gamma = float(discount_factor)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Parámetros específicos de DQN
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Creamos red principal (la que aprende)
        self.q_network = DQN_Network(self.nA, self.nS)

        # Creamos red objetivo (la que calcula el target estable)
        # Al principio ambas redes son iguales
        self.target_network = DQN_Network(self.nA, self.nS)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # No se entrena

        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Memoria de experiencia
        self.memory = ReplayBuffer(buffer_capacity)

        # Función de pérdida (error cuadrático medio)
        self.loss_fn = nn.MSELoss()

        # Variables de estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.list_stats_success = []
        self.step_count = 0
        self.global_step = 0
        self.t = 0
        self.episode_return = 0.0


    # Elegimos acción con política ε-greedy
    def get_action(self, state):
        """
        Si exploramos: acción aleatoria.
        Si explotamos: acción con mayor Q(s,a).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())


    # Política greedy pura (sin exploración)
    def get_greedy_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())


    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        En cada paso:

        1) Guardamos transición en memoria.
        2) Si hay suficientes muestras -> entrenamos.
        3) Cada cierto número de pasos -> actualizamos red objetivo.
        """

        done = terminated or truncated

        # Guardamos (S,A,R,S')
        self.memory.push(state, action, reward, next_state, terminated)

        self.step_count += 1
        self.global_step += 1
        self.episode_return += reward

        # Si ya tenemos suficientes experiencias, entrenamos
        if len(self.memory) >= self.batch_size:
            self._learn()

        # Actualizamos red objetivo cada C pasos
        if self.global_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Fin de episodio
        if done:

            self.episode_lengths.append(self.step_count)

            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            # Decaimiento exponencial de epsilon
            if self.decay:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon * self.epsilon_decay)

            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0


    def _learn(self):
        """
        Implementa la actualización DQN:

        1) Muestreamos minibatch.
        2) Calculamos Q(S,A) actual.
        3) Calculamos target:
           target = R + γ max_a' Q_target(S', a')
        4) Minimizamos el error cuadrático.
        """

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q(S,A) actual (lo que creemos ahora)
        current_q_values = self.q_network(states).gather(1, actions)

        # Calculamos el target usando la red objetivo
        with torch.no_grad():

            # max_a' Q_target(S', a')
            max_next_q_values = \
                self.target_network(next_states).max(1)[0].unsqueeze(1)

            # Si es estado terminal no sumamos nada futuro
            target_q_values = rewards + \
                              self.gamma * max_next_q_values * (1 - dones)

        # Diferencia entre lo que creemos y lo que debería valer
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_stats(self):
        return self.q_network, self.list_stats, \
               self.episode_lengths, self.list_stats_success