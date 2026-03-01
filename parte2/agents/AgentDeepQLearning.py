import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# =====================================================================
# 1. RED NEURONAL (Código proporcionado por el profesor)
# =====================================================================
class DQN_Network(nn.Module):
    """
    Red neuronal para el algoritmo Deep Q-Network (DQN).
    Se compone de capas totalmente conectadas (FC) con activaciones ReLU.
    """
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()
        # Definición de las capas de la red neuronal
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),      # Capa de entrada con 12 neuronas
            nn.ReLU(inplace=True),         # Función de activación ReLU
            nn.Linear(12, 8),              # Capa oculta con 8 neuronas
            nn.ReLU(inplace=True),         # Otra activación ReLU
            nn.Linear(8, num_actions)      # Capa de salida con 'num_actions' neuronas
        ) 
        
        # Inicialización de pesos usando He initialization (Kaiming Uniform)
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        Q = self.FC(x) # Pasa el estado a través de la red neuronal
        return Q 

# =====================================================================
# 2. REPLAY BUFFER (Memoria de experiencia)
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# =====================================================================
# 3. AGENTE DEEP Q-LEARNING
# =====================================================================
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
        # En DQN los estados son vectores continuos, tomamos su dimensión
        self.nS = env.observation_space.shape[0] 

        # Parámetros de Q-learning / DQN
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.gamma = float(discount_factor)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        
        # Parámetros específicos de DQN
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Inicializar Red Principal y Red Objetivo usando la clase del profesor
        self.q_network = DQN_Network(num_actions=self.nA, input_dim=self.nS)
        self.target_network = DQN_Network(num_actions=self.nA, input_dim=self.nS)
        self.target_network.load_state_dict(self.q_network.state_dict()) # Copia inicial
        self.target_network.eval() # La red objetivo solo evalúa, no entrena

        # Optimizador y Buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.MSELoss()

        # Variables para estadísticas (basadas en tu código original)
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.list_stats_success = []
        self.step_count = 0
        self.global_step = 0
        self.t = 0
        self.episode_return = 0.0

    def get_action(self, state):
        """Política epsilon-greedy para explorar/explotar durante el entrenamiento."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        
        # Pasar el estado por la red neuronal
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_greedy_action(self, state):
        """Política greedy pura (solo para evaluación, sin epsilon)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Almacena la experiencia y ejecuta un paso de entrenamiento si hay datos suficientes.
        Este método es el que llama el bucle externo en cada paso.
        """
        done = terminated or truncated
        
        # 1. Guardar la transición en el Replay Buffer
        # (Usamos 'terminated' para saber si fue un final real del juego o solo límite de tiempo)
        self.memory.push(state, action, reward, next_state, terminated)
        
        self.step_count += 1
        self.global_step += 1
        self.episode_return += reward

        # 2. Entrenar la red interna (si el buffer ya tiene el tamaño mínimo del batch)
        if len(self.memory) >= self.batch_size:
            self._learn()

        # 3. Actualizar los pesos de la Red Objetivo periódicamente
        if self.global_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 4. Actualizaciones de fin de episodio (Estadísticas y Epsilon)
        if done:
            self.episode_lengths.append(self.step_count)
            self.list_stats_success.append(1 if terminated and not truncated else 0)
            
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Resetear para el siguiente episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

    def _learn(self):
        """
        Realiza el cálculo de la pérdida y el descenso de gradiente.
        Se ejecuta automáticamente desde update()
        """
        # Extraer un mini-batch aleatorio de la memoria
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convertir a tensores de PyTorch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculamos Q(S,A) actual con la red principal
        current_q_values = self.q_network(states).gather(1, actions)

        # Calculamos el Target usando la Red Objetivo
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            # La ecuación de Bellman: si 'done' es 1, el segundo término se anula (1-1 = 0)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Calculamos la pérdida (MSE) y optimizamos
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_stats(self):
        """Retorna los arrays de estadísticas para graficar."""
        # Nota: Retornamos la red entera en lugar de la matriz Q
        return self.q_network, self.list_stats, self.episode_lengths, self.list_stats_success