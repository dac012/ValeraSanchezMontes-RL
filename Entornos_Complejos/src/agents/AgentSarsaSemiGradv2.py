import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# =====================================================================
# 1. TILE CODER (Aproximación de funciones para espacios continuos)
# =====================================================================
class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):
        self.n_tilings = n_tilings
        self.n_bins = n_bins
        self.low = low
        self.high = high
        # Calculamos el ancho de los bins
        self.scale = (high - low) / (n_bins - 1)
        # Espacio total de características (dimensión del vector de pesos w)
        self.n_features = n_tilings * (n_bins ** len(low))
        # Desplazamientos para cada tiling
        self.offsets = [(i / n_tilings) * (high - low) / n_bins for i in range(n_tilings)]

    def get_features(self, state):
        state = np.clip(state, self.low, self.high)
        active_indices = []
        for i, offset in enumerate(self.offsets):
            # Normalizamos y discretizamos el estado
            bins = ((state - self.low + offset) / self.scale).astype(int)
            # Convertimos las coordenadas de los bins en un índice único
            idx = i * (self.n_bins ** len(state)) + np.sum(bins * (self.n_bins ** np.arange(len(state))))
            active_indices.append(int(idx % self.n_features))
        return active_indices

# =====================================================================
# 2. AGENTE SARSA SEMI-GRADIENTE (Estructura compatible con DQN)
# =====================================================================
class AgentSemiGradientSARSAv2:
    def __init__(self, env: gym.Env,
                 n_tilings: int = 8,
                 n_bins: int = 10,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 epsilon_decay: float = 0.992, 
                 discount_factor: float = 0.99,
                 alpha: float = 0.2):

        self.env = env
        self.nA = env.action_space.n
        
        # Inicializar Tile Coder
        self.tc = TileCoder(n_tilings, n_bins, env.observation_space.low, env.observation_space.high)
        
        # Inicializar pesos w (Equivalente a los pesos de la red neuronal en DQN)
        self.w = np.zeros([self.nA, self.tc.n_features], dtype=np.float64)

        # Parámetros del algoritmo
        self.epsilon = float(epsilon)          
        self.decay = bool(decay)               
        self.epsilon_decay = float(epsilon_decay)         
        self.gamma = float(discount_factor)
        self.alpha = float(alpha) / n_tilings # Alpha normalizado por el número de tilings

        # Variables para estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.list_stats_success = []
        self.step_count = 0
        self.t = 0
        self.episode_return = 0.0

    def _get_q_value(self, f, action):
        """Calcula q_hat(S, A, w) como la suma de pesos de las características activas."""
        return np.sum(self.w[action, f])

    def get_action(self, state):
        """Política epsilon-greedy para entrenamiento."""
        f = self.tc.get_features(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))

    def get_greedy_action(self, state):
        """Política greedy pura para evaluación."""
        f = self.tc.get_features(state)
        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """Implementa la actualización de pesos Semi-gradiente SARSA."""
        self.step_count += 1
        self.episode_return += reward
        done = terminated or truncated

        # Obtener características activas
        f = self.tc.get_features(state)
        f_next = self.tc.get_features(next_state)
        
        # Valor Q actual
        q_curr = self._get_q_value(f, action)

        if not done:
            # Seleccionar siguiente acción A'
            next_action = self.get_action(next_state)
            q_next = self._get_q_value(f_next, next_action)
            # Target de SARSA
            target = reward + self.gamma * q_next
        else:
            next_action = None
            target = reward

        # Actualización de pesos: w = w + alpha * [Target - Q] * gradiente(Q)
        # En aproximación lineal con Tile Coding, el gradiente es 1 para los índices activos
        self.w[action, f] += self.alpha * (target - q_curr)

        # Gestión de fin de episodio
        if done:
            self.episode_lengths.append(self.step_count)
            # Éxito si termina por objetivo (terminated) y no por tiempo (truncated)
            self.list_stats_success.append(1 if terminated and not truncated else 0)
            
            # Media acumulada (Estilo DQN)
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        return next_action

    def get_stats(self):
        """Retorna pesos y listas de rendimiento."""
        return self.w, self.list_stats, self.episode_lengths, self.list_stats_success

# =====================================================================
# 3. BUCLE DE EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    # Configuración del entorno
    env = gym.make("Acrobot-v1")
    
    # Instanciar agente con los hiperparámetros optimizados
    agente = AgentSemiGradientSARSAv2(env, alpha=0.2, epsilon_decay=0.992)
    
    num_episodes = 5000 # Puedes subirlo a 1000 o 2000 para una curva más estable
    print(f"Entrenando agente en {env.spec.id}...")

    for ep in range(num_episodes):
        state, _ = env.reset()
        action = agente.get_action(state)
        
        done = False
        while not done:
            next_state, reward, term, trunc, info = env.step(action)
            
            # Actualización y selección de la siguiente acción
            action = agente.update(state, action, next_state, reward, term, trunc, info)
            
            state = next_state
            done = term or trunc
            
        if (ep + 1) % 50 == 0:
            print(f"Episodio {ep+1:4d} | Media Recompensa: {agente.list_stats[-1]:.2f} | Epsilon: {agente.epsilon:.3f}")

    # Visualización básica
    pesos, media_acumulada, duracion, exitos = agente.get_stats()
    
    plt.figure(figsize=(10, 5))
    plt.plot(media_acumulada)
    plt.title("Rendimiento: Media Acumulada de Recompensas")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Promedio")
    plt.grid(True)
    plt.show()