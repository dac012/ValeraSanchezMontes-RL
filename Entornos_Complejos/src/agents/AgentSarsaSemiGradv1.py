import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DEL TILE CODER (Aproximación)
# ==========================================
class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):
        self.n_tilings = n_tilings
        self.n_bins = n_bins
        self.low = low
        self.high = high
        # Calculamos el ancho de los bins
        self.scale = (high - low) / (n_bins - 1)
        # Espacio total de características (d en la imagen)
        self.n_features = n_tilings * (n_bins ** len(low))
        # Desplazamientos para cada tiling para cubrir el espacio
        self.offsets = [(i / n_tilings) * (high - low) / n_bins for i in range(n_tilings)]

    def get_features(self, state):
        state = np.clip(state, self.low, self.high)
        active_indices = []
        for i, offset in enumerate(self.offsets):
            bins = ((state - self.low + offset) / self.scale).astype(int)
            # Indexación para convertir coordenadas multidimensionales en un índice plano
            idx = i * (self.n_bins ** len(state)) + np.sum(bins * (self.n_bins ** np.arange(len(state))))
            active_indices.append(int(idx % self.n_features))
        return active_indices

# ==========================================
# 2. CLASE DEL AGENTE (Estructura Sarsa Semi-gradiente)
# ==========================================
class AgentSemiGradientSARSAv1:
    def __init__(self, env: gym.Env,
                 n_tilings: int = 8,
                 n_bins: int = 10,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 decay_c: float = 500.0,
                 discount_factor: float = 0.99,
                 alpha: float = 0.2):

        self.env = env
        self.nA = env.action_space.n
        
        # Inicializamos el TileCoder con los límites del entorno
        self.tc = TileCoder(n_tilings, n_bins, env.observation_space.low, env.observation_space.high)
        
        # Initialize value-function weights w arbitrarily (w = 0)
        # Matriz de pesos: [Acciones x Características]
        self.w = np.zeros([self.nA, self.tc.n_features], dtype=np.float64)

        # Parámetros
        self.epsilon = float(epsilon)          
        self.decay = bool(decay)               
        self.decay_c = float(decay_c)         
        self.discount_factor = float(discount_factor)
        # Dividimos alpha por el número de tilings para evitar divergencia
        self.alpha = float(alpha) / n_tilings 

        # Variables de estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0  
        self.list_stats_success = []
        self.episode_return = 0.0 

    def _get_q_value(self, f, action):
        """ q_hat(S, A, w) = suma de pesos de las características activas """
        return np.sum(self.w[action, f])

    def get_action(self, state):
        """ Política epsilon-greedy para entrenamiento """
        f = self.tc.get_features(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Update siguiendo el pseudocódigo de la imagen:
        w <- w + alpha [R + gamma*q(S',A',w) - q(S,A,w)] * grad(q)
        """
        self.step_count += 1
        self.episode_return += reward
        done = terminated or truncated

        f = self.tc.get_features(state)
        f_next = self.tc.get_features(next_state)
        
        # q(S, A, w)
        q_curr = self._get_q_value(f, action)

        if not done:
            # Choose A' as a function of q(S', ., w)
            next_action = self.get_action(next_state)
            # q(S', A', w)
            q_next = self._get_q_value(f_next, next_action)
            target = reward + self.discount_factor * q_next
        else:
            # S' is terminal
            next_action = None
            target = reward

        # Actualización de los pesos w (el gradiente es 1 para los índices f)
        self.w[action, f] += self.alpha * (target - q_curr)

        # Final de episodio y estadísticas
        if done:
            self.episode_lengths.append(self.step_count)
            success = 1 if (terminated and not truncated) else 0
            self.list_stats_success.append(success)
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        return next_action

    def get_stats(self):
        return self.w, self.list_stats, self.episode_lengths, self.list_stats_success

# ==========================================
# 3. EJECUCIÓN (SCRIPT PRINCIPAL)
# ==========================================
if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    
    # Parámetros recomendados para Acrobot:
    # n_tilings=8, n_bins=10, alpha=0.2, decay_c=400
    agente = AgentSemiGradientSARSAv1(env, alpha=0.2, decay_c=400)
    
    num_episodes = 5000
    print(f"Entrenando AgentSemiGradientSARSA en {env.spec.id}...")

    for ep in range(num_episodes):
        state, _ = env.reset()
        # S, A <- initial state and action
        action = agente.get_action(state)
        
        done = False
        while not done:
            # Take action A, observe R, S'
            next_state, reward, term, trunc, info = env.step(action)
            
            # w <- w + alpha[...] y elige A'
            action = agente.update(state, action, next_state, reward, term, trunc, info)
            
            state = next_state
            done = term or trunc
            
        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1:3d} | Recompensa Media: {agente.list_stats[-1]:.2f} | Epsilon: {agente.epsilon:.2f}")

    # Visualización de resultados
    _, media_stats, lengths, _ = agente.get_stats()
    
    plt.figure(figsize=(10, 5))
    plt.plot(media_stats, label='Media de Retorno Acumulada')
    plt.axhline(y=-100, color='r', linestyle='--', label='Umbral de éxito aprox.')
    plt.title("Progreso del Entrenamiento (Sarsa Semi-gradiente)")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)
    plt.show()