import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# TILE CODER (Aproximación de funciones para espacios continuos)
class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):

        # Guardamos hiperparámetros del Tile Coding
        # n_tilings = número de rejillas solapadas (tilings)
        # n_bins    = número de divisiones por dimensión (discretización)
        self.n_tilings = n_tilings
        self.n_bins = n_bins

        # low/high definen el rango continuo del estado (por dimensión)
        self.low = low
        self.high = high

        # Calculamos el ancho del "bin" (tamaño de cada casilla)
        # scale[d] = (high[d] - low[d]) / (n_bins - 1)
        # (esto permite convertir valores continuos a índices discretos)
        self.scale = (high - low) / (n_bins - 1)

        # Dimensión total del vector de características:
        # por cada tiling tenemos n_bins^dim posibles casillas
        # y como hay n_tilings, total = n_tilings * (n_bins^dim)
        self.n_features = n_tilings * (n_bins ** len(low))

        # Creamos offsets (desplazamientos) distintos para cada tiling
        # Objetivo: que cada rejilla esté "corrida" un poco,
        # y así el estado active diferentes casillas según el tiling.
        self.offsets = [(i / n_tilings) * (high - low) / n_bins for i in range(n_tilings)]

    def get_features(self, state):
        """
        Devuelve los índices de las características activas para un estado.
        En tile coding:
          - cada tiling activa 1 casilla (feature)
          - por tanto devuelve n_tilings índices activos
        """

        # Aseguramos que el estado no se salga de los límites del entorno
        state = np.clip(state, self.low, self.high)

        active_indices = []

        # Para cada tiling:
        for i, offset in enumerate(self.offsets):

            # Normalizamos y discretizamos:
            # bins[d] = floor((state[d] - low[d] + offset[d]) / scale[d])
            # Esto transforma el estado continuo a coordenadas discretas (casilla)
            bins = ((state - self.low + offset) / self.scale).astype(int)

            # Convertimos las coordenadas discretas (bins) en un índice único
            # Idea: "aplanamos" un índice multidimensional a un índice 1D
            # idx_local representa la casilla dentro del tiling i
            idx_local = np.sum(bins * (self.n_bins ** np.arange(len(state))))

            # Ahora sumamos el desplazamiento para distinguir el tiling
            # idx = (tiling i) * (n_bins^dim) + idx_local
            idx = i * (self.n_bins ** len(state)) + idx_local

            # Aseguramos que queda dentro del rango total
            active_indices.append(int(idx % self.n_features))

        # Devolvemos lista de índices activos (tamaño = n_tilings)
        return active_indices


# AGENTE SARSA SEMI-GRADIENTE 
class AgentSemiGradientSARSAv2:
    def __init__(self, env: gym.Env,
                 n_tilings: int = 8,
                 n_bins: int = 10,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 epsilon_decay: float = 0.992,
                 discount_factor: float = 0.99,
                 alpha: float = 0.2):

        # Guardamos el entorno y el número de acciones
        self.env = env
        self.nA = env.action_space.n

        # Inicializar Tile Coder para convertir estados continuos en features discretas
        self.tc = TileCoder(n_tilings, n_bins,
                            env.observation_space.low,
                            env.observation_space.high)

        # Inicializar pesos w
        # En vez de Q-table Q[s,a], tenemos aproximación lineal:
        #   q_hat(s,a) = suma(w[a, features_activas(s)])
        # w tiene forma: (acciones, n_features_totales)
        self.w = np.zeros([self.nA, self.tc.n_features], dtype=np.float64)

        # Declaramos parámetros del algoritmo
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.epsilon_decay = float(epsilon_decay)
        self.gamma = float(discount_factor)

        # En tile coding se suele dividir alpha entre n_tilings
        # porque en cada paso se actualizan n_tilings características
        self.alpha = float(alpha) / n_tilings

        # Variables para estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.list_stats_success = []
        self.step_count = 0
        self.t = 0
        self.episode_return = 0.0


    def _get_q_value(self, f, action):
        """
        Calcula q_hat(S, A, w):
          - f = índices de features activas (una por tiling)
          - q_hat = suma de pesos w[action] en esas posiciones
        """
        return np.sum(self.w[action, f])


    def get_action(self, state):
        """
        Política epsilon-greedy (entrenamiento):
          - con prob epsilon: exploramos (acción aleatoria)
          - si no: explotamos (acción con mayor q_hat)
        """
        f = self.tc.get_features(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        # Calculamos q_hat(s,a) para cada acción y elegimos la mejor
        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))


    def get_greedy_action(self, state):
        """Política greedy pura (evaluación)."""
        f = self.tc.get_features(state)
        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))


    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Implementa Semi-gradient SARSA (aproximación lineal):

        1) Observar transición (S,A,R,S')
        2) Elegir A' con epsilon-greedy (si no terminal)
        3) Target = R + gamma * q_hat(S',A')
        4) Error = Target - q_hat(S,A)
        5) Actualizar pesos de las features activas de (S,A):
             w[a, f] <- w[a, f] + alpha * Error
        """

        self.step_count += 1
        self.episode_return += reward

        done = terminated or truncated

        # Obtenemos features activas del estado actual y del siguiente
        f = self.tc.get_features(state)
        f_next = self.tc.get_features(next_state)

        # Calculamos el valor actual q_hat(S,A)
        q_curr = self._get_q_value(f, action)

        if not done:
            # Elegimos A' usando la política ε-greedy (como en SARSA)
            next_action = self.get_action(next_state)

            # Calculamos q_hat(S',A')
            q_next = self._get_q_value(f_next, next_action)

            # Target de SARSA: R + γ q_hat(S',A')
            target = reward + self.gamma * q_next
        else:
            # Si es terminal, no existe futuro
            next_action = None
            target = reward

        # Error temporal: (Target - q_hat(S,A))
        # si error > 0: la acción fue mejor de lo esperado -> subimos pesos
        # si error < 0: fue peor -> bajamos pesos
        td_error = target - q_curr

        # Actualización de pesos (semi-gradiente):
        # En aproximación lineal con tile coding:
        #   gradiente de q_hat respecto a w es 1 en features activas
        # Por eso solo actualizamos w[action, f]
        self.w[action, f] += self.alpha * td_error

        # Fin de episodio: estadísticas y decaimiento de epsilon
        if done:
            self.episode_lengths.append(self.step_count)

            # Éxito si terminó por objetivo (terminated) y no por límite de tiempo (truncated)
            self.list_stats_success.append(1 if terminated and not truncated else 0)

            # Media acumulada de recompensas (igual estilo que tu DQN)
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            # Decaimiento de epsilon
            if self.decay:
                self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            # Reseteamos variables del episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        # En SARSA devolvemos A' para continuar con (S <- S', A <- A')
        return next_action


    def get_stats(self):
        return self.w, self.list_stats, self.episode_lengths, self.list_stats_success


