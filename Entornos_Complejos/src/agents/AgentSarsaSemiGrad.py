import numpy as np
import gymnasium as gym

# Clase para el tile coder 
class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):

        # Guardamos hiperparámetros del Tile Coding
        # n_tilings = número de rejillas solapadas
        # n_bins    = número de divisiones por dimensión
        self.n_tilings = n_tilings
        self.n_bins = n_bins

        # Guardamos límites continuos del estado
        self.low = low
        self.high = high

        # Calculamos el tamaño de cada bin (ancho de casilla) por dimensión
        # Nos servirá para convertir un valor continuo a un índice discreto
        self.scale = (high - low) / (n_bins - 1)

        # Total de características (d en la imagen/pseudocódigo):
        # por cada tiling hay n_bins^dim casillas
        # y como hay n_tilings -> total = n_tilings * (n_bins^dim)
        self.n_features = n_tilings * (n_bins ** len(low))

        # Creamos offsets (desplazamientos) distintos para cada tiling
        # Objetivo: que cada rejilla esté "corrida" un poco para mejorar generalización
        self.offsets = [(i / n_tilings) * (high - low) / n_bins for i in range(n_tilings)]

    def get_features(self, state):
        """
        Convertimos un estado continuo en índices de features activas.

        En tile coding:
          - cada tiling activa exactamente 1 casilla
          - por tanto devolvemos una lista con n_tilings índices
        """

        # Recortamos el estado para que esté dentro de los límites
        state = np.clip(state, self.low, self.high)

        active_indices = []

        # Repetimos para cada tiling (rejilla desplazada)
        for i, offset in enumerate(self.offsets):

            # Convertimos estado continuo a "bins" discretos:
            # bins[d] = floor((state[d] - low[d] + offset[d]) / scale[d])
            bins = ((state - self.low + offset) / self.scale).astype(int)

            # Convertimos coordenadas multidimensionales (bins) en un índice 1D
            # idx_local representa la casilla dentro de este tiling
            idx_local = np.sum(bins * (self.n_bins ** np.arange(len(state))))

            # Sumamos el offset del tiling para que no se mezclen entre tilings
            idx = i * (self.n_bins ** len(state)) + idx_local

            # Guardamos índice final activo
            active_indices.append(int(idx % self.n_features))

        return active_indices


# Clase para el agente
class AgentSemiGradientSARSA:
    def __init__(self, env: gym.Env,
                 n_tilings: int = 8,
                 n_bins: int = 10,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 decay_c: float = 500.0,
                 discount_factor: float = 0.99,
                 alpha: float = 0.2):

        # Guardamos entorno y número de acciones
        self.env = env
        self.nA = env.action_space.n

        # Inicializamos TileCoder con los límites del entorno
        # (así sabemos cómo discretizar estados continuos)
        self.tc = TileCoder(n_tilings, n_bins,
                            env.observation_space.low,
                            env.observation_space.high)

        # Inicializar pesos w arbitrariamente (aquí todo a 0)
        # En vez de Q-table, usamos aproximación lineal:
        #   q_hat(s,a,w) = suma(w[a, features_activas(s)])
        # Matriz de pesos: [acciones x características]
        self.w = np.zeros([self.nA, self.tc.n_features], dtype=np.float64)

        # Parámetros SARSA
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.decay_c = float(decay_c)
        self.discount_factor = float(discount_factor)

        # Dividimos alpha por el número de tilings para evitar divergencia
        # (porque en cada update modificamos n_tilings pesos a la vez)
        self.alpha = float(alpha) / n_tilings

        # Variables de estadísticas (igual estilo que tus otros agentes)
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0
        self.list_stats_success = []
        self.episode_return = 0.0


    def _get_q_value(self, f, action):
        """
        Calcula q_hat(S, A, w)
        = suma de pesos de las características activas f para esa acción
        """
        return np.sum(self.w[action, f])


    def get_action(self, state):
        """
        Elegimos la próxima acción con epsilon-greedy:
          - con prob epsilon -> acción aleatoria (explorar)
          - si no -> acción con mayor q_hat(s,a) (explotar)
        """
        f = self.tc.get_features(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))


    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Update siguiendo SARSA semi-gradiente:

        w <- w + alpha * [ R + gamma*q_hat(S',A',w) - q_hat(S,A,w) ] * grad(q_hat)

        En aproximación lineal con tile coding:
          - grad(q_hat) = 1 en las features activas (y 0 en el resto)
          - por eso actualizamos SOLO w[action, f]
        """

        self.step_count += 1
        self.episode_return += reward

        done = terminated or truncated

        # Obtenemos features activas del estado actual y del siguiente
        f = self.tc.get_features(state)
        f_next = self.tc.get_features(next_state)

        # Calculamos q_hat(S,A,w)
        q_curr = self._get_q_value(f, action)

        if not done:
            # Elegir A' usando política derivada de q_hat (epsilon-greedy)
            next_action = self.get_action(next_state)

            # Calculamos q_hat(S',A',w)
            q_next = self._get_q_value(f_next, next_action)

            # Target SARSA: R + gamma*q_hat(S',A')
            target = reward + self.discount_factor * q_next
        else:
            # Si S' es terminal, no hay futuro que sumar
            next_action = None
            target = reward

        # Error TD: [Target - q_hat(S,A)]
        # si > 0: fue mejor de lo esperado -> subimos pesos
        # si < 0: fue peor -> bajamos pesos
        td_error = target - q_curr

        # Actualización de pesos:
        # gradiente = 1 para las features activas -> sumamos alpha*td_error en esos índices
        self.w[action, f] += self.alpha * td_error


        if done:
            self.episode_lengths.append(self.step_count)

            # Éxito si terminó por objetivo (terminated) y no por tiempo (truncated)
            success = 1 if (terminated and not truncated) else 0
            self.list_stats_success.append(success)

            # Media acumulada del retorno
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            # Decaimiento tipo: epsilon = min(1.0, decay_c/(t+1))
            if self.decay:
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            # Reset para siguiente episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        # En SARSA devolvemos A' para continuar:
        # S <- S'
        # A <- A'
        return next_action


    def get_stats(self):
        return self.w, self.list_stats, self.episode_lengths, self.list_stats_success


