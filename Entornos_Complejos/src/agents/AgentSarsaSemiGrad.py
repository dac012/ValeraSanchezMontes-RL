import numpy as np
import gymnasium as gym 

# Clase para el tile coder 
class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):
        #  Configuración de la estructura de las rejillas (tilings)
        self.n_tilings = n_tilings # Cantidad de capas solapadas
        self.n_bins = n_bins # Divisiones por cada dimensión

        #  Límites físicos del entorno 
        self.low = low
        self.high = high

        #  Cálculo del ancho de cada división para mapear valores continuos
        self.scale = (high - low) / (n_bins - 1)

        #  Cálculo del número total de características (neuronas de entrada)
        self.n_features = n_tilings * (n_bins ** len(low))

        #  Definición de desplazamientos para que cada rejilla vea algo distinto
        self.offsets = [(i / n_tilings) * (high - low) / n_bins for i in range(n_tilings)]

    def get_features(self, state):
        #  Mantenemos el estado dentro de los rangos permitidos
        state = np.clip(state, self.low, self.high)
        active_indices = []

        #  Para cada capa de rejilla, calculamos qué casilla se activa
        for i, offset in enumerate(self.offsets):
            #  Convertimos la posición continua a un índice de celda discreto
            bins = ((state - self.low + offset) / self.scale).astype(int)

            #  Linealizamos las coordenadas de la celda a un índice único 1D
            idx_local = np.sum(bins * (self.n_bins ** np.arange(len(state))))

            #  Ajustamos el índice global sumando el bloque correspondiente al tiling
            idx = i * (self.n_bins ** len(state)) + idx_local

            #  Añadimos el índice de la característica "encendida"
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

        #  Inicialización del entorno y acciones disponibles
        self.env = env
        self.nA = env.action_space.n

        #  Configuramos el codificador de estados continuos
        self.tc = TileCoder(n_tilings, n_bins,
                            env.observation_space.low,
                            env.observation_space.high)

        #  Inicializamos los pesos w a cero 
        #  En lugar de Q[s,a], el valor será la suma de pesos de rasgos activos
        self.w = np.zeros([self.nA, self.tc.n_features], dtype=np.float64)

        #  Parámetros de la política y el aprendizaje
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.decay_c = float(decay_c)
        self.discount_factor = float(discount_factor)

        #  Normalizamos alpha dividiendo por el número de tilings activos
        #  Esto asegura que el aprendizaje sea estable y no de saltos enormes
        self.alpha = float(alpha) / n_tilings

        #  Variables para seguimiento de estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0
        self.list_stats_success = []
        self.episode_return = 0.0

    def _get_q_value(self, f, action):
        #  Aproximación lineal: sumamos los pesos de las casillas activas
        return np.sum(self.w[action, f])

    def get_action(self, state):
        #  Obtenemos qué características están activas para este estado
        f = self.tc.get_features(state)

        #  Lógica de exploración epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        #  Calculamos el valor Q para cada acción posible y elegimos la mejor
        qs = [self._get_q_value(f, a) for a in range(self.nA)]
        return int(np.argmax(qs))

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        #  Actualizamos contadores del episodio
        self.step_count += 1
        self.episode_return += reward

        #  Detectamos si el episodio ha llegado al final
        done = terminated or truncated

        #  Obtenemos los rasgos activos del estado actual y del siguiente
        f = self.tc.get_features(state)
        f_next = self.tc.get_features(next_state)

        #  Calculamos el valor Q estimado para la acción que acabamos de hacer
        q_curr = self._get_q_value(f, action)

        if not done:
            #  Elegimos la siguiente acción A' (On-Policy)
            next_action = self.get_action(next_state)

            #  Estimamos el valor del siguiente paso: Q(S', A')
            q_next = self._get_q_value(f_next, next_action)

            #  Calculamos el objetivo (target) de SARSA
            target = reward + self.discount_factor * q_next
        else:
            #  Si es terminal, el valor futuro es nulo
            next_action = None
            target = reward

        #  Calculamos el error entre nuestra predicción y la realidad observada
        td_error = target - q_curr

        #  Actualizamos solo los pesos de las características que estaban activas
        self.w[action, f] += self.alpha * td_error

        if done:
            #  Guardamos duración y éxito del episodio
            self.episode_lengths.append(self.step_count)
            success = 1 if (terminated and not truncated) else 0
            self.list_stats_success.append(success)

            #  Actualizamos estadísticas de recompensa media
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            #  Aplicamos el decaimiento de la exploración
            if self.decay:
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            #  Reiniciamos parámetros para el nuevo episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        #  Devolvemos la siguiente acción para mantener el flujo SARSA
        return next_action

    def get_stats(self):
        #  Devolvemos los pesos aprendidos y las métricas recogidas
        return self.w, self.list_stats, self.episode_lengths, self.list_stats_success