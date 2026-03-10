import numpy as np
import gymnasium as gym 

class AgentQLearning:
    def __init__(self, env: gym.Env,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 decay_c: float = 1000.0,         
                 discount_factor: float = 0.99,
                 alpha: float = 0.1):

        #  Inicialización del entorno y espacio de acciones
        self.env = env
        self.nA = env.action_space.n
        
        #  Inicializamos la tabla Q con ceros
        self.Q = np.zeros([env.observation_space.n, self.nA], dtype=np.float64)

        #  Definición de parámetros de aprendizaje Q-Learning
        self.epsilon = float(epsilon) # Probabilidad de exploración inicial
        self.decay = bool(decay) # Indica si epsilon debe reducirse
        self.decay_c = float(decay_c) # Constante para el ritmo de decaimiento           
        self.discount_factor = float(discount_factor) # Factor de descuento gamma
        self.alpha = float(alpha) # Tasa de aprendizaje

        #  Variables para estadísticas de rendimiento
        self.stats = 0.0 # Acumulador de retornos para la media
        self.list_stats = [] # Historial de recompensas promedio
        self.episode_lengths = [] # Pasos realizados por episodio
        self.step_count = 0 # Contador de pasos del episodio actual
        self.t = 0 # Contador de episodios totales realizados                              
        self.list_stats_success = [] # Registro de éxitos por episodio

        #  Variable para guardar la recompensa total del episodio actual
        self.episode_return = 0.0

    def get_action(self, state):
        #  Política epsilon-greedy para entrenamiento
        #  Elegimos una acción al azar para explorar con probabilidad epsilon
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        
        #  Si no exploramos, elegimos la mejor acción según la tabla Q
        return int(np.argmax(self.Q[state]))

    def get_greedy_action(self, state):
        #  Política puramente greedy para evaluación final
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        #  Incrementamos los pasos y sumamos la recompensa al retorno del episodio
        self.step_count += 1
        self.episode_return += reward

        #  Comprobamos si el episodio ha terminado
        done = terminated or truncated

        #  Lógica de actualización Q-Learning (Off-Policy)
        if not done:
            #  A diferencia de SARSA, aquí no usamos get_action para el siguiente paso
            #  Simplemente buscamos el valor de la mejor acción posible en el siguiente estado
            best_next_value = np.max(self.Q[next_state])
            
            #  Calculamos el objetivo: R + gamma * max_a Q(S', a)
            estimation_reward = reward + self.discount_factor * best_next_value
        else:
            #  Si el estado es terminal, el valor futuro es cero
            estimation_reward = reward

        #  Calculamos el error TD: [objetivo - valor_actual]
        difference_error = estimation_reward - self.Q[state, action]

        #  Actualizamos Q 
        self.Q[state, action] += self.alpha * difference_error

        #  Fin de episodio y estadísticas
        if done:
            #  Guardamos la duración del episodio
            self.episode_lengths.append(self.step_count)

            #  Registramos éxito (1) si terminó por objetivo, (0) si no
            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            #  Actualizamos la media acumulada de los retornos
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            #  Aplicamos decaimiento a epsilon si está configurado
            if self.decay:
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            #  Reseteamos variables para el próximo episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        #  En Q-learning no devolvemos la siguiente acción 
        return None

    def get_stats(self):
        #  Devuelve la tabla aprendida y las métricas recogidas
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success