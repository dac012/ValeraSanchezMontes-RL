import numpy as np 
import gymnasium as gym 

class AgentSARSA:
    def __init__(self, env: gym.Env,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 decay_c: float = 1000.0,
                 discount_factor: float = 0.99,
                 alpha: float = 0.1):

        #  Inicialización del entorno y espacio de acciones
        self.env = env
        self.nA = env.action_space.n
        
        #  Inicializamos la tabla Q con ceros para todos los estados y acciones
        self.Q = np.zeros([env.observation_space.n, self.nA], dtype=np.float64)

        #  Definición de parámetros de aprendizaje SARSA
        self.epsilon = float(epsilon) # Probabilidad de exploración inicial          
        self.decay = bool(decay) # Indica si epsilon debe reducirse con el tiempo               
        self.decay_c = float(decay_c) # Constante para controlar la velocidad del decaimiento         
        self.discount_factor = float(discount_factor) # Factor de descuento gamma
        self.alpha = float(alpha) # Tasa de aprendizaje

        #  Variables para estadísticas de rendimiento
        self.stats = 0.0 # Acumulador de retornos para la media
        self.list_stats = [] # Historial de recompensas promedio
        self.episode_lengths = [] # Pasos realizados en cada episodio
        self.step_count = 0 # Contador de pasos del episodio actual
        self.t = 0 # Contador de episodios totales realizados  
        self.list_stats_success = [] # Registro de éxitos por episodio

        #  Variable para guardar la recompensa total del episodio actual
        self.episode_return = 0.0 

    def get_action(self, state):
        #  Política epsilon-greedy para entrenamiento
        #  Elegimos una acción aleatoria con probabilidad epsilon
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        
        #  En caso contrario, elegimos la mejor acción según la tabla Q
        return int(np.argmax(self.Q[state]))

    def get_greedy_action(self, state):
        #  Política puramente greedy para evaluación final
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        #  Incrementamos los pasos y sumamos la recompensa al retorno del episodio
        self.step_count += 1
        self.episode_return += reward

        #  Comprobamos si el episodio ha llegado a su fin
        done = terminated or truncated

        #  Lógica de actualización SARSA (Estado-Acción-Recompensa-Estado-Acción)
        if not done:
            #  Elegimos la SIGUIENTE acción A' usando la política epsilon-greedy
            next_action = self.get_action(next_state)

            #  Calculamos el objetivo: R + gamma * Q(S', A')
            #  Usamos la acción que realmente vamos a tomar en el siguiente paso
            estimation_reward = reward + self.discount_factor * self.Q[next_state, next_action]
        else:
            #  Si el estado es terminal, no hay valor futuro
            next_action = None
            estimation_reward = reward

        #  Calculamos el error TD: [objetivo - valor_actual]
        difference_error = estimation_reward - self.Q[state, action]

        #  Actualizamos Q 
        self.Q[state, action] += self.alpha * difference_error

        #  Fin de episodio y estadísticas
        if done:
            #  Guardamos la duración del episodio
            self.episode_lengths.append(self.step_count)

            #  Registramos éxito (1) o fracaso (0)
            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            #  Actualizamos la media histórica de retornos
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            #  Aplicamos decaimiento a epsilon si está activado
            if self.decay:
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            #  Reseteamos contadores para empezar el nuevo episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        #  Retornamos la acción elegida para ejecutarla en el siguiente paso del bucle
        return next_action

    def get_stats(self):
        #  Devuelve la tabla Q y las métricas recogidas durante el entrenamiento
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success