import numpy as np 
import gymnasium as gym 

class AgentMonteCarloOnPolicy:
    def __init__(self, env: gym.Env, epsilon: float = 0.4, decay: bool = False, 
                 discount_factor: float = 1.0, first_visit: bool = False):
        #  Inicialización del entorno y espacio de acciones
        self.env = env
        self.nA = env.action_space.n
        
        #  Definición de hiperparámetros del algoritmo
        self.epsilon = epsilon # Probabilidad de exploración
        self.decay = decay # Booleano para reducir exploración en el tiempo
        self.discount_factor = discount_factor # Factor de descuento gamma
        self.first_visit = first_visit # Variante de primera visita o toda visita
        
        #  Tablas de aprendizaje inicializadas a cero
        #  Q almacena el valor estimado de cada par estado-acción
        self.Q = np.zeros([env.observation_space.n, self.nA])
        #  n_visits cuenta las veces que se ha visitado cada par (s, a)
        self.n_visits = np.zeros([env.observation_space.n, self.nA])
        
        #  Lista para almacenar la trayectoria del episodio actual
        self.episode = [] 
        
        #  Variables para estadísticas de rendimiento
        self.stats = 0.0 # Sumatorio de retornos para la media
        self.list_stats = [] # Historial de promedios de recompensa
        self.episode_lengths = [] # Pasos por episodio
        self.step_count = 0 # Contador de pasos del episodio en curso
        self.t = 0 # Contador global de episodios
        self.list_stats_success = [] # Registro de objetivos alcanzados

    def get_action(self, state):
        #  Implementación de política epsilon-greedy
        #  Asignamos probabilidad mínima a todas las acciones para explorar
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        
        #  Buscamos la acción con mayor valor en la tabla Q
        best_action = np.argmax(self.Q[state])
        
        #  Asignamos la probabilidad máxima a la mejor acción encontrada
        pi_A[best_action] += (1.0 - self.epsilon)
        
        #  Seleccionamos acción basándonos en la distribución de probabilidad pi_A
        return np.random.choice(np.arange(self.nA), p=pi_A)

    def get_greedy_action(self, state):
        #  Política puramente greedy para evaluar el agente sin explorar
        return np.argmax(self.Q[state])
    
    def update(self, state, action, next_state, reward, terminated, truncated, info):
        #  Incrementamos el contador de pasos del episodio actual
        self.step_count += 1
        
        #  Guardamos la transición en la memoria para procesarla al final
        self.episode.append((state, action, reward))
        
        #  Determinamos si el episodio ha finalizado por cualquier causa
        done = terminated or truncated
        
        if done:
            #  Procesamiento del éxito del episodio para estadísticas
            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            #  Almacenamos la longitud final del episodio
            self.episode_lengths.append(self.step_count)
            
            #  Inicializamos el retorno acumulado G
            G = 0.0
            
            #  Bucle hacia atrás para calcular retornos y actualizar Q
            for i in range(len(self.episode) - 1, -1, -1):
                s, a, r = self.episode[i] # Obtenemos estado, acción y recompensa
                
                #  Actualización del retorno: G = R + gamma * G
                G = r + self.discount_factor * G
                
                #  Lógica para decidir si se actualiza según First-Visit o All-Visit
                update_flag = True
                
                if self.first_visit:
                    #  Comprobamos si el par (s, a) ocurrió antes en el episodio
                    previous_occurrences = self.episode[:i]
                    for prev_s, prev_a, _ in previous_occurrences:
                        if prev_s == s and prev_a == a:
                            update_flag = False # Si ya ocurrió, no actualizamos
                            break
                
                if update_flag:
                    #  Anotamos la visita y calculamos el tamaño del paso alpha
                    self.n_visits[s, a] += 1.0
                    alpha = 1.0 / self.n_visits[s, a] # Promedio incremental
                    
                    #  Actualizamos el valor Q con el error del retorno (G - Q)
                    self.Q[s, a] += alpha * (G - self.Q[s, a])
            
            #  Actualización de estadísticas de recompensa promedio
            self.stats += G 
            self.list_stats.append(self.stats / (self.t + 1))
            
            #  Lógica de decaimiento logarítmico del factor de exploración
            if self.decay:
                self.epsilon = max(0.01, min(1.0, 1.0 - np.log10((self.t + 1) / 25)))
                
            #  Preparamos variables para el inicio del siguiente episodio
            self.t += 1
            self.episode = [] 
            self.step_count = 0

    def get_stats(self):
        #  Devuelve la tabla Q y las listas de métricas recogidas
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success