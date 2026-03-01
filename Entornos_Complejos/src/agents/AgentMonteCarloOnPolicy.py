import numpy as np
import gymnasium as gym

class AgentMonteCarloOnPolicy:
    def __init__(self, env: gym.Env, epsilon: float = 0.4, decay: bool = False, 
                 discount_factor: float = 1.0, first_visit: bool = False):
        """
        Inicialización del agente de Monte Carlo On-Policy.
        """
        self.env = env
        self.nA = env.action_space.n
        
        # Hiperparámetros
        self.epsilon = epsilon
        self.decay = decay
        self.discount_factor = discount_factor
        self.first_visit = first_visit  # <--- Nuevo parámetro
        
        # Tablas de aprendizaje
        # Q(s, a) -> Valor estimado de tomar la acción 'a' en el estado 's'
        self.Q = np.zeros([env.observation_space.n, self.nA])
        # N(s, a) -> Número de veces que se ha visitado el par (s, a)
        self.n_visits = np.zeros([env.observation_space.n, self.nA])
        
        # Memoria del episodio
        self.episode = [] 
        
        # Variables para estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0 # Contador de episodios
        self.list_stats_success = [] # Para medir éxito en episodios terminados por truncamiento vs terminación normal

    def get_action(self, state):
        """
        Política epsilon-greedy (Soft-policy).
        Basado en Diapositiva 23: Política epsilon-soft.
        """
        # Probabilidad base para todas las acciones: epsilon / |A|
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        
        # Elegir la mejor acción (rompiendo empates si es necesario)
        best_action = np.argmax(self.Q[state])
        
        # Sumar (1 - epsilon) a la mejor acción
        pi_A[best_action] += (1.0 - self.epsilon)
        
        # Elegir acción basándonos en la distribución
        return np.random.choice(np.arange(self.nA), p=pi_A)

    def get_greedy_action(self, state):
        """
        Política totalmente codiciosa para evaluación final.
        """
        return np.argmax(self.Q[state])
    
    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Almacena la transición y, al terminar el episodio, ejecuta el algoritmo MC.
        """
        self.step_count += 1
        
        # Guardamos la tupla. Nota: en Gym el reward es R_{t+1} tras hacer A_t en S_t
        self.episode.append((state, action, reward))
        
        done = terminated or truncated
        
        if done:
            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            self.episode_lengths.append(self.step_count)
            
            G = 0.0
            
            # Recorremos el episodio hacia atrás para calcular G de forma eficiente
            # Usamos índices para poder verificar "si ocurrió antes" en el caso First-Visit
            for i in range(len(self.episode) - 1, -1, -1):
                s, a, r = self.episode[i]
                
                # Actualizamos el Retorno acumulado: G = R_{t+1} + gamma * G_{t+1}
                G = r + self.discount_factor * G
                
                # --- LÓGICA FIRST-VISIT vs EVERY-VISIT ---
                update_flag = True
                
                if self.first_visit:
                    # Comprobamos si el par (s, a) aparece en los pasos anteriores del episodio (0 hasta i-1)
                    # Diapositiva 25: "Unless the pair S_t, A_t appears in S_0... S_{t-1}"
                    previous_occurrences = self.episode[:i]
                    for prev_s, prev_a, _ in previous_occurrences:
                        if prev_s == s and prev_a == a:
                            update_flag = False
                            break
                
                if update_flag:
                    self.n_visits[s, a] += 1.0
                    # Alpha variable (promedio simple), como indica la fórmula V(s) = S(s)/N(s) de la Diapositiva 9
                    alpha = 1.0 / self.n_visits[s, a]
                    
                    # Actualización incremental (Diapositiva 9, abajo)
                    self.Q[s, a] += alpha * (G - self.Q[s, a])
            
            # Estadísticas
            self.stats += G # Suma de retornos (solo del G_0 del episodio)
            self.list_stats.append(self.stats / (self.t + 1)) # Promedio histórico
            
            # Decaimiento del epsilon
            if self.decay:
                # Un decaimiento simple para garantizar convergencia a GLIE (Greedy in the Limit)
                self.epsilon = max(0.01, min(1.0, 1.0 - np.log10((self.t + 1) / 25)))
                
            self.t += 1
            self.episode = [] # Limpiar memoria para el siguiente episodio
            self.step_count = 0

    def get_stats(self):
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success