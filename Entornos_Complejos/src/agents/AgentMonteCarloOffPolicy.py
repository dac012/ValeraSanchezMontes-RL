import numpy as np
import gymnasium as gym

class AgentMonteCarloOffPolicy:
    def __init__(self, env: gym.Env, epsilon: float = 0.1, decay: bool = False, discount_factor: float = 1.0):
        self.env = env
        self.nA = env.action_space.n
        
        # Hiperparámetros
        self.epsilon = epsilon
        self.decay = decay
        self.discount_factor = discount_factor
        
        # Tablas de aprendizaje
        self.Q = np.zeros([env.observation_space.n, self.nA])
        self.C = np.zeros([env.observation_space.n, self.nA]) # Suma de pesos W
        
        # Memoria del episodio
        self.episode = [] 
        
        # Variables para estadísticas 
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0
        self.list_stats_success = [] 

    def get_action(self, state):
        """
        Política de Comportamiento 'b' (Behavior Policy).
        Es una política epsilon-soft (epsilon-greedy sobre Q).
        """
        probs = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1.0 - self.epsilon)
        
        return np.random.choice(np.arange(self.nA), p=probs)

    def get_greedy_action(self, state):
        """
        Política Objetivo 'pi' (Target Policy).
        Es determinista greedy con respecto a Q.
        """
        return np.argmax(self.Q[state])

    def update(self, state, action, next_state, reward, terminated, truncated, info):
        self.step_count += 1
        self.episode.append((state, action, reward))
        
        done = terminated or truncated
        
        if done:
            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            self.episode_lengths.append(self.step_count)
            
            G = 0.0
            W = 1.0
            
            episode_return = sum([x[2] * (self.discount_factor ** i) for i, x in enumerate(self.episode)])
            self.stats += episode_return
            # Evitamos división por cero en la primera iteración si t=0
            self.list_stats.append(self.stats / (self.t + 1))
            
            # Recorremos el episodio hacia atrás
            for i in range(len(self.episode) - 1, -1, -1):
                s, a, r = self.episode[i]
                
                # G <- gamma * G + R_{t+1}
                G = self.discount_factor * G + r
                
                # C(St, At) <- C(St, At) + W
                self.C[s, a] += W
                
                # Q(St, At) <- Q(St, At) + (W / C(St, At)) * [G - Q(St, At)]
                self.Q[s, a] += (W / self.C[s, a]) * (G - self.Q[s, a])
                
                # pi(St) <- argmax Q(St, a) (Mejor acción actual)
                greedy_action = np.argmax(self.Q[s])
                
                # Si At != pi(St) entonces salir del bucle
                if a != greedy_action:
                    break
                
                # W <- W * (1 / b(At|St))
                # Calculamos la probabilidad b(At|St) usada al momento de generar la acción
                # Como b es epsilon-greedy:
                if a == greedy_action:
                    prob_behavior = 1.0 - self.epsilon + (self.epsilon / self.nA)
                else:
                    prob_behavior = self.epsilon / self.nA
                
                # Si por redondeo la prob es 0 (no debería con epsilon > 0), paramos para evitar error
                if prob_behavior == 0:
                    break
                    
                W = W * (1.0 / prob_behavior)
            
            # Decaimiento del epsilon
            if self.decay:
                self.epsilon = max(0.01, min(1.0, 1.0 - np.log10((self.t + 1) / 25)))
                
            self.t += 1
            self.episode = []
            self.step_count = 0

    def get_stats(self):
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success