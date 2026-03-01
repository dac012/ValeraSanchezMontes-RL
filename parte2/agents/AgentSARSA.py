import numpy as np
import gymnasium as gym

class AgentSARSA:
    def __init__(self, env: gym.Env,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 discount_factor: float = 0.99,
                 alpha: float = 0.1,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995):

        # Inicializar Q(s,a)
        self.env = env
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA], dtype=np.float64)

        # Declaramos parámetros de sarsa
        self.epsilon = float(epsilon)          
        self.decay = bool(decay)               
        self.discount_factor = float(discount_factor)
        self.alpha = float(alpha)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        

        # Declaramos variables para estadísticas (mismas que en MonteCarloOnPolicy)
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0  
        self.list_stats_success = []

        # Guardamos información del ultimo episodio
        self.episode_return = 0.0 
       


    #  Elegimos las proxima acción A usando política derivada de Q (ε-greedy)
    def get_action(self, state):
        """Política epsilon-greedy (entrenamiento)."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        return int(np.argmax(self.Q[state]))


    def get_greedy_action(self, state):
        """Política greedy pura (evaluación)."""
        return int(np.argmax(self.Q[state]))


    # Mientras el state no sea terminal o trucnated
    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Implementa la actualización SARSA:

        Q(S,A) ← Q(S,A) + α [ R + γ Q(S',A') - Q(S,A) ]
        """

        self.step_count += 1
        self.episode_return += reward

        done = terminated or truncated

        # Tomamos una acción A
        # Observar R y S'
        # Elegir A' usando la política ε-greedy (pseudocodigo del profesor, aunque tb se podria usar otra politica)

        if not done:
            # Cogemos A' 
            next_action = self.get_action(next_state)

            # Calculamos la estimación de cuánto vale realmente hacer la acción actual:
            #  R + γ Q(S', A') estimation_reward=recompensa actual + valor descontado del siguiente paso
            estimation_reward = reward + self.discount_factor * self.Q[next_state, next_action]
        else:
            # Si llegamos al final, no hace falta sumar lo que hay despues, por que no hay nada 
            next_action = None
            estimation_reward = reward


        # Calculamos: [ target − Q(S,A) ]
        # Calculamos la diferencia entre lo que ahora creemos que deberia valer la acción menos lo que actualmente creemos que vale
        # si difference_error > 0 la acción que estamos haciendo es mejor de lo que pensabamos --> subimos Q
        # si difference_error < 0 la acción que estamos haciendo es peor de lo que pensabamos --> bajamos Q
        difference_error = estimation_reward - self.Q[state, action]

        # Actualizamos Q con Q(S,A) ← Q(S,A) + α [ target − Q(S,A) ]
        self.Q[state, action] += self.alpha * difference_error


        # Actualizamos por fin de episodio 
        if done:
            self.episode_lengths.append(self.step_count)

            if terminated and not truncated:
                self.list_stats_success.append(1)
            else:
                self.list_stats_success.append(0)

            # Actualizamos la media acumulada del return
            self.stats += self.episode_return
            self.list_stats.append(self.stats / (self.t + 1))

            if self.decay:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon * self.epsilon_decay)

            # Reseteamos para el siguiente episodio 
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        # Actualizamos
        # S ← S'
        # A ← A'
        return next_action


    def get_stats(self):
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success