import numpy as np
import gymnasium as gym


class AgentQLearning:
    def __init__(self, env: gym.Env,
                 epsilon: float = 1.0,
                 decay: bool = True,
                 decay_c: float = 1000.0,         
                 discount_factor: float = 0.99,
                 alpha: float = 0.1):

        # Inicializar Q(s,a)
        self.env = env
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA], dtype=np.float64)

        # Declaramos parámetros de Q-learning
        self.epsilon = float(epsilon)
        self.decay = bool(decay)
        self.decay_c = float(decay_c)           
        self.discount_factor = float(discount_factor)
        self.alpha = float(alpha)

        # Declaramos variables para estadísticas
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
        self.step_count = 0
        self.t = 0                              
        self.list_stats_success = []

        # Guardamos información del ultimo episodio
        self.episode_return = 0.0


    # Elegimos la próxima acción A usando política derivada de Q (ε-greedy)
    # (Q-learning es off-policy, pero normalmente actúa con ε-greedy para explorar)
    def get_action(self, state):
        """Política epsilon-greedy (entrenamiento)."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        return int(np.argmax(self.Q[state]))


    def get_greedy_action(self, state):
        """Política greedy pura (evaluación)."""
        return int(np.argmax(self.Q[state]))


    # Mientras el state no sea terminal o truncated
    def update(self, state, action, next_state, reward, terminated, truncated, info):
        """
        Implementa la actualización Q-Learning:

        Q(S,A) ← Q(S,A) + α [ R + γ max_a Q(S',a) - Q(S,A) ]

        """

        self.step_count += 1
        self.episode_return += reward

        done = terminated or truncated

        # Tomamos una acción A
        # Observar R y S'
        # En Q-learning NO necesitamos elegir A' para el target:
        # en sarsa haciamos: next_action = self.get_action(next_state)
        # en vez de Q(S',A') usamos max_a Q(S',a) --> OFF POLICY

        if not done:
            # Calculamos la estimación de cuánto vale realmente hacer la acción actual:
            # R + γ max_a Q(S', a)
            # (estimation_reward = recompensa actual + valor descontado del mejor siguiente paso)
            best_next_value = np.max(self.Q[next_state])
            estimation_reward = reward + self.discount_factor * best_next_value
        else:
            # Si llegamos al final, no hace falta sumar lo que hay despues, por que no hay nada
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
                self.epsilon = min(1.0, self.decay_c / (self.t + 1))

            # Reseteamos para el siguiente episodio
            self.t += 1
            self.step_count = 0
            self.episode_return = 0.0

        # En Q-learning no hace falta devolver A' porque no se usa en la actualización
        return None


    def get_stats(self):
        return self.Q, self.list_stats, self.episode_lengths, self.list_stats_success