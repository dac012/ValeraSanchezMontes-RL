import numpy as np

from .algorithm import Algorithm

class EpsilonDecay(Algorithm):

    def __init__(self, k: int, epsilon_0: float = 1.0, lambda_decay: float = 0.01, epsilon_min: float = 0.0):
        """
        Inicializa el algoritmo epsilon-decay.

        :param k: Número de brazos.
        :param epsilon_0: Valor inicial de epsilon.
        :param lambda_decay: Tasa de decaimiento de epsilon. Controla lo rápido que baja epsilon.
        :param epsilon_min: Valor mínimo de epsilon.
        :raises ValueError: Si epsilon_0 no está en [0, 1].
        :raises ValueError: Si epsilon_min no está en [0, 1].
        :raises ValueError: Si lambda_decay no es positivo.
        """
        assert 0 <= epsilon_0 <= 1, "El parámetro epsilon_0 debe estar entre 0 y 1."
        assert lambda_decay >= 0, "El parámetro lambda_decay debe ser positivo."
        assert 0 <= epsilon_min <= 1, "El parámetro epsilon_min debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon_0 = epsilon_0
        self.lambda_decay = lambda_decay
        self.epsilon_min = epsilon_min

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-decay.

        :return: índice del brazo seleccionado.
        """

        # Si hay algún brazo que nunca se ha probado, lo probamos primero.
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # t es la suma de todas las veces que hemos accionado algún brazo.
        t = np.sum(self.counts)

        # Calcular epsilon_t con decaimiento inversamente proporcional
        current_epsilon = self.epsilon_0 / (1.0 + self.lambda_decay * t)

        # Limitamos el valor mínimo usando max()
        epsilon_t = max(self.epsilon_min, current_epsilon)

        if np.random.random() < epsilon_t:
            # Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Introducimos desempate aleatorio entre brazos con valor máximo
            # para asegurar un comportamiento consistente con la definición
            # teórica del algoritmo greedy. Sin este ajuste, np.argmax introduce
            # un sesgo determinista hacia los primeros brazos.
            max_value = np.max(self.values)
            candidates = np.flatnonzero(self.values == max_value)
            chosen_arm = np.random.choice(candidates)


        return chosen_arm