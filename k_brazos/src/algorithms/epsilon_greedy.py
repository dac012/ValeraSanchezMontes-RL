"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from .algorithm import Algorithm

class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.

        :return: índice del brazo seleccionado.
        """

        # Si hay algún brazo que nunca se ha probado, lo probamos primero.
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        if np.random.random() < self.epsilon:
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




