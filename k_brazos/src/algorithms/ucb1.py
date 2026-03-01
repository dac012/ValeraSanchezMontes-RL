import numpy as np

from .algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (constante de confianza). Controla el peso de la incertidumbre.
        :raises ValueError: Si c no es positivo.
        """
        assert 0 <= c <= 1, "El parámetro c debe ser positivo."

        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona el brazo que maximiza el índice UCB.
        Fórmula: Q(a) + c * sqrt(ln(t) / N(a))

        :return: índice del brazo seleccionado.
        """

        # La fórmula UCB tiene N(a) en el denominador. Si N(a) es 0, obtenemos infinito.
        # Por tanto, si hay brazos que nunca se han probado, los seleccionamos primero.
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # 't' es la suma de todas las veces que hemos tirado cualquier brazo.
        t = np.sum(self.counts)
        
        # Término de explotación (Q(a))
        exploitation = self.values
        
        # Término de exploración (c * sqrt(ln(t) / N(a)))
        exploration = self.c * np.sqrt(np.log(t) / self.counts)
        
        # Valores finales
        ucb_values = exploitation + exploration

        # Seleccionamos el brazo con el valor máximo.
        # Usamos random.choice para desempatar si hay varios máximos iguales.
        max_value = np.max(ucb_values)
        candidates = np.flatnonzero(ucb_values == max_value)
        chosen_arm = np.random.choice(candidates)
        
        return chosen_arm




