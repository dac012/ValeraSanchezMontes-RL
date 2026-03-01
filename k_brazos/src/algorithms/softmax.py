import numpy as np

from .algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo softmax.

        :param k: Número de brazos.
        :param tau: Controla la aleatoriedad de la selección. Cuanto mayor sea, más aleatoria será la selección (exploración), mientras que cuanto menor sea, más se favorecerán los brazos con mayor recompensa promedio (explotación).
        :raises ValueError: Si tau no es positivo.
        """
        assert tau > 0, "El parámetro tau debe ser positivo."

        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basándose en la función de Gibbs.
        pi(a) = exp(Q(a)/tau) / sum(exp(Q(b)/tau))

        :return: índice del brazo seleccionado.
        """

        # Obtenemos Q(a)
        q_values = self.values
        
        # Calculamos el numerador de la fórmula
        preferences = q_values / self.tau

        # Para evitar problemas numéricos con exp(), restamos el máximo valor de preferences a todos los preferences.
        preferences -= np.max(preferences)

        exp_values = np.exp(preferences)
        
        # Calculamos el denominador de la fórmula
        sum_exp_values = np.sum(exp_values)
        
        # Calculamos las probabilidades pi_t(a)
        probabilities = exp_values / sum_exp_values
        
        # Seleccionamos una acción At de acuerdo con la distribución pi_t(a)
        chosen_arm = np.random.choice(self.k, p=probabilities)
        
        return chosen_arm