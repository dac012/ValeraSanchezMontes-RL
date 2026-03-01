import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de acciones.
        :param p: Probabilidad de tener éxito al realizar una acción.
        """
        assert n > 0, "El número de acciones tiene que ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad tiene que ser un valor entre 0 y 1"

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """

        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10):
        """
        Genera k brazos con probabilidades únicas.

        :param k: Número de brazos a generar.
        :param n: Número de acciones.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El valor de n debe ser mayor que 0."

        # Generar k-valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(0, 1)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBinomial(n, p) for p in p_values]

        return arms


