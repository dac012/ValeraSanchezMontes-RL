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
    def generate_arms(cls, k: int, n_min: int = 1, n_max: int = 10.0):
        """
        Genera k brazos con número de acciones únicos en el rango [n_min, n_max].

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo del número de acciones.
        :param n_max: Valor máximo del número de acciones.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min < n_max, "El valor de n_min debe ser menor que n_max."
        assert n_max-n_min >= k, "El número posible de valores tiene que ser mayor o igual que el número de brazos que queremos generar"

        # Generar k-valores únicos de n
        n_values = set()
        while len(n_values) < k:
            n = np.random.randint(n_min, n_max+1) # incluye n_max
            n_values.add(n)

        n_values = list(n_values)

        # TODO
        # ¿Generar probabilidades aleatorias?
        sigma = 1.0

        arms = [ArmBinomial(n, p) for n in n_values]

        return arms


