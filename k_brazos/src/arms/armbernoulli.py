import numpy as np

from .armbinomial import ArmBinomial


class ArmBernoulli(ArmBinomial):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli.

        :param p: Probabilidad de tener éxito al realizar la acción.
        """
        super().__init__(n=1, p=p)

    def __str__(self):
        """
        Representación en cadena del brazo de Bernoulli.

        :return: Descripción detallada del brazo de Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos con probabilidades únicas.

        :param k: Número de brazos a generar.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        # Generar k-valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(0, 1)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBernoulli(p) for p in p_values]

        return arms


