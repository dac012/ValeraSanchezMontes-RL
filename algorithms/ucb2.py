"""
UCB2

Extensión de UCB1 que selecciona brazos por BLOQUES (épocas) en lugar de
recalcular el índice en cada tirada.

Idea principal:
Cuando se elige un brazo, se ejecuta varias veces consecutivas antes
de volver a comparar índices. El tamaño del bloque crece de forma
controlada según:

    τ(k) = ceil((1 + alpha)^k)  -> número de tiradas consecutivas a un mismo brazo

donde:
    - k_a es el número de épocas del brazo a
    - alpha ∈ (0,1) controla el balance exploración/explotación

Índice UCB2: --> que tan prometedor es el brazo 

    UCB2(a) = Q(a) +
              sqrt( (1+alpha) * ln(e * t / τ(k_a)) / (2 * τ(k_a)) )

Funcionamiento:
1. Se prueba cada brazo una vez.
2. Se calcula el índice UCB2 para cada brazo.
3. Se selecciona el brazo con mayor índice.
4. Se ejecuta durante:
       ceil( τ(k_a + 1) - τ(k_a) )
   tiradas consecutivas.
5. Se incrementa k_a.

Diferencia clave con UCB1:
- UCB1 decide brazo en cada tirada.
- UCB2 decide brazo por bloques crecientes.

Esto reduce cambios frecuentes de acción y mantiene regret O(log T).
"""
import numpy as np

from algorithms.algorithm import Algorithm

class UCB2(Algorithm):

    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para el balance entre exploración y explotación.
        :raises ValueError: Si alpha no está en [0, 1].
        """
        assert 0 <= alpha <= 1, "El parámetro alpha debe estar en [0, 1]."

        super().__init__(k)
        self.alpha = alpha 

        # Incializamos k_a: número de épocas de la acción a
        self.epochs = np.zeros(self.k, dtype=int)

        # Variable para guardar qué brazo está actualmente en ejecución dentro del bloque.
        self.current_arm: None
        # Variable para guardar cuántas veces más debemos repetir ese brazo antes de volver a calcular el índice UCB2.
        self.block_remaining: int = 0

    def _tau(self, epoch: int) -> int:
        """Calcula el tamaño del bloque 
        τ(k) = ceil((1+alpha)^k)"""
        return int(np.ceil((1.0 + self.alpha) ** epoch))
    
    def _ucb2_index(self, t: int) -> np.ndarray:
        """
        Calcula el índice UCB2 para todos los brazos:
            UCB2(a) = Q(a) + sqrt( (1+alpha) * ln( e*t / tau(k_a) ) / (2*tau(k_a)) )

        donde:
            Q(a)         = self.values[a]
            t            = total de tiradas
            k_a          = self.epochs[a]
            tau(k_a)     = ceil((1+alpha)^(k_a))
        """

        # τ(k_a) para cada brazo a
        tau_k = np.array([self._tau(k_a) for k_a in self.epochs], dtype=float)

        # Aseguramos t >= 1
        t = max(int(t), 1)

        # Argumento del logaritmo: (e * t) / τ(k_a)
        log_argument = (np.e * t) / tau_k

        # En teoría, cuando (e*t)/τ(k_a) < 1, el log es negativo y la raíz daría NaN.
        # Para evitarlo, forzamos log_argument >= 1  -> log >= 0
        log_argument = np.maximum(log_argument, 1.0)

        # Numerador dentro de la raíz: (1+alpha) * ln( (e*t)/τ(k_a) )
        numerator = (1.0 + self.alpha) * np.log(log_argument)

        # Denominador dentro de la raíz: 2 * τ(k_a)
        denominator = 2.0 * tau_k

        # Parte dentro de la raíz completa
        inside_sqrt = numerator / denominator

        # Evitar valores negativos por redondeos
        inside_sqrt = np.maximum(inside_sqrt, 0.0)

        # Bono de exploración
        exploration_bonus = np.sqrt(inside_sqrt)

        # Índice final = explotación (Q) + exploración (bonus)
        ucb2_values = self.values + exploration_bonus

        return ucb2_values
        
    
    def select_arm(self) -> int:
        """
        Selecciona brazo según UCB2 con ejecución por bloques (épocas).
        Si estamos dentro de un bloque, repetimos el mismo brazo.
        """

          # Si hay brazos sin probar, probamos cada uno primero (inicialización)
        for i in range(self.k):
            if self.counts[i] == 0:
                self.current_arm = i
                self.block_remaining = 0  # solo hacemos una tirada dentro de ese bloque para solo probar cada uno una vez al menos
                return i
            
        # Si quedan tiradas del bloque actual, seguimos con el mismo brazo
        if self.block_remaining > 0 and self.current_arm is not None:
            self.block_remaining -= 1
            return self.current_arm

        # Si se ha acabado ese bloque: entonces cogemos un nuevo brazos y calculamos cuantas veces lo vamos a ejecutar

        # Calculamos el número total de tiradas realizadas hasta ahora
        t = int(np.sum(self.counts))

        # Elegimos el brazo con mayor índice UCB2
        ucb2_values = self._ucb2_index(t)
        max_value = np.max(ucb2_values)
        candidates = np.flatnonzero(ucb2_values == max_value)
        chosen_arm = int(np.random.choice(candidates))

        # Definimos el tamaño del bloque: ceil(tau(k+1)-tau(k))
        k_a = self.epochs[chosen_arm]
        block_len = int(np.ceil(self._tau(k_a + 1) - self._tau(k_a)))
        block_len = max(block_len, 1)

        # Guardamos estado de bloque actual:
        # Cual es el brazo con el que estamos jugando y cuantas tiradas quedan según el tamaño de bloque
        self.current_arm = chosen_arm
        self.block_remaining = block_len - 1

        # Incrementamos la época del brazo (se completa al asignar el bloque)
        self.epochs[chosen_arm] += 1

        return chosen_arm




