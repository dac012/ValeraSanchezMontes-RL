"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Dict, Any

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from algorithms import Algorithm, EpsilonGreedy, UCB1, Softmax, EpsilonDecay, UCB2


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    elif isinstance(algo, EpsilonDecay):
        label += f"(epsilon_0={algo.epsilon_0}, lambda={algo.lambda_decay}, epsilon_min={algo.epsilon_min})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    # Configuración del tema visual (mismo estilo que rewards)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        
        y_values = optimal_selections[idx] 
        
        plt.plot(range(steps), y_values, label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Óptima', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    
    # Fijamos el eje Y entre 0 y 100 (con un pequeño margen) para mejor visualización
    plt.ylim(0, 105)

    plt.legend(title='Algoritmos', loc='lower right') # 'lower right' suele tapar menos en estas gráficas
    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))

    # Curvas por algoritmo
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    # TODO: Implementar otros parametros
    for i, item in enumerate(args, start=1):
        if isinstance(item, tuple) and len(item) == 2:
            curve, label = item
        else:
            curve, label = item, f"Extra {i}"

        curve = np.asarray(curve)
        if curve.shape[0] != steps:
            raise ValueError(f"La curva adicional '{label}' debe tener longitud {steps}, pero tiene {curve.shape[0]}.")

        plt.plot(range(steps), curve, linestyle="--", linewidth=2, label=label)

    plt.xlabel("Pasos de Tiempo", fontsize=14)
    plt.ylabel("Regret Acumulado", fontsize=14)
    plt.title("Regret Acumulado vs Pasos de Tiempo", fontsize=16)
    plt.legend(title="Algoritmos", loc="upper left")
    plt.tight_layout()
    plt.show()
    

def plot_arm_statistics(arm_stats: List[Dict[int, Dict[str, Any]]], 
                        algorithms: List[Any], 
                        experiment_label: str = "Resultados"):
    """
    Genera un panel de gráficas de barras comparando el rendimiento de cada brazo por algoritmo.

    :param arm_stats:  Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param experiment_label: Texto opcional para el título.
    """
    n_algos = len(algorithms)
    
    # Configuración dinámica del tamaño de la figura según el nº de algoritmos
    # Usamos subplots verticales para facilitar la lectura de las etiquetas del eje X
    fig, axes = plt.subplots(nrows=n_algos, ncols=1, figsize=(10, 5 * n_algos), constrained_layout=True)
    
    # Aseguramos que axes sea iterable incluso si hay solo 1 algoritmo
    if n_algos == 1:
        axes = [axes]

    for idx, (ax, stats) in enumerate(zip(axes, arm_stats)):
        # Preparar datos
        algo_obj = algorithms[idx]
        algo_name = get_algorithm_label(algo_obj) if hasattr(algo_obj, 'select_arm') else str(algo_obj)
        
        # Ordenamos por ID de brazo para que el eje X sea coherente
        brazos_ids = sorted(stats.keys())
        
        medias = [stats[arm]['avg_reward'] for arm in brazos_ids]
        conteos = [stats[arm]['times_selected'] for arm in brazos_ids]
        es_optimo = [stats[arm]['optimal_arm'] == 1 for arm in brazos_ids]

        # Configurar colores (verde para óptimo, rojo para el resto)
        colores = ['#2ecc71' if opt else '#e74c3c' for opt in es_optimo]

        # Dibujar barras
        barras = ax.bar(brazos_ids, medias, color=colores, alpha=0.85, edgecolor='black', linewidth=0.7)

        # Añadir etiquetas de texto encima de la barra
        max_y = max(medias) if medias else 1.0
        offset = max_y * 0.02
        
        for bar, n_count, is_opt in zip(barras, conteos, es_optimo):
            height = bar.get_height()
            
            # Nº de selecciones
            text_label = f"N={int(n_count)}"
            
            font_weight = 'bold' if is_opt else 'normal'
            
            ax.text(bar.get_x() + bar.get_width()/2, height + offset, 
                    text_label, ha='center', va='bottom', fontsize=9, fontweight=font_weight)

        ax.set_title(f"{algo_name}", fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel("Recompensa Media")
        ax.set_xticks(brazos_ids)
        ax.set_xticklabels([f"Brazo {i}" for i in brazos_ids])
        
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Línea base en 0
        ax.axhline(0, color='black', linewidth=0.8)

    # Título
    fig.suptitle(f"Análisis de Exploración vs Explotación: {experiment_label}", fontsize=16, y=1.02)
    plt.show()