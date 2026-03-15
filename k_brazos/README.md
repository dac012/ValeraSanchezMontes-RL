# Práctica 1: Aprendizaje por Refuerzo - Bandido de K-Brazos

## Información

**Authors:**  
David Valera López  
Lucía Sánchez Montes Gómez  

**Emails:**  
d.valeralopez1@um.es  
lucia.s1@um.es  

**Asignatura:** Extensiones de Machine Learning  
**Curso:** 2025/2026  

## Descripción

En este repositorio se presentan todos los notebooks y el código desarrollado para la **Práctica 1 de Aprendizaje por Refuerzo** de la asignatura **Extensiones de Machine Learning** del Máster en Inteligencia Artificial de la **Universidad de Murcia**.

El objetivo de esta práctica es estudiar y experimentar con distintos algoritmos de aprendizaje por refuerzo, aplicándolos a dos tipos de problemas:

- Problema de los **k-bandidos**

Para facilitar la ejecución, todos los notebooks pueden abrirse directamente en **Google Colab**.

---

# Estructura del repositorio

El repositorio se divide en dos directorios principales:

## 1. k_brazos

En este directorio se encuentran los experimentos relacionados con el **problema de los k-bandidos**.

Incluye diferentes notebooks donde se aplican distintos algoritmos sobre distintas distribuciones de recompensas.

### Notebooks principales

- [Notebook principal de k-brazos](https://colab.research.google.com/github/dac012/ValeraSanchezMontes-RL/blob/main/k_brazos/main.ipynb)

### Experimentos

- [Notebook usando distribución Bernoulli](https://colab.research.google.com/github/dac012/ValeraSanchezMontes-RL/blob/main/k_brazos/bernoulli.ipynb)

- [Notebook usando distribución Binomial](https://colab.research.google.com/github/dac012/ValeraSanchezMontes-RL/blob/main/k_brazos/binomial.ipynb)

- [Notebook usando distribución Normal](https://colab.research.google.com/github/dac012/ValeraSanchezMontes-RL/blob/main/k_brazos/normal.ipynb)

- [Notebook de análisis de la cota teórica](https://colab.research.google.com/github/dac012/ValeraSanchezMontes-RL/blob/main/k_brazos/cota.ipynb)

El notebook `main.ipynb` sirve como punto de acceso a todos los experimentos y contiene gráficas comparativas entre los diferentes métodos.

---

# Instalación y Uso

No es necesaria ninguna instalación específica.

Los notebooks pueden ejecutarse de dos formas:

### En Google Colab (recomendado)

Simplemente accediendo a los enlaces anteriores.

### En local

1. Clonar el repositorio

```bash
git clone https://github.com/dac012/ValeraSanchezMontes-RL.git
```

2. Mantener la estructura del repositorio.

3. Ejecutar los notebooks con Jupyter Notebook o Jupyter Lab.