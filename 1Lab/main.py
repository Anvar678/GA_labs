# Первая лаба
# Номер варика 1 + (17*31+7*0+26)%20 = 14
# d = 7


import numpy as np

def f(x:list[float|int], d) -> float|int:
    return 0.5 * sum(x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i] for i in range(d))


def generate_zero_population(population_size: int, d: int) -> np.ndarray:
    return np.random.uniform(-5, 5, size=(population_size, d))

def fitness(X: np.ndarray) -> np.ndarray:
    return np.array([f(x, X.shape[1]) for x in X])

def selection(X: np.ndarray, elitism: int = 10, take_pop: int = 70) -> np.ndarray:
    y = fitness(X)
    population_size = X.shape[0]

    elite_count = int(population_size * elitism / 100)
    selected_count = int(population_size * take_pop / 100)

    elite_indices = np.argsort(y)[:elite_count]
    elite = X[elite_indices]

    selection_weights = np.max(y) - y
    if np.sum(selection_weights) == 0:
        probabilities = np.full(population_size, 1 / population_size)
    else:
        probabilities = selection_weights / np.sum(selection_weights)

    selected_indices = np.random.choice(
        population_size,
        size=selected_count,
        replace=True,
        p=probabilities,
    )
    selected = X[selected_indices]

    return np.vstack((elite, selected))

def crossover(X: np.ndarray, children_count: int) -> np.ndarray:
    children = np.empty((children_count, X.shape[1]))

    for i in range(children_count):
        parent_indices = np.random.choice(X.shape[0], size=2, replace=False)
        parent_1, parent_2 = X[parent_indices]

        genes_from_parent_1 = np.random.choice(
            X.shape[1],
            size=X.shape[1] // 2,
            replace=False,
        )
        child = parent_2.copy()
        child[genes_from_parent_1] = parent_1[genes_from_parent_1]
        children[i] = child

    return children

def mutation(
    X: np.ndarray,
    p_of_mutation: int = 5,
    mutation_scale: float = 0.5,
) -> np.ndarray:
    mutated_X = X.copy()
    mutation_mask = np.random.random(X.shape) < p_of_mutation / 100
    gene_changes = np.random.normal(0, mutation_scale, size=X.shape)
    mutated_X[mutation_mask] += gene_changes[mutation_mask]

    return np.clip(mutated_X, -5, 5)

def main():
    population_size = 30
    X = generate_zero_population(population_size, 7)

    for generation in range(100):
        selected_X = selection(X)
        children_count = population_size - selected_X.shape[0]
        children = crossover(selected_X, children_count)
        children = mutation(children)
        X = np.vstack((selected_X, children)) # объединяет массивы в матрицу детей и тех кого отобрали

if __name__ == '__main__':
    main()
