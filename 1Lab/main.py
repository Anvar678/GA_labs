# Первая лаба
# Номер варика 1 + (17*31+7*0+26)%20 = 14
# d = 7


import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def f(x:list[float|int], d) -> float|int:
    return 0.5 * sum(x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i] for i in range(d))

def generate_zero_population(population_size: int, d: int) -> np.ndarray:
    return np.random.uniform(-5, 5, size=(population_size, d))

def fitness(X: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(X ** 4 - 16 * X ** 2 + 5 * X, axis=1)

def selection(X: np.ndarray, elitism: int = 10, take_pop: int = 70):
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

    return np.vstack((elite, selected)), y

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

def random_search(samples_count: int, d: int):
    X = generate_zero_population(samples_count, d)
    y = fitness(X)
    best_index = np.argmin(y)

    return X[best_index].copy(), float(y[best_index])

def save_graphs(histories,
                results,
                graphics_path: Path):
    graphics_path.mkdir(exist_ok=True)

    graph_groups = [
        (
            ['generations_100', 'generations_300', 'generations_600'],
            'Influence of generation count',
            'generations.png',
        ),
        (
            ['population_50', 'population_100', 'population_200'],
            'Influence of population size',
            'population.png',
        ),
    ]

    for configuration_names, title, file_name in graph_groups:
        plt.figure(figsize=(10, 6))
        for configuration_name in configuration_names:
            values = np.array(histories[configuration_name])
            generations = np.arange(values.shape[1])
            mean_values = np.mean(values, axis=0)
            plt.plot(generations, mean_values, label=configuration_name)
            plt.fill_between(
                generations,
                np.min(values, axis=0),
                np.max(values, axis=0),
                alpha=0.15,
            )

        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Best function value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(graphics_path / file_name, dpi=150)
        plt.close()

    configuration_names = list(histories.keys())
    ga_mean = []
    random_mean = []
    for configuration_name in configuration_names:
        configuration_results = [
            result for result in results
            if result['configuration'] == configuration_name
        ]
        ga_mean.append(np.mean([result['ga_best_y'] for result in configuration_results]))
        random_mean.append(np.mean([result['random_best_y'] for result in configuration_results]))

    positions = np.arange(len(configuration_names))
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(positions - width / 2, ga_mean, width, label='Genetic algorithm')
    plt.bar(positions + width / 2, random_mean, width, label='Random search')
    plt.xticks(positions, configuration_names, rotation=20)
    plt.ylabel('Mean best function value')
    plt.title('Genetic algorithm and random search')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'ga_vs_random.png', dpi=150)
    plt.close()

def main():
    configurations = [
        ('generations_100', 100, 100, 5, 0.5),
        ('generations_300', 100, 300, 5, 0.5),
        ('generations_600', 100, 600, 5, 0.5),
        ('population_50', 50, 300, 5, 0.5),
        ('population_100', 100, 300, 5, 0.5),
        ('population_200', 200, 300, 5, 0.5),
    ]
    results = []
    histories = {configuration[0]: [] for configuration in configurations}
    elitism = 10
    take_pop = 70
    runs_count = 20
    d = 7

    for configuration in configurations:
        configuration_name, population_size, generations, p_of_mutation, mutation_scale = configuration

        for run in range(1, runs_count + 1):
            seed = 100 + run
            np.random.seed(seed)
            X = generate_zero_population(population_size, d)
            best_history = []

            for generation in range(generations):
                selected_X, y = selection(X, elitism, take_pop)
                best_history.append(float(np.min(y)))
                children_count = population_size - selected_X.shape[0]
                children = crossover(selected_X, children_count)

                elite_count = int(population_size * elitism / 100)
                selected_X[elite_count:] = mutation(
                    selected_X[elite_count:],
                    p_of_mutation,
                    mutation_scale,
                )
                children = mutation(children, p_of_mutation, mutation_scale)
                X = np.vstack((selected_X, children)) # объединяет массивы в матрицу детей и тех кого отобрали

            y = fitness(X)
            best_index = np.argmin(y)
            best_x = X[best_index].copy()
            best_y = float(y[best_index])
            best_history.append(best_y)
            histories[configuration_name].append(best_history)

            fitness_budget = population_size * (generations + 1)
            random_seed = seed + 10000
            np.random.seed(random_seed)
            random_best_x, random_best_y = random_search(fitness_budget, d)

            results.append({
                'configuration': configuration_name,
                'run': run,
                'population_size': population_size,
                'generations': generations,
                'p_of_mutation': p_of_mutation,
                'mutation_scale': mutation_scale,
                'elitism': elitism,
                'take_pop': take_pop,
                'seed': seed,
                'random_seed': random_seed,
                'fitness_budget': fitness_budget,
                'ga_best_y': best_y,
                'ga_best_x': best_x,
                'random_best_y': random_best_y,
                'random_best_x': random_best_x,
                'history': best_history,
            })

    print(f"{'Config':<18}{'Run':<5}{'Population':<12}{'Generations':<13}{'Seed':<7}{'GA best':<15}{'Random best':<15}Best x")
    print('-' * 135)
    for result in results:
        best_x_string = np.array2string(result['ga_best_x'], precision=4, floatmode='fixed')
        print(
            f"{result['configuration']:<18}{result['run']:<5}"
            f"{result['population_size']:<12}{result['generations']:<13}"
            f"{result['seed']:<7}{result['ga_best_y']:<15.6f}"
            f"{result['random_best_y']:<15.6f}{best_x_string}"
        )

    print('\nComparison by configuration')
    print(f"{'Config':<18}{'GA mean':<15}{'GA median':<15}{'GA std':<15}{'GA best':<15}{'Random mean':<15}{'GA wins':<8}")
    print('-' * 101)
    for configuration in configurations:
        configuration_name = configuration[0]
        configuration_results = [
            result for result in results
            if result['configuration'] == configuration_name
        ]
        ga_values = np.array([result['ga_best_y'] for result in configuration_results])
        random_values = np.array([result['random_best_y'] for result in configuration_results])
        print(
            f"{configuration_name:<18}{np.mean(ga_values):<15.6f}"
            f"{np.median(ga_values):<15.6f}{np.std(ga_values):<15.6f}"
            f"{np.min(ga_values):<15.6f}{np.mean(random_values):<15.6f}"
            f"{np.sum(ga_values < random_values):<8}"
        )

    lab_path = Path(__file__).resolve().parent
    csv_path = lab_path / 'Exp.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([
            'configuration', 'run', 'population_size', 'generations',
            'mutation_probability', 'mutation_scale', 'elitism', 'selection_percent',
            'seed', 'random_seed', 'fitness_budget', 'ga_best_y',
            'ga_x1', 'ga_x2', 'ga_x3', 'ga_x4', 'ga_x5', 'ga_x6', 'ga_x7',
            'random_best_y', 'random_x1', 'random_x2', 'random_x3',
            'random_x4', 'random_x5', 'random_x6', 'random_x7', 'ga_history',
        ])
        for result in results:
            writer.writerow([
                result['configuration'], result['run'], result['population_size'],
                result['generations'], result['p_of_mutation'], result['mutation_scale'],
                result['elitism'], result['take_pop'], result['seed'],
                result['random_seed'], result['fitness_budget'], result['ga_best_y'],
                *result['ga_best_x'], result['random_best_y'],
                *result['random_best_x'],
                '|'.join(str(value) for value in result['history']),
            ])

    save_graphs(histories, results, lab_path / 'Grafics')

    return results

if __name__ == '__main__':
    main()
