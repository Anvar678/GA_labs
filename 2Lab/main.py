# Вторая лаба
# Номер варика 1 + (17*31+7*0+26)%20 = 14

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_zero_population(population_size: int, vertex_count: int, towers_count: int) -> np.ndarray:
    population = np.zeros((population_size, vertex_count), dtype=int)

    for individual in population:
        tower_vertices = np.random.choice(vertex_count, size=towers_count, replace=False)
        individual[tower_vertices] = 1

    return population


def fitness(X: np.ndarray, graph: np.ndarray) -> np.ndarray:
    covered = (X @ graph > 0) | (X == 1) # первая часть находит соседей вышки а вторая учитывает саму вышку
    return np.sum(covered, axis=1)


def selection(X: np.ndarray, graph: np.ndarray, elitism: int = 10, take_pop: int = 70):
    y = fitness(X, graph)
    population_size = X.shape[0]

    elite_count = int(population_size * elitism / 100)
    selected_count = int(population_size * take_pop / 100)

    elite_indices = np.argsort(y)[-elite_count:]
    elite = X[elite_indices]

    selection_weights = y - np.min(y)
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
    children = np.empty((children_count, X.shape[1]), dtype=int)

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


def mutation(X: np.ndarray, p_of_mutation: int = 5) -> np.ndarray:
    mutated_X = X.copy()
    mutation_mask = np.random.random(X.shape) < p_of_mutation / 100
    mutated_X[mutation_mask] = 1 - mutated_X[mutation_mask]

    return mutated_X


def swap_mutation(X: np.ndarray, p_of_mutation: int = 5) -> np.ndarray:
    mutated_X = X.copy()

    for individual in mutated_X:
        if np.random.random() < p_of_mutation / 100:
            tower_vertices = np.where(individual == 1)[0]
            empty_vertices = np.where(individual == 0)[0]

            if len(tower_vertices) > 0 and len(empty_vertices) > 0:
                tower_vertex = np.random.choice(tower_vertices)
                empty_vertex = np.random.choice(empty_vertices)

                individual[tower_vertex] = 0
                individual[empty_vertex] = 1

    return mutated_X


def repair(X: np.ndarray, towers_count: int) -> np.ndarray:
    repaired_X = X.copy()

    for individual in repaired_X:
        while np.sum(individual) > towers_count:
            tower_vertices = np.where(individual == 1)[0]
            vertex = np.random.choice(tower_vertices)
            individual[vertex] = 0

    return repaired_X


def penalty_fitness(X: np.ndarray, graph: np.ndarray, towers_count: int, penalty: int) -> np.ndarray:
    y = fitness(X, graph)
    extra_towers = np.maximum(np.sum(X, axis=1) - towers_count, 0)

    return y - extra_towers * penalty


def penalty_selection(X: np.ndarray, graph: np.ndarray, towers_count: int, penalty: int, elitism: int = 10, take_pop: int = 70):
    y = penalty_fitness(X, graph, towers_count, penalty)
    population_size = X.shape[0]

    elite_count = int(population_size * elitism / 100)
    selected_count = int(population_size * take_pop / 100)

    elite_indices = np.argsort(y)[-elite_count:]
    elite = X[elite_indices]

    selection_weights = y - np.min(y)
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


def generate_graph(vertex_count: int, edge_probability: int = 15) -> np.ndarray:
    graph = np.zeros((vertex_count, vertex_count), dtype=int)

    for i in range(vertex_count - 1):
        graph[i, i + 1] = 1
        graph[i + 1, i] = 1

    graph[0, vertex_count - 1] = 1
    graph[vertex_count - 1, 0] = 1

    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            if graph[i, j] == 0 and np.random.random() < edge_probability / 100:
                graph[i, j] = 1
                graph[j, i] = 1

    return graph


def random_search(samples_count: int, vertex_count: int, towers_count: int, graph: np.ndarray):
    X = generate_zero_population(samples_count, vertex_count, towers_count)
    y = fitness(X, graph)
    best_index = np.argmax(y)

    return X[best_index].copy(), int(y[best_index])


def best_valid_solution(X: np.ndarray, graph: np.ndarray, towers_count: int):
    valid_X = X[np.sum(X, axis=1) <= towers_count]
    y = fitness(valid_X, graph)
    best_index = np.argmax(y)

    return valid_X[best_index].copy(), int(y[best_index])


def save_graph_data(graph: np.ndarray, data_path: Path):
    data_path.mkdir(exist_ok=True)

    with (data_path / 'graph.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['vertex_1', 'vertex_2'])

        for i in range(graph.shape[0]):
            for j in range(i + 1, graph.shape[1]):
                if graph[i, j] == 1:
                    writer.writerow([i, j])


def save_examples(graph: np.ndarray, towers_count: int, lab_path: Path):
    examples = [
        ('valid_1', np.array([1] * towers_count + [0] * (graph.shape[0] - towers_count))),
        ('valid_2', np.array([0, 1] * towers_count + [0] * (graph.shape[0] - towers_count * 2))),
        ('invalid_1', np.array([1] * (towers_count + 1) + [0] * (graph.shape[0] - towers_count - 1))),
        ('invalid_2', np.array([1] * (towers_count + 2) + [0] * (graph.shape[0] - towers_count - 2))),
    ]

    with (lab_path / 'Examples.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'towers', 'coverage', 'valid', 'individual'])

        for name, individual in examples:
            coverage = fitness(individual.reshape(1, -1), graph)[0]
            towers = np.sum(individual)
            valid = towers <= towers_count
            writer.writerow([name, towers, coverage, valid, '|'.join(str(value) for value in individual)])


def save_configurations(configurations, lab_path: Path, vertex_count: int, towers_count: int, graph_seed: int, edge_probability: int, elitism: int, take_pop: int, penalty: int, runs_count: int):
    with (lab_path / 'Configurations.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['configuration', 'constraint_method', 'mutation_type', 'population_size', 'generations', 'mutation_probability', 'vertex_count', 'towers_count', 'graph_seed', 'edge_probability', 'elitism', 'selection_percent', 'penalty', 'runs_count'])

        for configuration in configurations:
            configuration_name, constraint_method, mutation_type, population_size, generations, p_of_mutation = configuration
            writer.writerow([configuration_name, constraint_method, mutation_type, population_size, generations, p_of_mutation, vertex_count, towers_count, graph_seed, edge_probability, elitism, take_pop, penalty, runs_count])


def save_best_solution(graph: np.ndarray, best_x: np.ndarray, graphics_path: Path):
    vertex_count = graph.shape[0]
    angles = np.linspace(0, 2 * np.pi, vertex_count, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    covered = (best_x @ graph > 0) | (best_x == 1)
    towers = best_x == 1

    plt.figure(figsize=(10, 10))

    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            if graph[i, j] == 1:
                plt.plot([x[i], x[j]], [y[i], y[j]], color='gray', alpha=0.3)

    plt.scatter(x[covered & ~towers], y[covered & ~towers], s=200, label='Covered')
    plt.scatter(x[~covered], y[~covered], s=200, label='Not covered')
    plt.scatter(x[towers], y[towers], s=350, marker='*', label='Tower')

    for i in range(vertex_count):
        plt.text(x[i], y[i], str(i), ha='center', va='center', fontsize=8)

    plt.title(f'Best solution: {np.sum(covered)}/{vertex_count} vertices covered')
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'best_solution.png', dpi=150)
    plt.close()


def save_graphs(histories,
                results,
                graphics_path: Path,
                graph: np.ndarray,
                best_x: np.ndarray):
    graphics_path.mkdir(exist_ok=True)

    graph_groups = [
        (
            ['repair_bit', 'penalty_bit'],
            'Constraint handling comparison',
            'constraints.png',
        ),
        (
            ['repair_bit', 'repair_swap'],
            'Mutation comparison',
            'mutations.png',
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
        plt.ylabel('Best coverage')
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
    plt.ylabel('Mean best coverage')
    plt.title('Genetic algorithm and random search')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'ga_vs_random.png', dpi=150)
    plt.close()

    valid_mean = []
    for configuration_name in configuration_names:
        configuration_results = [
            result for result in results
            if result['configuration'] == configuration_name
        ]
        valid_mean.append(np.mean([result['valid_percent'] for result in configuration_results]))

    plt.figure(figsize=(10, 6))
    plt.bar(configuration_names, valid_mean)
    plt.xticks(rotation=20)
    plt.ylabel('Valid solutions, %')
    plt.title('Valid solutions in final population')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphics_path / 'valid_solutions.png', dpi=150)
    plt.close()

    save_best_solution(graph, best_x, graphics_path)


def main():
    configurations = [
        ('repair_bit', 'repair', 'bit', 100, 300, 5),
        ('penalty_bit', 'penalty', 'bit', 100, 300, 5),
        ('repair_swap', 'repair', 'swap', 100, 300, 5),
    ]
    results = []
    histories = {configuration[0]: [] for configuration in configurations}
    elitism = 10
    take_pop = 70
    runs_count = 20

    vertex_count = 30
    towers_count = 5
    graph_seed = 1400
    edge_probability = 15
    penalty = vertex_count + 1

    lab_path = Path(__file__).resolve().parent

    np.random.seed(graph_seed)
    graph = generate_graph(vertex_count, edge_probability)

    save_graph_data(graph, lab_path / 'Data')
    save_examples(graph, towers_count, lab_path)
    save_configurations(configurations, lab_path, vertex_count, towers_count, graph_seed, edge_probability, elitism, take_pop, penalty, runs_count)

    for configuration in configurations:
        configuration_name, constraint_method, mutation_type, population_size, generations, p_of_mutation = configuration

        for run in range(1, runs_count + 1):
            seed = 200 + run
            np.random.seed(seed)
            X = generate_zero_population(population_size, vertex_count, towers_count)
            best_history = []

            for generation in range(generations):
                if constraint_method == 'repair':
                    selected_X, y = selection(X, graph, elitism, take_pop)
                else:
                    selected_X, y = penalty_selection(X, graph, towers_count, penalty, elitism, take_pop)

                valid_X = X[np.sum(X, axis=1) <= towers_count]
                valid_y = fitness(valid_X, graph)
                best_history.append(float(np.max(valid_y)))

                children_count = population_size - selected_X.shape[0]
                children = crossover(selected_X, children_count)

                elite_count = int(population_size * elitism / 100)

                if mutation_type == 'bit':
                    selected_X[elite_count:] = mutation(selected_X[elite_count:], p_of_mutation)
                    children = mutation(children, p_of_mutation)
                else:
                    selected_X[elite_count:] = swap_mutation(selected_X[elite_count:], p_of_mutation)
                    children = swap_mutation(children, p_of_mutation)

                if constraint_method == 'repair':
                    selected_X[elite_count:] = repair(selected_X[elite_count:], towers_count)
                    children = repair(children, towers_count)

                X = np.vstack((selected_X, children)) # объединяет массивы в матрицу детей и тех кого отобрали

            best_x, best_y = best_valid_solution(X, graph, towers_count)
            best_history.append(best_y)
            histories[configuration_name].append(best_history)

            valid_percent = np.mean(np.sum(X, axis=1) <= towers_count) * 100

            fitness_budget = population_size * (generations + 1)
            random_seed = seed + 10000
            np.random.seed(random_seed)
            random_best_x, random_best_y = random_search(fitness_budget, vertex_count, towers_count, graph)

            results.append({
                'configuration': configuration_name,
                'run': run,
                'population_size': population_size,
                'generations': generations,
                'p_of_mutation': p_of_mutation,
                'constraint_method': constraint_method,
                'mutation_type': mutation_type,
                'elitism': elitism,
                'take_pop': take_pop,
                'seed': seed,
                'random_seed': random_seed,
                'fitness_budget': fitness_budget,
                'ga_best_y': best_y,
                'ga_best_x': best_x,
                'valid_percent': valid_percent,
                'random_best_y': random_best_y,
                'random_best_x': random_best_x,
                'history': best_history,
            })

    print(f"{'Config':<18}{'Run':<5}{'Population':<12}{'Generations':<13}{'Seed':<7}{'GA best':<12}{'Random best':<15}{'Valid %':<10}Best x")
    print('-' * 145)
    for result in results:
        best_x_string = np.array2string(result['ga_best_x'])
        print(
            f"{result['configuration']:<18}{result['run']:<5}"
            f"{result['population_size']:<12}{result['generations']:<13}"
            f"{result['seed']:<7}{result['ga_best_y']:<12}"
            f"{result['random_best_y']:<15}{result['valid_percent']:<10.2f}"
            f"{best_x_string}"
        )

    print('\nComparison by configuration')
    print(f"{'Config':<18}{'GA mean':<15}{'GA median':<15}{'GA std':<15}{'GA best':<12}{'GA worst':<12}{'Random mean':<15}{'GA wins':<8}")
    print('-' * 110)
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
            f"{np.max(ga_values):<12}{np.min(ga_values):<12}"
            f"{np.mean(random_values):<15.6f}{np.sum(ga_values > random_values):<8}"
        )

    csv_path = lab_path / 'Exp.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['configuration', 'run', 'population_size', 'generations', 'mutation_probability', 'constraint_method', 'mutation_type', 'elitism', 'selection_percent', 'seed', 'random_seed', 'fitness_budget', 'ga_best_y', 'ga_towers', 'valid_percent', 'ga_best_x', 'random_best_y', 'random_best_x', 'ga_history'])

        for result in results:
            writer.writerow([result['configuration'], result['run'], result['population_size'], result['generations'], result['p_of_mutation'], result['constraint_method'], result['mutation_type'], result['elitism'], result['take_pop'], result['seed'], result['random_seed'], result['fitness_budget'], result['ga_best_y'], np.sum(result['ga_best_x']), result['valid_percent'], '|'.join(str(value) for value in result['ga_best_x']), result['random_best_y'], '|'.join(str(value) for value in result['random_best_x']), '|'.join(str(value) for value in result['history'])])

    best_result = max(results, key=lambda result: result['ga_best_y'])
    save_graphs(histories, results, lab_path / 'Grafics', graph, best_result['ga_best_x'])

    return results


if __name__ == '__main__':
    main()