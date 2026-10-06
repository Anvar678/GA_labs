import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_equipment(equipment_seed: int):
    names = np.array([
        'Tent', 'Tarp', 'Bivy', 'Groundsheet', 'Tent_repair_kit', 'Stakes',
        'Sleeping_bag', 'Sleeping_mat', 'Liner', 'Blanket', 'Pillow', 'Emergency_bag',
        'Map', 'Compass', 'GPS_navigator', 'Satellite_beacon', 'Headlamp', 'Batteries',
        'Gas_stove', 'Fuel', 'Pot', 'Water_filter', 'Flask', 'Utensils',
        'First_aid_kit', 'Medicines', 'Bandages', 'Antiseptic', 'Splint', 'Thermal_blanket',
    ])
    category_names = np.array(['Shelter', 'Sleep', 'Navigation', 'Cooking', 'Medical'])
    category_ids = np.repeat(np.arange(category_names.size), 6)

    np.random.seed(equipment_seed)
    utility = np.random.randint(25, 101, size=names.size)
    mass = np.round(np.random.uniform(0.2, 4.2, size=names.size), 2)
    reliability = np.round(np.random.uniform(0.80, 0.99, size=names.size), 3)

    return {
        'names': names,
        'category_names': category_names,
        'category_ids': category_ids,
        'utility': utility,
        'mass': mass,
        'reliability': reliability,
    }


def repair(X: np.ndarray, equipment, maximum_mass: float) -> np.ndarray:
    repaired_X = X.copy()
    category_ids = equipment['category_ids']
    utility = equipment['utility']
    mass = equipment['mass']
    reliability = equipment['reliability']
    category_count = equipment['category_names'].size
    item_quality = utility * reliability / mass

    for category in range(category_count):
        category_items = np.where(category_ids == category)[0]
        missing_rows = np.sum(repaired_X[:, category_items], axis=1) == 0
        best_item = category_items[np.argmax(item_quality[category_items])]
        repaired_X[missing_rows, best_item] = 1

    overweight_rows = np.where(repaired_X @ mass > maximum_mass + 1e-9)[0]
    for row in overweight_rows:
        individual = repaired_X[row]
        while individual @ mass > maximum_mass + 1e-9:
            selected_items = np.where(individual == 1)[0]
            category_counts = np.bincount(
                category_ids[selected_items],
                minlength=category_count,
            )
            removable_items = selected_items[category_counts[category_ids[selected_items]] > 1]

            if removable_items.size == 0:
                break

            worst_item = removable_items[np.argmin(item_quality[removable_items])]
            individual[worst_item] = 0

    return repaired_X


def generate_population(population_size: int, equipment, maximum_mass: float) -> np.ndarray:
    item_count = equipment['names'].size
    category_ids = equipment['category_ids']
    category_count = equipment['category_names'].size
    population = np.zeros((population_size, item_count), dtype=int)
    rows = np.arange(population_size)

    for category in range(category_count):
        category_items = np.where(category_ids == category)[0]
        selected_items = np.random.choice(category_items, size=population_size)
        population[rows, selected_items] = 1

    target_count = np.random.randint(category_count, 15, size=population_size)
    additional_probability = (target_count - category_count) / (item_count - category_count)
    additional_items = np.random.random((population_size, item_count)) < additional_probability[:, np.newaxis]
    population = np.maximum(population, additional_items.astype(int))

    return repair(population, equipment, maximum_mass)


def is_valid(X: np.ndarray, equipment, maximum_mass: float) -> np.ndarray:
    category_ids = equipment['category_ids']
    category_count = equipment['category_names'].size
    valid = X @ equipment['mass'] <= maximum_mass + 1e-9

    for category in range(category_count):
        category_items = np.where(category_ids == category)[0]
        valid &= np.sum(X[:, category_items], axis=1) > 0

    return valid


def objective_values(X: np.ndarray, equipment) -> np.ndarray:
    utility = X @ equipment['utility']
    mass = X @ equipment['mass']
    category_ids = equipment['category_ids']
    category_count = equipment['category_names'].size
    reliability = np.ones(X.shape[0])

    for category in range(category_count):
        category_items = np.where(category_ids == category)[0]
        category_X = X[:, category_items]
        category_reliability = 1 - np.prod(
            np.where(category_X == 1, 1 - equipment['reliability'][category_items], 1),
            axis=1,
        )
        reliability *= category_reliability

    return np.column_stack((-utility, mass, -reliability * 100))


def balanced_score(values: np.ndarray, equipment, maximum_mass: float) -> np.ndarray:
    maximum_utility = np.sum(equipment['utility'])
    utility_score = -values[:, 0] / maximum_utility
    mass_score = 1 - values[:, 1] / maximum_mass
    reliability_score = -values[:, 2] / 100

    return (utility_score + mass_score + reliability_score) / 3


def non_dominated_sort(values: np.ndarray):
    less_or_equal = np.all(values[:, np.newaxis, :] <= values[np.newaxis, :, :], axis=2)
    strictly_less = np.any(values[:, np.newaxis, :] < values[np.newaxis, :, :], axis=2)
    domination_matrix = less_or_equal & strictly_less
    domination_count = np.sum(domination_matrix, axis=0)
    assigned = np.zeros(values.shape[0], dtype=bool)
    fronts = []
    front = np.where(domination_count == 0)[0]

    while front.size > 0:
        fronts.append(front)
        assigned[front] = True
        domination_count -= np.sum(domination_matrix[front], axis=0)
        front = np.where((domination_count == 0) & ~assigned)[0]

    return fronts


def crowding_distance(values: np.ndarray, front: np.ndarray) -> np.ndarray:
    distance = np.zeros(front.size)

    if front.size <= 2:
        distance[:] = np.inf
        return distance

    front_values = values[front]

    for objective in range(values.shape[1]):
        order = np.argsort(front_values[:, objective])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        objective_range = front_values[order[-1], objective] - front_values[order[0], objective]

        if objective_range > 0:
            for position in range(1, front.size - 1):
                if not np.isinf(distance[order[position]]):
                    previous_value = front_values[order[position - 1], objective]
                    next_value = front_values[order[position + 1], objective]
                    distance[order[position]] += (next_value - previous_value) / objective_range

    return distance


def rank_and_crowding(values: np.ndarray):
    fronts = non_dominated_sort(values)
    ranks = np.zeros(values.shape[0], dtype=int)
    distances = np.zeros(values.shape[0])

    for rank, front in enumerate(fronts):
        ranks[front] = rank
        distances[front] = crowding_distance(values, front)

    return fronts, ranks, distances


def selection(X: np.ndarray, ranks: np.ndarray, distances: np.ndarray, selected_count: int) -> np.ndarray:
    selected_indices = []

    for _ in range(selected_count):
        first, second = np.random.choice(X.shape[0], size=2, replace=False)

        if ranks[first] < ranks[second]:
            selected_indices.append(first)
        elif ranks[second] < ranks[first]:
            selected_indices.append(second)
        elif distances[first] > distances[second]:
            selected_indices.append(first)
        elif distances[second] > distances[first]:
            selected_indices.append(second)
        else:
            selected_indices.append(np.random.choice([first, second]))

    return X[selected_indices]


def weighted_selection(X: np.ndarray, values: np.ndarray, weights: np.ndarray, equipment, maximum_mass: float) -> np.ndarray:
    scores = weighted_score(values, weights, equipment, maximum_mass)
    selection_weights = scores - np.min(scores)

    if np.sum(selection_weights) == 0:
        probabilities = np.full(X.shape[0], 1 / X.shape[0])
    else:
        probabilities = selection_weights / np.sum(selection_weights)

    selected_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=probabilities)

    return X[selected_indices]


def crossover(X: np.ndarray, children_count: int, crossover_type: str) -> np.ndarray:
    children = np.empty((children_count, X.shape[1]), dtype=int)

    for child_index in range(children_count):
        parent_indices = np.random.choice(X.shape[0], size=2, replace=False)
        parent_1, parent_2 = X[parent_indices]

        if crossover_type == 'uniform':
            genes_from_parent_1 = np.random.random(X.shape[1]) < 0.5
            child = parent_2.copy()
            child[genes_from_parent_1] = parent_1[genes_from_parent_1]
        else:
            point = np.random.randint(1, X.shape[1])
            child = np.hstack((parent_1[:point], parent_2[point:]))

        children[child_index] = child

    return children


def mutation(X: np.ndarray, p_of_mutation: int = 5) -> np.ndarray:
    mutated_X = X.copy()
    mutation_mask = np.random.random(X.shape) < p_of_mutation / 100
    mutated_X[mutation_mask] = 1 - mutated_X[mutation_mask]

    return mutated_X


def environmental_selection(X: np.ndarray, values: np.ndarray, population_size: int) -> np.ndarray:
    fronts = non_dominated_sort(values)
    selected_indices = []

    for front in fronts:
        remaining_count = population_size - len(selected_indices)

        if front.size <= remaining_count:
            selected_indices.extend(front)
        else:
            distances = crowding_distance(values, front)
            order = np.argsort(distances)[::-1]
            selected_indices.extend(front[order[:remaining_count]])
            break

    return X[selected_indices]


def weighted_score(values: np.ndarray, weights: np.ndarray, equipment, maximum_mass: float) -> np.ndarray:
    maximum_utility = np.sum(equipment['utility'])
    utility_score = -values[:, 0] / maximum_utility
    mass_score = 1 - values[:, 1] / maximum_mass
    reliability_score = -values[:, 2] / 100

    return weights[0] * utility_score + weights[1] * mass_score + weights[2] * reliability_score


def make_history(values: np.ndarray, fronts, equipment, maximum_mass: float):
    front = fronts[0]

    return {
        'balanced': float(np.max(balanced_score(values[front], equipment, maximum_mass))),
        'utility': float(np.max(-values[front, 0])),
        'mass': float(np.min(values[front, 1])),
        'reliability': float(np.max(-values[front, 2])),
        'front_size': int(front.size),
    }


def nsga_ii(population_size: int,
            generations: int,
            p_of_mutation: int,
            crossover_type: str,
            equipment,
            maximum_mass: float):
    X = generate_population(population_size, equipment, maximum_mass)
    history = []

    for _ in range(generations):
        values = objective_values(X, equipment)
        fronts, ranks, distances = rank_and_crowding(values)
        history.append(make_history(values, fronts, equipment, maximum_mass))

        parents = selection(X, ranks, distances, population_size)
        children = crossover(parents, population_size, crossover_type)
        children = mutation(children, p_of_mutation)
        children = repair(children, equipment, maximum_mass)

        combined_X = np.vstack((X, children))
        combined_values = objective_values(combined_X, equipment)
        X = environmental_selection(combined_X, combined_values, population_size)

    values = objective_values(X, equipment)
    fronts = non_dominated_sort(values)
    history.append(make_history(values, fronts, equipment, maximum_mass))

    return X, values, X[fronts[0]].copy(), values[fronts[0]].copy(), history


def weighted_sum(population_size: int,
                 generations: int,
                 p_of_mutation: int,
                 crossover_type: str,
                 weights: np.ndarray,
                 equipment,
                 maximum_mass: float,
                 elitism: int = 10):
    X = generate_population(population_size, equipment, maximum_mass)
    history = []
    elite_count = int(population_size * elitism / 100)

    for _ in range(generations):
        values = objective_values(X, equipment)
        fronts = non_dominated_sort(values)
        history.append(make_history(values, fronts, equipment, maximum_mass))

        scores = weighted_score(values, weights, equipment, maximum_mass)
        elite_indices = np.argsort(scores)[-elite_count:]
        elite = X[elite_indices]
        parents = weighted_selection(X, values, weights, equipment, maximum_mass)
        children = crossover(parents, population_size - elite_count, crossover_type)
        children = mutation(children, p_of_mutation)
        children = repair(children, equipment, maximum_mass)
        X = np.vstack((elite, children))

    values = objective_values(X, equipment)
    fronts = non_dominated_sort(values)
    history.append(make_history(values, fronts, equipment, maximum_mass))

    return X, values, X[fronts[0]].copy(), values[fronts[0]].copy(), history


def random_search(samples_count: int, equipment, maximum_mass: float):
    X = generate_population(samples_count, equipment, maximum_mass)
    values = objective_values(X, equipment)
    best_index = np.argmax(balanced_score(values, equipment, maximum_mass))

    return X[best_index:best_index + 1].copy(), values[best_index:best_index + 1].copy()


def merge_fronts(fronts_X, fronts_values):
    X = np.vstack(fronts_X)
    values = np.vstack(fronts_values)
    X, unique_indices = np.unique(X, axis=0, return_index=True)
    values = values[unique_indices]
    fronts = non_dominated_sort(values)

    return X[fronts[0]], values[fronts[0]]


def selected_names(individual: np.ndarray, equipment) -> str:
    return '|'.join(equipment['names'][individual == 1])


def characteristic_solutions(X: np.ndarray, values: np.ndarray, equipment, maximum_mass: float):
    candidate_indices = [
        ('lightest', int(np.argmin(values[:, 1]))),
        ('most_useful', int(np.argmax(-values[:, 0]))),
        ('most_reliable', int(np.argmax(-values[:, 2]))),
        ('balanced', int(np.argmax(balanced_score(values, equipment, maximum_mass)))),
    ]
    selected = []
    used_indices = set()

    for name, index in candidate_indices:
        if index not in used_indices:
            selected.append((name, index))
            used_indices.add(index)

    for index in np.argsort(balanced_score(values, equipment, maximum_mass))[::-1]:
        if len(selected) >= 4:
            break
        if int(index) not in used_indices:
            selected.append(('alternative', int(index)))
            used_indices.add(int(index))

    return selected


def save_equipment(equipment, lab_path: Path, equipment_seed: int):
    with (lab_path / 'Equipment.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'name', 'category', 'utility', 'mass_kg', 'reliability', 'seed'])

        for item in range(equipment['names'].size):
            category = equipment['category_names'][equipment['category_ids'][item]]
            writer.writerow([
                item + 1,
                equipment['names'][item],
                category,
                equipment['utility'][item],
                equipment['mass'][item],
                equipment['reliability'][item],
                equipment_seed,
            ])


def save_examples(equipment, maximum_mass: float, lab_path: Path):
    item_count = equipment['names'].size
    category_ids = equipment['category_ids']
    category_count = equipment['category_names'].size
    mass = equipment['mass']
    utility = equipment['utility']
    reliability = equipment['reliability']
    quality = utility * reliability / mass
    valid_1 = np.zeros(item_count, dtype=int)

    for category in range(category_count):
        category_items = np.where(category_ids == category)[0]
        valid_1[category_items[np.argmax(quality[category_items])]] = 1

    valid_2 = valid_1.copy()
    for item in np.argsort(quality)[::-1]:
        if valid_2[item] == 0 and valid_2 @ mass + mass[item] <= maximum_mass:
            valid_2[item] = 1
        if np.sum(valid_2) >= 10:
            break

    invalid_1 = valid_1.copy()
    invalid_1[np.where(category_ids == 2)[0]] = 0
    invalid_2 = np.ones(item_count, dtype=int)
    examples = [
        ('valid_light', valid_1),
        ('valid_extended', valid_2),
        ('invalid_without_navigation', invalid_1),
        ('invalid_overweight', invalid_2),
    ]

    with (lab_path / 'Examples.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'utility', 'mass_kg', 'reliability_percent', 'valid', 'equipment', 'individual'])

        for name, individual in examples:
            values = objective_values(individual.reshape(1, -1), equipment)[0]
            valid = is_valid(individual.reshape(1, -1), equipment, maximum_mass)[0]
            writer.writerow([
                name,
                -values[0],
                values[1],
                -values[2],
                valid,
                selected_names(individual, equipment),
                '|'.join(str(value) for value in individual),
            ])


def save_configurations(configurations,
                        lab_path: Path,
                        equipment_seed: int,
                        item_count: int,
                        category_count: int,
                        maximum_mass: float,
                        runs_count: int,
                        elitism: int):
    with (lab_path / 'Configurations.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['configuration', 'algorithm', 'crossover_type', 'population_size', 'generations', 'mutation_probability', 'equipment_seed', 'item_count', 'category_count', 'maximum_mass_kg', 'runs_count', 'elitism'])

        for configuration in configurations:
            configuration_name, algorithm, crossover_type, population_size, generations, p_of_mutation = configuration
            writer.writerow([
                configuration_name,
                algorithm,
                crossover_type,
                population_size,
                generations,
                p_of_mutation,
                equipment_seed,
                item_count,
                category_count,
                maximum_mass,
                runs_count,
                elitism,
            ])


def save_pareto_solutions(fronts, equipment, maximum_mass: float, lab_path: Path):
    with (lab_path / 'Pareto_solutions.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['configuration', 'solution_type', 'utility', 'mass_kg', 'reliability_percent', 'balanced_score', 'equipment', 'individual'])

        for configuration_name, (X, values) in fronts.items():
            for solution_type, index in characteristic_solutions(X, values, equipment, maximum_mass):
                writer.writerow([
                    configuration_name,
                    solution_type,
                    -values[index, 0],
                    values[index, 1],
                    -values[index, 2],
                    balanced_score(values[index:index + 1], equipment, maximum_mass)[0],
                    selected_names(X[index], equipment),
                    '|'.join(str(value) for value in X[index]),
                ])


def save_graphs(histories, results, fronts, graphics_path: Path, equipment, maximum_mass: float):
    graphics_path.mkdir(exist_ok=True)
    configuration_names = list(histories.keys())

    plt.figure(figsize=(10, 6))
    for configuration_name in configuration_names:
        values = np.array([
            [generation['balanced'] for generation in history]
            for history in histories[configuration_name]
        ])
        generations = np.arange(values.shape[1])
        plt.plot(generations, np.mean(values, axis=0), label=configuration_name)
        plt.fill_between(generations, np.min(values, axis=0), np.max(values, axis=0), alpha=0.15)

    plt.title('Balanced score by generation')
    plt.xlabel('Generation')
    plt.ylabel('Balanced score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'convergence.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for configuration_name in configuration_names:
        values = np.array([
            [generation['front_size'] for generation in history]
            for history in histories[configuration_name]
        ])
        generations = np.arange(values.shape[1])
        plt.plot(generations, np.mean(values, axis=0), label=configuration_name)

    plt.title('Pareto front size by generation')
    plt.xlabel('Generation')
    plt.ylabel('Solutions in first front')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'front_size.png', dpi=150)
    plt.close()

    nsga_values = fronts['nsga_uniform'][1]
    utility = -nsga_values[:, 0]
    mass = nsga_values[:, 1]
    reliability = -nsga_values[:, 2]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    first = axes[0].scatter(mass, utility, c=reliability, cmap='viridis')
    axes[0].set_xlabel('Mass, kg')
    axes[0].set_ylabel('Utility')
    axes[0].set_title('Mass and utility')
    figure.colorbar(first, ax=axes[0], label='Reliability, %')
    second = axes[1].scatter(reliability, utility, c=mass, cmap='plasma')
    axes[1].set_xlabel('Reliability, %')
    axes[1].set_ylabel('Utility')
    axes[1].set_title('Reliability and utility')
    figure.colorbar(second, ax=axes[1], label='Mass, kg')
    third = axes[2].scatter(mass, reliability, c=utility, cmap='magma')
    axes[2].set_xlabel('Mass, kg')
    axes[2].set_ylabel('Reliability, %')
    axes[2].set_title('Mass and reliability')
    figure.colorbar(third, ax=axes[2], label='Utility')
    figure.suptitle('NSGA-II Pareto front projections')
    figure.tight_layout()
    figure.savefig(graphics_path / 'pareto_front.png', dpi=150)
    plt.close(figure)

    plt.figure(figsize=(10, 6))
    for configuration_name in ['nsga_uniform', 'weighted_sum']:
        values = fronts[configuration_name][1]
        plt.scatter(values[:, 1], -values[:, 0], label=configuration_name, alpha=0.75)

    plt.title('NSGA-II and weighted sum')
    plt.xlabel('Mass, kg')
    plt.ylabel('Utility')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'pareto_comparison.png', dpi=150)
    plt.close()

    ga_mean = []
    random_mean = []
    for configuration_name in configuration_names:
        configuration_results = [
            result for result in results
            if result['configuration'] == configuration_name
        ]
        ga_mean.append(np.mean([result['ga_balanced_score'] for result in configuration_results]))
        random_mean.append(np.mean([result['random_balanced_score'] for result in configuration_results]))

    positions = np.arange(len(configuration_names))
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(positions - width / 2, ga_mean, width, label='Evolutionary algorithm')
    plt.bar(positions + width / 2, random_mean, width, label='Random search')
    plt.xticks(positions, configuration_names, rotation=20)
    plt.ylabel('Best balanced score')
    plt.title('Evolutionary algorithms and random search')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics_path / 'ga_vs_random.png', dpi=150)
    plt.close()


def main():
    configurations = [
        ('nsga_uniform', 'nsga_ii', 'uniform', 100, 300, 5),
        ('nsga_one_point', 'nsga_ii', 'one_point', 100, 300, 5),
        ('weighted_sum', 'weighted_sum', 'uniform', 100, 300, 5),
    ]
    weighted_sets = [
        np.array([0.60, 0.20, 0.20]),
        np.array([0.20, 0.60, 0.20]),
        np.array([0.20, 0.20, 0.60]),
        np.array([0.34, 0.33, 0.33]),
    ]
    results = []
    histories = {configuration[0]: [] for configuration in configurations}
    fronts_X = {configuration[0]: [] for configuration in configurations}
    fronts_values = {configuration[0]: [] for configuration in configurations}
    runs_count = 20
    elitism = 10
    equipment_seed = 1400
    maximum_mass = 25.0
    lab_path = Path(__file__).resolve().parent
    equipment = generate_equipment(equipment_seed)

    save_equipment(equipment, lab_path, equipment_seed)
    save_examples(equipment, maximum_mass, lab_path)
    save_configurations(
        configurations,
        lab_path,
        equipment_seed,
        equipment['names'].size,
        equipment['category_names'].size,
        maximum_mass,
        runs_count,
        elitism,
    )

    for configuration in configurations:
        configuration_name, algorithm, crossover_type, population_size, generations, p_of_mutation = configuration

        for run in range(1, runs_count + 1):
            seed = 200 + run
            weights = weighted_sets[(run - 1) % len(weighted_sets)]
            np.random.seed(seed)

            if algorithm == 'nsga_ii':
                X, values, pareto_X, pareto_values, history = nsga_ii(
                    population_size,
                    generations,
                    p_of_mutation,
                    crossover_type,
                    equipment,
                    maximum_mass,
                )
            else:
                X, values, pareto_X, pareto_values, history = weighted_sum(
                    population_size,
                    generations,
                    p_of_mutation,
                    crossover_type,
                    weights,
                    equipment,
                    maximum_mass,
                    elitism,
                )

            fitness_budget = population_size * (generations + 1)
            random_seed = seed + 10000
            np.random.seed(random_seed)
            random_X, random_values = random_search(fitness_budget, equipment, maximum_mass)
            best_index = np.argmax(balanced_score(pareto_values, equipment, maximum_mass))
            random_best_index = np.argmax(balanced_score(random_values, equipment, maximum_mass))

            histories[configuration_name].append(history)
            fronts_X[configuration_name].append(pareto_X)
            fronts_values[configuration_name].append(pareto_values)
            results.append({
                'configuration': configuration_name,
                'algorithm': algorithm,
                'run': run,
                'population_size': population_size,
                'generations': generations,
                'p_of_mutation': p_of_mutation,
                'crossover_type': crossover_type,
                'weights': weights,
                'seed': seed,
                'random_seed': random_seed,
                'fitness_budget': fitness_budget,
                'pareto_size': pareto_X.shape[0],
                'ga_best_x': pareto_X[best_index],
                'ga_best_values': pareto_values[best_index],
                'ga_balanced_score': balanced_score(pareto_values[best_index:best_index + 1], equipment, maximum_mass)[0],
                'random_best_x': random_X[random_best_index],
                'random_best_values': random_values[random_best_index],
                'random_balanced_score': balanced_score(random_values[random_best_index:random_best_index + 1], equipment, maximum_mass)[0],
                'valid_percent': np.mean(is_valid(X, equipment, maximum_mass)) * 100,
                'history': history,
            })

    fronts = {}
    for configuration in configurations:
        configuration_name = configuration[0]
        fronts[configuration_name] = merge_fronts(
            fronts_X[configuration_name],
            fronts_values[configuration_name],
        )

    print(f"{'Конфигурация':<18}{'Запуск':<8}{'Популяция':<12}{'Генерации':<13}{'Сид':<7}{'Парето':<8}{'Оценка ГА':<12}{'Случ. оценка':<15}{'Допустимо %':<12}Сбалансированный набор")
    print('-' * 175)
    for result in results:
        equipment_names = selected_names(result['ga_best_x'], equipment)
        print(
            f"{result['configuration']:<18}{result['run']:<5}"
            f"{result['population_size']:<12}{result['generations']:<13}"
            f"{result['seed']:<7}{result['pareto_size']:<8}"
            f"{result['ga_balanced_score']:<12.6f}{result['random_balanced_score']:<15.6f}"
            f"{result['valid_percent']:<10.2f}{equipment_names}"
        )

    print('\nСравнение конфигураций')
    print(f"{'Конфигурация':<18}{'Размер Парето':<15}{'Макс. полезность':<18}{'Мин. масса':<15}{'Макс. надёжность':<20}{'Средняя оценка':<18}{'Случ. среднее':<16}{'Победы ГА':<10}")
    print('-' * 125)
    for configuration in configurations:
        configuration_name = configuration[0]
        configuration_results = [
            result for result in results
            if result['configuration'] == configuration_name
        ]
        pareto_values = fronts[configuration_name][1]
        ga_scores = np.array([result['ga_balanced_score'] for result in configuration_results])
        random_scores = np.array([result['random_balanced_score'] for result in configuration_results])
        print(
            f"{configuration_name:<18}{pareto_values.shape[0]:<15}"
            f"{np.max(-pareto_values[:, 0]):<15.2f}{np.min(pareto_values[:, 1]):<15.2f}"
            f"{np.max(-pareto_values[:, 2]):<18.2f}{np.mean(ga_scores):<15.6f}"
            f"{np.mean(random_scores):<15.6f}{np.sum(ga_scores > random_scores):<8}"
        )

    with (lab_path / 'Exp.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['configuration', 'algorithm', 'run', 'population_size', 'generations', 'mutation_probability', 'crossover_type', 'weights', 'seed', 'random_seed', 'fitness_budget', 'pareto_size', 'ga_utility', 'ga_mass_kg', 'ga_reliability_percent', 'ga_balanced_score', 'ga_equipment', 'random_utility', 'random_mass_kg', 'random_reliability_percent', 'random_balanced_score', 'random_equipment', 'valid_percent', 'balanced_history', 'front_size_history'])

        for result in results:
            writer.writerow([
                result['configuration'],
                result['algorithm'],
                result['run'],
                result['population_size'],
                result['generations'],
                result['p_of_mutation'],
                result['crossover_type'],
                '|'.join(f'{weight:.2f}' for weight in result['weights']),
                result['seed'],
                result['random_seed'],
                result['fitness_budget'],
                result['pareto_size'],
                -result['ga_best_values'][0],
                result['ga_best_values'][1],
                -result['ga_best_values'][2],
                result['ga_balanced_score'],
                selected_names(result['ga_best_x'], equipment),
                -result['random_best_values'][0],
                result['random_best_values'][1],
                -result['random_best_values'][2],
                result['random_balanced_score'],
                selected_names(result['random_best_x'], equipment),
                result['valid_percent'],
                '|'.join(str(generation['balanced']) for generation in result['history']),
                '|'.join(str(generation['front_size']) for generation in result['history']),
            ])

    save_pareto_solutions(fronts, equipment, maximum_mass, lab_path)
    save_graphs(histories, results, fronts, lab_path / 'Grafics', equipment, maximum_mass)

    return results


if __name__ == '__main__':
    main()
