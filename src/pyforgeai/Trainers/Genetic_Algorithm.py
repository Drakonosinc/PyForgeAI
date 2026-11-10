from __future__ import annotations

import random
from typing import Any, Callable, Sequence
import numpy as np

from ..Environments.Game_Bridge_Env import GameFitnessBridge
from ..Interfaces.IEnvironment import IFitnessEnvironment
from ..Interfaces.ITrainer import ITrainer, ITrainingCallback, TrainingResult
from ..Models.Model_Builder import (
    build_model,
    ensure_built_if_tf,
    get_weights_np,
    set_weights_np,
)
from ..Serialization.Model_Serializer import load_genetic_model, save_genetic_model


# Aliases kept for backward compatibility with existing unit tests & game imports
_get_weights_np = get_weights_np
_set_weights_np = set_weights_np


def _new_model(
    type_model: str,
    input_size: int,
    output_size: int,
    hidden_sizes: Sequence[int] | None = None,
    output_activation: str | None = None,
) -> Any:
    resolved_out_act = output_activation if output_activation is not None else ("tanh" if int(output_size) == 1 else None)
    model = build_model(
        type_model,
        input_size,
        output_size,
        hidden_sizes=hidden_sizes,
        output_activation=resolved_out_act,
    )
    ensure_built_if_tf(model, input_size)
    return model


def fitness_function(model: Any, target: Any) -> float | list[float]:
    """Evaluate a single model (or a population list on batch games) against a fitness environment, callable, or game object."""
    if isinstance(target, IFitnessEnvironment):
        if isinstance(target, GameFitnessBridge) and isinstance(model, (list, tuple)) and target.is_batch_population_target():
            return target.evaluate_population_batch(model)
        return float(target.evaluate_model(model))
    bridge = GameFitnessBridge(target)
    if isinstance(model, (list, tuple)) and bridge.is_batch_population_target():
        return bridge.evaluate_population_batch(model)
    return bridge.evaluate_model(model)


def initialize_population(
    type_model: str,
    size: int,
    input_size: int,
    output_size: int,
    hidden_sizes: Sequence[int] | None = None,
    output_activation: str | None = None,
) -> list[Any]:
    """Create an initial population of neural networks with random weights."""
    return [
        _new_model(
            type_model,
            input_size,
            output_size,
            hidden_sizes=hidden_sizes,
            output_activation=output_activation,
        )
        for _ in range(int(size))
    ]


def evaluate_population(
    population: Sequence[Any],
    target: Any,
    num_trials: int = 3,
) -> tuple[list[Any], list[float]]:
    """Evaluate every individual in `population` over `num_trials` episodes and return positive shifted fitness scores."""
    bridge = target if isinstance(target, GameFitnessBridge) else GameFitnessBridge(target)
    evaluated_population: list[Any] = []
    raw_scores: list[float] = []

    if bridge.is_batch_population_target():
        pop_list = list(population)
        if not pop_list or bridge.should_stop():
            return [], []
        accumulated: list[float] = []
        trials_completed = 0
        for _ in range(max(1, int(num_trials))):
            if bridge.should_stop():
                break
            batch_scores = bridge.evaluate_population_batch(pop_list)
            if not batch_scores:
                break
            if not accumulated:
                accumulated = [float(s) for s in batch_scores]
            else:
                accumulated = [acc + float(s) for acc, s in zip(accumulated, batch_scores)]
            trials_completed += 1
            if bridge.should_stop():
                break
        if not accumulated or trials_completed == 0:
            return [], []
        raw_scores = [acc / trials_completed for acc in accumulated]
        evaluated_population = pop_list[: len(raw_scores)]
    else:
        for model in population:
            if bridge.should_stop():
                break
            trial_scores: list[float] = []
            for _ in range(max(1, int(num_trials))):
                trial_scores.append(bridge.evaluate_model(model))
                if bridge.should_stop():
                    break
            if not trial_scores:
                break
            evaluated_population.append(model)
            raw_scores.append(sum(trial_scores) / len(trial_scores))
            if bridge.should_stop():
                break

    if not raw_scores:
        return [], []

    min_score = abs(min(raw_scores)) if min(raw_scores) < 0 else 0.0
    shifted_scores = [score + min_score + 1.0 for score in raw_scores]
    return evaluated_population, shifted_scores


def select_parents(
    population: Sequence[Any],
    fitness_scores: Sequence[float],
    num_parents: int,
) -> list[Any]:
    """Select parents from the top 50% of the population using fitness-proportionate roulette selection."""
    sorted_pop_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    top_count = max(2, len(sorted_pop_fitness) // 2)
    top_population = [ind for ind, _ in sorted_pop_fitness[:top_count]]
    top_fitness = [max(1e-6, float(fit)) for _, fit in sorted_pop_fitness[:top_count]]
    return random.choices(top_population, weights=top_fitness, k=int(num_parents))


def crossover(
    parent1: Any,
    parent2: Any,
    type_model: str,
    input_size: int,
    output_size: int,
    hidden_sizes: Sequence[int] | None = None,
) -> tuple[Any, Any]:
    """Uniform parameter-tensor crossover producing two offspring models."""
    w1 = get_weights_np(parent1)
    w2 = get_weights_np(parent2)
    child1_w: list[np.ndarray] = []
    child2_w: list[np.ndarray] = []
    for a, b in zip(w1, w2):
        mask = np.random.rand(*a.shape) > 0.5
        child1_w.append(np.where(mask, a, b))
        child2_w.append(np.where(mask, b, a))

    child1 = _new_model(type_model, input_size, output_size, hidden_sizes)
    child2 = _new_model(type_model, input_size, output_size, hidden_sizes)
    set_weights_np(child1, child1_w)
    set_weights_np(child2, child2_w)
    return child1, child2


def mutate(
    model: Any,
    mutation_rate: float = 0.01,
    mutation_strength: float = 0.1,
    strong_mutation_rate: float = 0.0,
    strong_mutation_strength: float = 0.5,
) -> Any:
    """Apply fine Gaussian mutation and optional strong mutation impulses to model weights."""
    weights = get_weights_np(model)
    mutated: list[np.ndarray] = []
    for w in weights:
        mask = np.random.rand(*w.shape) < float(mutation_rate)
        noise = np.random.randn(*w.shape) * float(mutation_strength)
        updated = w + noise * mask
        if strong_mutation_rate > 0.0:
            strong_mask = np.random.rand(*w.shape) < float(strong_mutation_rate)
            strong_noise = np.clip(
                np.random.randn(*w.shape) * float(strong_mutation_strength),
                -1.0,
                1.0,
            )
            updated = updated + strong_noise * strong_mask
        mutated.append(updated.astype(w.dtype, copy=False))
    set_weights_np(model, mutated)
    return model


class GeneticTrainer(ITrainer):
    """Modular Neuroevolution trainer combining multi-backend support, elitism, adaptive mutation, and diversity injection."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        type_model: str = "Pytorch",
        hidden_sizes: Sequence[int] | None = None,
        generations: int = 100,
        population_size: int = 20,
        mutation_rate: float = 0.01,
        mutation_strength: float = 0.1,
        strong_mutation_rate: float = 0.0,
        elitism: int = 2,
        num_trials: int = 3,
        adaptive_mutation: bool = True,
        diversity_interval: int = 0,
        diversity_ratio: float = 0.2,
        save_path: str | None = None,
        callbacks: Sequence[ITrainingCallback] | None = None,
        verbose: bool = True,
    ) -> None:
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.type_model = type_model
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes is not None else [128]
        self.generations = int(generations)
        self.population_size = max(2, int(population_size))
        self.initial_mutation_rate = float(mutation_rate)
        self.mutation_rate = float(mutation_rate)
        self.mutation_strength = float(mutation_strength)
        self.strong_mutation_rate = float(strong_mutation_rate)
        self.elitism = min(max(1, int(elitism)), self.population_size)
        self.num_trials = max(1, int(num_trials))
        self.adaptive_mutation = bool(adaptive_mutation)
        self.diversity_interval = int(diversity_interval)
        self.diversity_ratio = float(diversity_ratio)
        self.save_path = save_path
        self.callbacks = list(callbacks) if callbacks else []
        self.verbose = bool(verbose)

        self.best_model: Any = None
        self.best_fitness: float = float("-inf")
        self.last_result: TrainingResult | None = None

    def train_with_result(self, target: Any) -> TrainingResult:
        """Run neuroevolution on `target` (IFitnessEnvironment, callable `fn(model)->float`, or game instance)."""
        bridge = GameFitnessBridge(target)
        population = initialize_population(
            self.type_model,
            self.population_size,
            self.input_size,
            self.output_size,
            hidden_sizes=self.hidden_sizes,
        )
        self.best_fitness = float("-inf")
        self.best_model = None
        self.mutation_rate = self.initial_mutation_rate
        history: list[float] = []
        stopped_early = False
        completed_gens = 0

        for cb in self.callbacks:
            cb.on_training_start(self)

        for generation in range(self.generations):
            if bridge.should_stop():
                stopped_early = True
                break
            bridge.set_generation(generation)

            evaluated_population, fitness_scores = evaluate_population(
                population,
                bridge,
                num_trials=self.num_trials,
            )
            if not fitness_scores:
                stopped_early = True
                break

            completed_gens = generation + 1
            current_best = float(max(fitness_scores))
            mean_fitness = float(sum(fitness_scores) / len(fitness_scores))
            history.append(current_best)

            if current_best > self.best_fitness:
                self.best_fitness = current_best
                best_idx = fitness_scores.index(current_best)
                best_parent = evaluated_population[best_idx]
                self.best_model = _new_model(
                    self.type_model,
                    self.input_size,
                    self.output_size,
                    self.hidden_sizes,
                )
                set_weights_np(self.best_model, get_weights_np(best_parent))
                if self.adaptive_mutation:
                    self.mutation_rate = self.initial_mutation_rate
            elif self.adaptive_mutation:
                self.mutation_rate = min(self.mutation_rate * 1.35, 0.15)

            if self.verbose:
                print(f"Generation {generation}: Best Fitness = {current_best:.4f} (Mean = {mean_fitness:.4f})")

            # Trigger callbacks
            should_halt = False
            for cb in self.callbacks:
                if cb.on_iteration_end(generation, self.best_fitness, mean_fitness, self):
                    should_halt = True
            if should_halt or bridge.should_stop():
                stopped_early = should_halt or bridge.should_stop()
                break

            # Elitism
            sorted_population = [
                model
                for _, model in sorted(
                    zip(fitness_scores, evaluated_population),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ]
            new_population: list[Any] = []
            for i in range(min(self.elitism, len(sorted_population))):
                elite_copy = _new_model(
                    self.type_model,
                    self.input_size,
                    self.output_size,
                    self.hidden_sizes,
                )
                set_weights_np(elite_copy, get_weights_np(sorted_population[i]))
                new_population.append(elite_copy)

            num_offsprings = self.population_size - len(new_population)
            parents = select_parents(evaluated_population, fitness_scores, max(2, num_offsprings))
            offspring: list[Any] = []
            for i in range(0, len(parents) - 1, 2):
                c1, c2 = crossover(
                    parents[i],
                    parents[i + 1],
                    self.type_model,
                    self.input_size,
                    self.output_size,
                    self.hidden_sizes,
                )
                offspring.append(
                    mutate(
                        c1,
                        self.mutation_rate,
                        self.mutation_strength,
                        strong_mutation_rate=self.strong_mutation_rate,
                    )
                )
                offspring.append(
                    mutate(
                        c2,
                        self.mutation_rate,
                        self.mutation_strength,
                        strong_mutation_rate=self.strong_mutation_rate,
                    )
                )

            while len(offspring) < num_offsprings and not bridge.should_stop():
                extra_parent = random.choice(population)
                extra_child = _new_model(
                    self.type_model,
                    self.input_size,
                    self.output_size,
                    self.hidden_sizes,
                )
                set_weights_np(extra_child, get_weights_np(extra_parent))
                offspring.append(
                    mutate(
                        extra_child,
                        self.mutation_rate,
                        self.mutation_strength,
                        strong_mutation_rate=self.strong_mutation_rate,
                    )
                )

            if bridge.should_stop():
                stopped_early = True
                break

            next_population = new_population + offspring[:num_offsprings]

            # Periodic diversity injection (from Snake_Game)
            if (
                self.diversity_interval > 0
                and generation > 0
                and generation % self.diversity_interval == 0
            ):
                num_random = max(1, int(self.population_size * self.diversity_ratio))
                num_keep = max(self.elitism, self.population_size - num_random)
                random_models = initialize_population(
                    self.type_model,
                    self.population_size - num_keep,
                    self.input_size,
                    self.output_size,
                    hidden_sizes=self.hidden_sizes,
                )
                next_population = next_population[:num_keep] + random_models

            population = next_population[: self.population_size]

        if self.best_model is None and population:
            self.best_model = population[0]

        bridge.assign_best_model(self.best_model)

        if self.save_path and self.best_model is not None:
            save_genetic_model(self.best_model, optimizer=None, path=self.save_path)

        result = TrainingResult(
            best_model=self.best_model,
            best_score=self.best_fitness,
            history=history,
            iterations_completed=completed_gens,
            stopped_early=stopped_early,
        )
        self.last_result = result
        for cb in self.callbacks:
            cb.on_training_end(result, self)
        return result

    def train(self, target: Any = None) -> Any:
        return self.train_with_result(target).best_model


def genetic_algorithm(
    game: Any,
    type_model: str = "Pytorch",
    input_size: int = 6,
    output_size: int = 2,
    generations: int = 100,
    population_size: int = 20,
    mutation_rate: float = 0.01,
    mutation_strength: float = 0.1,
    elitism: int = 2,
    num_trials: int = 3,
    hidden_sizes: Sequence[int] | None = None,
    adaptive_mutation: bool = False,
    diversity_interval: int = 0,
    verbose: bool = True,
) -> Any:
    """Functional neuroevolution entry point preserving 100% compatibility with Pong_Game / Snake_Game."""
    trainer = GeneticTrainer(
        input_size=input_size,
        output_size=output_size,
        type_model=type_model,
        hidden_sizes=hidden_sizes,
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        elitism=elitism,
        num_trials=num_trials,
        adaptive_mutation=adaptive_mutation,
        diversity_interval=diversity_interval,
        verbose=verbose,
    )
    return trainer.train(game)
