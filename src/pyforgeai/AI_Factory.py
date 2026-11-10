from __future__ import annotations

from typing import Any

from .Adapters import MockAdapter, PyTorchAdapter, TensorFlowAdapter, wrap_model
from .AI_Controller import AIController
from .Environments import CallbackRLEnvironment, GameFitnessBridge, TargetSeekEnvironment
from .Interfaces import IAIModel, ITrainer
from .Models import build_model
from .Serialization import AILoader, load_model, save_model
from .Trainers import (
    CheckpointCallback,
    DQNAgent,
    EarlyStoppingCallback,
    GeneticTrainer,
    HistoryCallback,
    QLearningTrainer,
)


class AIFactory:
    """Centralized factory for creating models, adapters, trainers, and controllers (mirrors PyForgeUI's ElementsFactory)."""

    def __init__(self, defaults: dict[str, Any] | None = None) -> None:
        self.defaults = dict(defaults) if defaults else {}

    def _merge(self, config: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self.defaults)
        if config:
            merged.update(config)
        return merged

    def create_Model(self, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        cfg = self._merge(config)
        cfg.update(kwargs)
        return build_model(
            type_model=cfg.get("type_model", "Pytorch"),
            input_size=int(cfg.get("input_size", 6)),
            output_size=int(cfg.get("output_size", 2)),
            hidden_sizes=cfg.get("hidden_sizes", [128]),
            activation=cfg.get("activation", "relu"),
            record_activations=bool(cfg.get("record_activations", True)),
        )

    def create_Adapter(self, model: Any = None, config: dict[str, Any] | None = None, **kwargs: Any) -> IAIModel:
        cfg = self._merge(config)
        cfg.update(kwargs)
        if model is None and "model" in cfg:
            model = cfg["model"]
        if model is None:
            return MockAdapter(
                output_size=int(cfg.get("output_size", 2)),
                fixed_output=cfg.get("fixed_output"),
            )
        return wrap_model(model, output_size=int(cfg.get("output_size", 2)))

    def create_GeneticTrainer(self, config: dict[str, Any] | None = None, **kwargs: Any) -> GeneticTrainer:
        cfg = self._merge(config)
        cfg.update(kwargs)
        return GeneticTrainer(
            input_size=int(cfg.get("input_size", 6)),
            output_size=int(cfg.get("output_size", 2)),
            type_model=cfg.get("type_model", "Pytorch"),
            hidden_sizes=cfg.get("hidden_sizes", [128]),
            generations=int(cfg.get("generations", cfg.get("generation_value", 100))),
            population_size=int(cfg.get("population_size", cfg.get("population_value", 20))),
            mutation_rate=float(cfg.get("mutation_rate", 0.01)),
            mutation_strength=float(cfg.get("mutation_strength", 0.1)),
            strong_mutation_rate=float(cfg.get("strong_mutation_rate", 0.0)),
            elitism=int(cfg.get("elitism", 2)),
            num_trials=int(cfg.get("num_trials", cfg.get("try_for_ai", 3))),
            adaptive_mutation=bool(cfg.get("adaptive_mutation", True)),
            diversity_interval=int(cfg.get("diversity_interval", 0)),
            diversity_ratio=float(cfg.get("diversity_ratio", 0.2)),
            save_path=cfg.get("save_path"),
            callbacks=cfg.get("callbacks"),
            verbose=bool(cfg.get("verbose", True)),
        )

    def create_QLearningTrainer(self, config: dict[str, Any] | None = None, **kwargs: Any) -> QLearningTrainer:
        cfg = self._merge(config)
        cfg.update(kwargs)
        return QLearningTrainer(
            game=cfg.get("game"),
            type_model=cfg.get("type_model", "Pytorch"),
            input_size=int(cfg.get("input_size", 6)),
            output_size=int(cfg.get("output_size", 2)),
            episodes=int(cfg.get("episodes", 500)),
            lr=float(cfg.get("lr", cfg.get("learning_rate", 1e-3))),
            gamma=float(cfg.get("gamma", 0.99)),
            epsilon_start=float(cfg.get("epsilon_start", 1.0)),
            epsilon_end=float(cfg.get("epsilon_end", 0.01)),
            epsilon_decay=float(cfg.get("epsilon_decay", 0.995)),
            memory_size=int(cfg.get("memory_size", 10000)),
            batch_size=int(cfg.get("batch_size", 32)),
            target_update=int(cfg.get("target_update", 100)),
            hidden_sizes=cfg.get("hidden_sizes", [128]),
            double_dqn=bool(cfg.get("double_dqn", True)),
            save_path=cfg.get("save_path"),
            callbacks=cfg.get("callbacks"),
            verbose=bool(cfg.get("verbose", True)),
        )

    def create_DQNAgent(self, config: dict[str, Any] | None = None, **kwargs: Any) -> DQNAgent:
        cfg = self._merge(config)
        cfg.update(kwargs)
        return DQNAgent(
            type_model=cfg.get("type_model", "Pytorch"),
            state_size=int(cfg.get("state_size", cfg.get("input_size", 6))),
            action_size=int(cfg.get("action_size", cfg.get("output_size", 2))),
            lr=float(cfg.get("lr", cfg.get("learning_rate", 1e-3))),
            gamma=float(cfg.get("gamma", 0.99)),
            epsilon_start=float(cfg.get("epsilon_start", 1.0)),
            epsilon_end=float(cfg.get("epsilon_end", 0.01)),
            epsilon_decay=float(cfg.get("epsilon_decay", 0.995)),
            memory_size=int(cfg.get("memory_size", 10000)),
            batch_size=int(cfg.get("batch_size", 32)),
            target_update=int(cfg.get("target_update", 100)),
            hidden_sizes=cfg.get("hidden_sizes", [128]),
            double_dqn=bool(cfg.get("double_dqn", True)),
        )

    def create_Controller(self, config: dict[str, Any] | None = None, **kwargs: Any) -> AIController:
        cfg = self._merge(config)
        cfg.update(kwargs)
        return AIController(
            state_provider=cfg.get("state_provider"),
            action_dispatcher=cfg.get("action_dispatcher"),
            default_backend=cfg.get("type_model", "Pytorch"),
            output_size=int(cfg.get("output_size", 2)),
        )

    def create_Trainer_from_game_config(
        self,
        config_ai: dict[str, Any],
        input_size: int,
        output_size: int,
        save_path: str | None = None,
        verbose: bool = True,
    ) -> ITrainer:
        """Build a GeneticTrainer or QLearningTrainer directly from a Pong/Snake `config_AI` dictionary."""
        nn_cfg = config_ai.get("nn", {"hidden_layers": 2, "neurons_per_layer": 6})
        arch = [int(nn_cfg.get("neurons_per_layer", 6))] * int(nn_cfg.get("hidden_layers", 2))
        type_model = next((k for k, v in config_ai.get("type_model", {"Pytorch": True}).items() if v), "Pytorch")
        training_type = next((k for k, v in config_ai.get("type_training", {"Genetic": True}).items() if v), "Genetic")
        effective_save = save_path if config_ai.get("model_save", False) else None

        if training_type.lower().startswith("genetic"):
            gen_cfg = config_ai.get("genetic", {})
            return self.create_GeneticTrainer(
                input_size=input_size,
                output_size=output_size,
                type_model=type_model,
                hidden_sizes=arch,
                generations=gen_cfg.get("generation_value", 100),
                population_size=gen_cfg.get("population_value", 20),
                num_trials=gen_cfg.get("try_for_ai", 3),
                save_path=effective_save,
                verbose=verbose,
            )

        q_cfg = config_ai.get("q_learning", {})
        return self.create_QLearningTrainer(
            input_size=input_size,
            output_size=output_size,
            type_model=type_model,
            hidden_sizes=arch,
            episodes=q_cfg.get("episodes", 500),
            lr=q_cfg.get("learning_rate", 1e-3),
            gamma=q_cfg.get("gamma", 0.99),
            epsilon_start=q_cfg.get("epsilon_start", 1.0),
            epsilon_end=q_cfg.get("epsilon_end", 0.01),
            epsilon_decay=q_cfg.get("epsilon_decay", 0.995),
            save_path=effective_save,
            verbose=verbose,
        )
