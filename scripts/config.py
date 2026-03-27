from typing import Any, Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    root_dir: str
    train_sz: float
    seed: int


class BatchSizeConfig(BaseModel):
    train: int
    valid: int


class OptimConfig(BaseModel):
    lr: float
    weight_decay: float
    lr_backbone: float | None = None


class TrainerConfig(BaseModel):
    epochs: int
    patience: int
    max_grad_norm: float = 1.0


class TrainConfig(BaseModel):
    batch_sz: BatchSizeConfig
    optim: OptimConfig
    trainer: TrainerConfig
    save_dir: str
    num_workers: int = 0


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "deeppcb-detr"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "offline"
    name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    dir: str | None = None


class EvalConfig(BaseModel):
    score_threshold: float = 0.5
    iou_match_threshold: float = 0.5
    class_metrics: bool = True


class Config(BaseModel):
    experiment_name: str | None = None
    data: DataConfig
    model_name: str
    train: TrainConfig
    eval: EvalConfig = EvalConfig()
    wandb: WandbConfig | None = None


class SuiteExperiment(BaseModel):
    name: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class SuiteConfig(BaseModel):
    suite_name: str | None = None
    base_config: dict[str, Any]
    experiments: list[SuiteExperiment]
