from pydantic import BaseModel

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


class TrainerConfig(BaseModel):
    epochs: int
    patience: int


class TrainConfig(BaseModel):
    batch_sz: BatchSizeConfig
    optim: OptimConfig
    trainer: TrainerConfig
    save_dir: str


class Config(BaseModel):
    data: DataConfig
    model_name: str
    train: TrainConfig