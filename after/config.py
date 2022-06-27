from dataclasses import dataclass


@dataclass
class Paths:
    """
    Args:
        log: Tensorboard log directory
        data: Input data directory
    """
    log: str
    data: str


@dataclass
class Files:
    train_data: str
    train_labels: str
    test_data: str
    test_labels: str


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int
