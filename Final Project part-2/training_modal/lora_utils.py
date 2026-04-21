from dataclasses import dataclass


@dataclass
class LoraConfigLite:
    r: int = 16
    alpha: int = 64
    dropout: float = 0.05
    target_modules: str = "all-linear"
