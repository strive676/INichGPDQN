"""Benchmark scenario definitions for dynamic job-shop experiments."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    name: str
    utilization: float
    failure_rate: float
    due_date_factor: float
    kind: str = "train"     # train/valid/test

@dataclass(frozen=True)
class Scale:
    n_jobs: int
    n_machines: int
    n_warmup: int
    min_ops: int
    max_ops: int
    proc_lo: float = 1.0
    proc_hi: float = 49.0
    repair_lo: float = 30.0
    repair_hi: float = 60.0

SMOKE = Scale(50, 5, 10, 2, 6)
MID = Scale(500, 10, 100, 2, 10)
FULL = Scale(2500, 10, 500, 2, 10)




SCENARIOS = [
    Scenario("T1", 0.50, 0.02, 2.0, "train"),
    Scenario("T2", 0.60, 0.05, 1.3, "train"),
    Scenario("T3", 0.70, 0.05, 2.0, "train"),
    Scenario("T4", 0.80, 0.10, 1.3, "train"),
    Scenario("T5", 0.85, 0.02, 2.0, "train"),
    Scenario("T6", 0.90, 0.10, 1.3, "train"),
    Scenario("T7", 0.95, 0.05, 2.0, "valid"),
    Scenario("T8", 0.95, 0.15, 1.3, "test"),
]
TRAIN_SCENARIOS = [s for s in SCENARIOS if s.kind == "train"]
VALID_SCENARIOS = [s for s in SCENARIOS if s.kind == "valid"]
TEST_SCENARIOS = [s for s in SCENARIOS if s.kind == "test"]
