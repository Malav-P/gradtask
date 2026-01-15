from dataclasses import dataclass, field
import numpy as np

CR3BP_MU = 1.215058560962404e-02
CR3BP_LU = 384400
CR3BP_TU = 3.751902619517228e+05




@dataclass
class Config:
    exp_name: str = None
    period: float = None
    n_points: int = None
    max_iters: int = None

    initial_state_target: np.ndarray = field(default=None)
    initial_state_target_phases: np.ndarray = field(default=None)
    initial_state_observer: np.ndarray = field(default=None)
    initial_state_observer_phases: np.ndarray = field(default=None)

    t_expose: float = None
    sigma: float = None
    assignment_penalty_lambda: float = None

    mu: float = CR3BP_MU
    TU: float = CR3BP_TU
    LU: float = CR3BP_LU

