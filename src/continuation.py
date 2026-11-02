import numpy as np
from src.cr3bp_model import CR3BPModel
from src.differential_correction import differential_correction
from src.numerical import NumericalSettings
from src.data import Integrator


def secant_predictor(state_1, state_2, scale=1e-2):
    """
    Secant predictor estimating the next state from last two solutions.
    """

    # tangent direction
    direction = state_1 - state_2
    direction /= np.linalg.norm(direction)

    # scale step by orbit amplitude (component-wise)
    step = scale * direction

    # create initial guess
    initial_guess = state_1.copy()
    initial_guess += step

    # call differential correction
    return initial_guess


def continuation_step(system: CR3BPModel,
                      integrator: Integrator,
                      numerical: NumericalSettings,
                      trajectories, time_step,
                      scale=1e-2):

    last_state = trajectories[-1][:4].copy()
    second_last_state = trajectories[-2][:4].copy()
    first_guess = secant_predictor(last_state, second_last_state, scale)

    return differential_correction(system=system,
                                   integrator=integrator,
                                   numerical=numerical,
                                   x=first_guess,
                                   dt=time_step,
                                   tol=1e-12)
