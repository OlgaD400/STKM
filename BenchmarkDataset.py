"""Create a benchmark dataset of a group of objects moving through space and time."""

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import pandas as pd
from typing import List, Type
from dataclasses import dataclass

@dataclass
class PyswarmParameters:
    """
    Store parameters required for PySwarm simulation and the creation of the final swarm.

    Args:
        cognitive_param (float): Cognitive parameter controlling how strongly each point follows its own personal best position at each time step.
        social_param (float): Social parameter controlling how strongly each point follows the swarm's global best position at each time step.
        inertia (float): Parameter controlling the intertia of the swarm's movement.  Inertia<1 reduces the accelration of the swarm over time.  Inertia>1 implies that the swarm velocity increases over time towards the maximum velocity.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations Particle Swarm Optimization runs for.
        trajectory_vector (List[float]): Trajectory of the swarm.  The swarm moves parallel to this vector.
        start_point (List[float]): The point which the trajectory of the swarm passes through.
    """

    def __init__(
        self,
        cognitive_param: float,
        social_param: float,
        inertia: float,
        n_particles: int,
        iters: int,
        trajectory_vector: List[float],
        start_point: List[float],
    ):

        self.cognitive_param = cognitive_param
        self.social_param = social_param
        self.inertia = inertia
        self.n_particles = n_particles
        self.iters = iters
        self.trajectory_vector = trajectory_vector
        self.start_point = start_point


class BenchmarkDataset:
    """
    Create a new benchmark dataset class for temporal clustering.

    Args:
        params_list (Union[None, List[PyswarmParameters]]): List of PyswarmParameters data classes, with each data class defining the parameters required for the creation of a swarm.

    Attributes:
        swarm_list (List[np.ndarray]): Each entry contains a list of the positional histories of each swarm.
        swarm_df (pd.DataFrame): Dataframe containing columns 'entity_id', 'lat', 'lon', 'iteration', 'swarm_id', 'time_ts'.
    """

    def __init__(
        self, params_list: List[Type[PyswarmParameters]],
    ):

        self.params_list = params_list

        # Parameters set in functions
        self.swarm_list = None
        self.swarm_df = None

    def create_moving_swarms(self) -> None:
        """Create list of swarms based on parameters given in params_list.  Each swarm will be a three dimensional np.ndarray."""
        self.swarm_list = []

        for params in self.params_list:
            self.swarm_list.append(
                create_swarm(
                    cognitive_param=params.cognitive_param,
                    social_param=params.social_param,
                    inertia=params.inertia,
                    n_particles=params.n_particles,
                    iters=params.iters,
                    trajectory_vector=params.trajectory_vector,
                    start_point=params.start_point,
                )
            )

    def create_swarm_df(self) -> None:
        """Create a dataframe containing columns 'entity_id', 'lat', 'lon', 'iteration', 'swarm_id', 'time_ts' for a list of swarms."""
        swarm_ids = np.arange(len(self.swarm_list))

        self.swarm_df = pd.DataFrame(
            columns=[
                'entity_id',
                'lat',
                'lon',
                "iteration",
                "swarm_id",
            ]
        )

        old_len = 0
        for swarm_id in swarm_ids:

            swarm_coords = self.swarm_list[swarm_id]
            num_points = np.array(swarm_coords).shape[1]

            for idx, coord in enumerate(swarm_coords):
                df = pd.DataFrame(
                    np.concatenate(
                        (
                            np.arange(old_len, old_len + num_points)[..., np.newaxis],
                            coord[:, 0][..., np.newaxis],
                            coord[:, 1][..., np.newaxis],
                            np.repeat(idx, num_points)[..., np.newaxis],
                            np.repeat(swarm_id, num_points)[..., np.newaxis],
                        ),
                        axis=1,
                    ),
                    columns=[
                        'entity_id',
                        'lat',
                        'lon',
                        "iteration",
                        "swarm_id",
                    ],
                )
                self.swarm_df = self.swarm_df.append(df)

            old_len += num_points

        self.swarm_df['time_ts'] = pd.to_datetime(
            self.swarm_df["iteration"], unit="m"
        ).dt.strftime("%H:%M")

        self.swarm_df.entity_id = self.swarm_df.entity_id.astype(int)
        self.swarm_df.iteration = self.swarm_df.iteration.astype(int)
        self.swarm_df.swarm_id = self.swarm_df.swarm_id.astype(int)

        self.swarm_df.sort_values(by=["time_ts", "entity_id"], inplace=True)
        self.swarm_df.reset_index(inplace=True)

    def return_labels(self) -> List[int]:
        """
        Return the long term cluster labels of the swarms, with each point belonging to a different swarm.

        Returns:
            labels (List[int]): labels associated with simulation long-term clusters.
        """
        labels = []

        for j, swarm in enumerate(self.swarm_list):
            labels.append(np.repeat(j, np.array(swarm).shape[1]))

        true_labels = [item for sublist in labels for item in sublist]

        return true_labels


def create_swarm(
    cognitive_param: float,
    social_param: float,
    inertia: float,
    n_particles: int,
    iters: int,
    trajectory_vector: List[float],
    start_point: List[float],
) -> np.ndarray:
    """
    Create swarm using PySwarm and give it a trajectory.  PySwarm uses Particle Swarm Optimization (PSO) to minimize some objective function. We use the sphere objective function. Every point in the swarm is considered a potential solution to the objective function, and points iteratively update their positions and velocities based on the locations of their own best previous cost as well as the best previous cost of their neighbors. The updates are shown below.

    The parameters v_{ij} and x_{ij} denote the velocity and position of the ith particle in the jth dimension, respectively. The parameters c_1 and c_2 are the cognitive and social parameters, which control how strongly each point follows its own personal best, y_{ij}(t), or the swarm’s global best, hat{y}_{ij}(t), position in the t'th time step.  The personal and global best positions are calculated every time step using the cost function.  The parameters r_1 and r_2 are independent and identically distributed random numbers.  The weight parameter omega controls the inertia of the swarm’s movement.

    v_{ij}(t+1) =omega v_{ij}(t) + c_1 r_{1}(t)[y_{ij}(t) − x_{ij}(t)]+ c_2 r_2(t)[hat{y}_{ij}(t) − x_{ij}(t)]

    x_{ij}(t+1) = x_{ij}(t) + v_{ij}(t+1)

    After PSO generates swarms, we give each swarm a trajectory passing through start point r_0 and parallel to the trajectory vector z.  The final positions of particles in the swarm is given below.

    p_i(t) = x_i(t) + r_0 + tz

    Args:
        cognitive_param (float): Cognitive parameter controlling how strongly each point follows its own personal best position at each time step.
        social_param (float): Social parameter controlling how strongly each point follows the swarm's global best position at each time step.
        inertia (float): Parameter controlling the intertia of the swarm's movement.  Inertia<1 reduces the accelration of the swarm over time.  Inertia>1 implies that the swarm velocity increases over time towards the maximum velocity.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations Particle Swarm Optimization runs for.
        trajectory_vector (List[float]): Trajectory of the swarm.  The swarm moves parallel to this vector.
        start_point (List[float]): The point which the trajectory of the swarm passes through.

    Returns:
        swarm (np.ndarray): Three dimensional array of size (iters, n_particles, 2), where 2 is the number of dimensions.
    """
    options = {"c1": cognitive_param, "c2": social_param, "w": inertia}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=2, options=options
    )

    optimizer.optimize(fx.sphere, iters=iters)

    swarm = optimizer.pos_history

    time = np.linspace(0, 1, len(optimizer.pos_history))
    x_parametric = start_point[0] + trajectory_vector[0] * time
    y_parametric = start_point[1] + trajectory_vector[1] * time

    for i in range(len(optimizer.pos_history)):
        swarm[i][:, 0] = swarm[i][:, 0] + x_parametric[i]
        swarm[i][:, 1] = swarm[i][:, 1] + y_parametric[i]

    return swarm


def create_switching_swarms(
    cognitive_param: List[float],
    social_param: List[float],
    inertia: List[float],
    n_particles: List[int],
    iters: int
) -> List[np.ndarray]:
    """
    Create 3 swarms using PySwarm, one traveling along pi/2, one traveling along -pi/2, and one traveling
    along tan(x).

    Args:
        cognitive_param (List[float]): Cognitive parameters controlling how strongly each point follows its own personal best position at each time step.
        social_param (List[float]): Social parameter controlling how strongly each point follows the swarm's global best position at each time step.
        inertia (List[float]): Parameter controlling the intertia of the swarm's movement.  Inertia<1 reduces the accelration of the swarm over time.  Inertia>1 implies that the swarm velocity increases over time towards the maximum velocity.
        n_particles (List[int]): Number of particles in the swarm.
        iters (int): Number of iterations Particle Swarm Optimization runs for.
        trajectory_vector (List[float]): Trajectory of the swarm.  The swarm moves parallel to this vector.
        start_point (List[float]): The point which the trajectory of the swarm passes through.

    Returns:
        swarm (np.ndarray): Three dimensional array of size (iters, n_particles, 2), where 2 is the number of dimensions.
    """
    n_swarms = len(cognitive_param)
    swarms = []

    for i in range(n_swarms):
        options = {"c1": cognitive_param[i], "c2": social_param[i], "w": inertia[i]}

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles[i], dimensions=2, options=options
        )

        optimizer.optimize(fx.sphere, iters=iters)

        swarm = optimizer.pos_history

        swarms.append(swarm)

    time = np.linspace(0, 1, iters)
    tan_time = np.linspace(-np.pi/2+.2, np.pi/2-.2, iters)

    x_parametric = [[np.pi/2]*iters, [-np.pi/2]*iters, tan_time]
    y_parametric = [1+time, -2-time, np.tan(tan_time)]

    for index, single_swarm in enumerate(swarms):
        for i in range(iters):
            single_swarm[i][:, 0] = single_swarm[i][:, 0] + x_parametric[index][i]
            single_swarm[i][:, 1] = single_swarm[i][:, 1] + y_parametric[index][i]

    return swarms
