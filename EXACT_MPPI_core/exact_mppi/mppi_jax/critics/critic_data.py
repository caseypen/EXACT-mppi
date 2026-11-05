import jax
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass

from ..models import State, Path, Trajectories, Rectangles, Polygons
from ..motion_models import MotionModelParams


from typing import Any, Union

@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CriticData:
    """
    Data to pass to critics for scoring, including state, trajectories,
    pruned path, global goal, costs, and important parameters to share
    """

    state: State
    trajectories: Trajectories
    path: Path
    goal: jax.Array  # (3,) -> [x, y, yaw]

    model_dt: float

    motion_model: MotionModelParams
    furthest_reached_path_point: int
    path_pts_valid: jax.Array

    # Obstacle Points
    obs_points: jax.Array
    obs_velocities: jax.Array
    obs_points_mask: jax.Array

    # Robot Footprint (Rectangles or Polygons)
    footprint: Union[Rectangles, Polygons]
