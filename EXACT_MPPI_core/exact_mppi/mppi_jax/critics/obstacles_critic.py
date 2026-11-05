import jax
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass
from typing import Tuple

from .critic_data import CriticData
from ..models import ControlConstraints, Rectangles, Polygons
from ..tools.signed_distance import (
    minimum_signed_distance_from_trajs_to_obs_points,
)
import exact_mppi.mppi_jax.tools.signed_distance_polygon as poly_sdf
from ..tools.point_flow_distance import minimum_signed_distance_from_point_flow

"""
Critic objective function for avoiding obstacles, allowing it to deviate off the planned path.
"""


@tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ObstaclesCriticParams:
    enabled: bool
    power: int
    repulsion_weight: float
    critical_weight: float
    collision_cost: float
    collision_margin_distance: float
    near_goal_distance: float
    repulsion_distance: float


def obstacles_critic_initialize(
    critic_params_dict: dict, constraints: ControlConstraints
) -> ObstaclesCriticParams:
    return ObstaclesCriticParams(
        enabled=critic_params_dict.get("enabled", True),
        power=critic_params_dict.get("cost_power", 1),
        repulsion_weight=critic_params_dict.get("repulsion_weight", 1.5),
        critical_weight=critic_params_dict.get("critical_weight", 20.0),
        collision_cost=critic_params_dict.get("collision_cost", 100000.0),
        collision_margin_distance=critic_params_dict.get(
            "collision_margin_distance", 0.1
        ),
        near_goal_distance=critic_params_dict.get("near_goal_distance", 0.5),
        repulsion_distance=critic_params_dict.get("repulsion_distance", 5.0),
    )


def obstacles_critic_score(
    data: CriticData,
    params: ObstaclesCriticParams,
    point_flow_distance_enabled: bool = False,
    collision_substeps: int = 1,
    point_chunk_size: int = 32,
) -> Tuple[jax.Array, bool]:
    """Evaluate cost related to obstacle avoidance"""

    def skip_score(_):
        return jnp.zeros(data.trajectories.x.shape[0]), {"fail_flag": jnp.array(False)}

    def do_score(_):
        near_goal = data.state.local_path_length < params.near_goal_distance

        # The experiment's point-flow path uses the same footprint SDFs, but
        # evaluates them at collision-only internal samples. Other examples
        # retain their exact existing static-point implementation.
        # This static setting is captured by CriticManager. Point-flow-disabled
        # experiment runs still enter this branch with zero point velocities,
        # so static and dynamic points use the same JIT kernel.
        if point_flow_distance_enabled:
            if isinstance(data.footprint, Rectangles):
                dist_to_obj = minimum_signed_distance_from_point_flow(
                    data.trajectories.x, data.trajectories.y, data.trajectories.yaws,
                    data.state.vx, data.state.wz, data.state.pose,
                    data.obs_points, data.obs_velocities, data.obs_points_mask,
                    data.model_dt, collision_substeps, point_chunk_size,
                    rect_centers=data.footprint.centers, rect_halfs=data.footprint.halfs,
                )
            elif isinstance(data.footprint, Polygons):
                dist_to_obj = minimum_signed_distance_from_point_flow(
                    data.trajectories.x, data.trajectories.y, data.trajectories.yaws,
                    data.state.vx, data.state.wz, data.state.pose,
                    data.obs_points, data.obs_velocities, data.obs_points_mask,
                    data.model_dt, collision_substeps, point_chunk_size,
                    vertices=data.footprint.vertices, vertex_counts=data.footprint.vertex_counts,
                )
            else:
                raise TypeError(f"Unknown footprint type: {type(data.footprint)}")
        elif isinstance(data.footprint, Rectangles):
            dist_to_obj = minimum_signed_distance_from_trajs_to_obs_points(
                data.trajectories.x,
                data.trajectories.y,
                data.trajectories.yaws,
                data.obs_points,
                data.obs_points_mask,
                data.footprint.centers,
                data.footprint.halfs,
            )  # (K, T)
        elif isinstance(data.footprint, Polygons):
            dist_to_obj = poly_sdf.minimum_signed_distance_from_trajs_to_obs_points(
                data.trajectories.x,
                data.trajectories.y,
                data.trajectories.yaws,
                data.obs_points,
                data.obs_points_mask,
                vertices=data.footprint.vertices,
                vertex_counts=data.footprint.vertex_counts,
            )  # (K, T)
        else:
            raise TypeError(f"Unknown footprint type: {type(data.footprint)}")

        def step(carry, dist_t):
            trajectory_collide, traj_cost, repulsive_cost = carry  # (K,)
            inCollision = dist_t < 0.0

            active = (~trajectory_collide) & (~inCollision)

            # Let near-collision trajectory points be punished severely
            traj_cost += jnp.where(
                active,
                jax.nn.relu(params.collision_margin_distance - dist_t),
                0.0,
            )  # (K,)

            # Generally prefer trajectories further from obstacles
            repulsive_cost += jnp.where(
                active & (~near_goal),
                jax.nn.relu(params.repulsion_distance - dist_t),
                0.0,
            )  # (K,)

            trajectory_collide = trajectory_collide | inCollision
            return (trajectory_collide, traj_cost, repulsive_cost), None

        K, T = data.trajectories.x.shape
        init_carry = (
            jnp.zeros((K,), dtype=jnp.bool_),
            jnp.zeros((K,)),
            jnp.zeros((K,)),
        )

        (trajectory_collide, traj_cost, repulsive_cost), _ = jax.lax.scan(
            step, init_carry, dist_to_obj.T
        )
        fail_flag = jnp.all(trajectory_collide)

        raw_cost = jnp.where(
            trajectory_collide, params.collision_cost, traj_cost
        )  # (K,)

        # Normalize repulsive cost by trajectory length & lowest score to not overweight importance
        # This is a preferential cost, not collision cost, to be tuned relative to desired behaviors
        repulsive_cost_normalized = (
            repulsive_cost - jnp.min(repulsive_cost)
        ) / T  # (K,)

        cost = (
            params.critical_weight * raw_cost
            + params.repulsion_weight * repulsive_cost_normalized
        )

        if params.power > 1:
            cost = cost**params.power

        return cost, {"fail_flag": fail_flag}

    return jax.lax.cond(
        params.enabled == False,
        skip_score,
        do_score,
        operand=None,
    )
