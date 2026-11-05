from typing import Tuple
import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial

from .critics import *
from .models import ControlConstraints, Rectangles, Polygons
from .tools.utils import findPathFurthestReachedPoint
from .tools.signed_distance import (
    minimum_signed_distance_from_traj_to_obs_points,
)
import exact_mppi.mppi_jax.tools.signed_distance_polygon as poly_sdf


class CriticManager:
    """
    Manager of objective function plugins for scoring trajectories
    """

    def __init__(
        self, params: dict, constraints: ControlConstraints, debug: bool = False
    ):
        self.critics_ = {}
        self.debug_ = debug
        path_follow_params = params.get("PathFollowCritic", {})
        self.furthest_path_lookahead_distance_ = float(
            path_follow_params.get("furthest_path_lookahead_distance", 5.0)
        )
        if self.furthest_path_lookahead_distance_ <= 0.0:
            raise ValueError("furthest_path_lookahead_distance must be positive")

        # Constraint Critic
        self.critics_["ConstraintCritic"] = {
            "initialize": constraint_critic_initialize,
            "score": constraint_critic_score,
        }

        # Goal Critic
        self.critics_["GoalCritic"] = {
            "initialize": goal_critic_initialize,
            "score": goal_critic_score,
        }

        # Goal Angle Critic
        self.critics_["GoalAngleCritic"] = {
            "initialize": goal_angle_critic_initialize,
            "score": goal_angle_critic_score,
        }

        # Obstacles Critic
        self.critics_["ObstaclesCritic"] = {
            "initialize": obstacles_critic_initialize,
            "score": obstacles_critic_score,
        }

        # Path Align Critic
        self.critics_["PathAlignCritic"] = {
            "initialize": path_align_critic_initialize,
            "score": path_align_critic_score,
        }

        # Path Follow Critic
        self.critics_["PathFollowCritic"] = {
            "initialize": path_follow_critic_initialize,
            "score": path_follow_critic_score,
        }

        # Path Angle Critic
        self.critics_["PathAngleCritic"] = {
            "initialize": path_angle_critic_initialize,
            "score": path_angle_critic_score,
        }

        # Prefer Forward Critic
        self.critics_["PreferForwardCritic"] = {
            "initialize": prefer_forward_critic_initialize,
            "score": prefer_forward_critic_score,
        }

        # Twirling Critic
        self.critics_["TwirlingCritic"] = {
            "initialize": twirling_critic_initialize,
            "score": twirling_critic_score,
        }

        # Velocity Deadband Critic
        self.critics_["VelocityDeadbandCritic"] = {
            "initialize": velocity_deadband_critic_initialize,
            "score": velocity_deadband_critic_score,
        }

        enabled_critics = {}
        for name, critic in self.critics_.items():
            critic_params = critic["initialize"](params.get(name, {}), constraints)

            if not critic_params.enabled:
                continue

            if name == "ObstaclesCritic":
                experiment_params = params.get(name, {})
                collision_substeps = int(experiment_params.get("collision_substeps", 1))
                if collision_substeps not in (1, 2, 4, 8):
                    raise ValueError("collision_substeps must be one of 1, 2, 4, or 8")
                point_chunk_size = int(experiment_params.get("point_chunk_size", 32))
                if point_chunk_size not in (16, 32, 64):
                    raise ValueError("point_chunk_size must be one of 16, 32, or 64")
                critic["score"] = partial(
                    critic["score"],
                    params=critic_params,
                    point_flow_distance_enabled=("collision_substeps" in experiment_params),
                    collision_substeps=collision_substeps,
                    point_chunk_size=point_chunk_size,
                )
            else:
                critic["score"] = partial(critic["score"], params=critic_params)
            enabled_critics[name] = critic

        self.critics_ = enabled_critics

    def evalTrajectoriesScores(
        self, critics_data: CriticData
    ) -> Tuple[jax.Array, jax.Array, dict]:
        """
        Score trajectories by the set of loaded critic functions

        Args:
            critics_data: Struct of necessary information to pass to the critic functions

        Returns:
            Tuple of [costs of the trajectories, fail flag, critics info]
        """

        total_costs = jnp.zeros(critics_data.trajectories.x.shape[0])  # (K,)

        furthest_reached_path_point = findPathFurthestReachedPoint(
            critics_data, self.furthest_path_lookahead_distance_
        )

        if isinstance(critics_data.footprint, Rectangles):
            dist_min = minimum_signed_distance_from_traj_to_obs_points(
                critics_data.path.x[:-1],
                critics_data.path.y[:-1],
                critics_data.path.yaws[:-1],
                critics_data.obs_points,
                critics_data.obs_points_mask,
                critics_data.footprint.centers,
                critics_data.footprint.halfs,
            )  # (P-1,)
        elif isinstance(critics_data.footprint, Polygons):
            dist_min = poly_sdf.minimum_signed_distance_from_traj_to_obs_points(
                critics_data.path.x[:-1],
                critics_data.path.y[:-1],
                critics_data.path.yaws[:-1],
                critics_data.obs_points,
                critics_data.obs_points_mask,
                vertices=critics_data.footprint.vertices,
                vertex_counts=critics_data.footprint.vertex_counts,
            )  # (P-1,)
        else:
            dist_min = jnp.full(critics_data.path.x[:-1].shape, 1000.0)

        path_pts_valid = dist_min > 0.0  # (P-1,)
        # path_pts_valid = jnp.ones(critics_data.path.x.shape[0] - 1, dtype=jnp.bool_)

        critics_data = replace(
            critics_data,
            furthest_reached_path_point=furthest_reached_path_point,
            path_pts_valid=path_pts_valid,
        )

        critics_info = {}
        if self.debug_:
            critics_info = {
                "costs_debug": {name: None for name in self.critics_.keys()}
            }

        fail_flag = jnp.array(False)

        for name, critic in self.critics_.items():
            costs, info = critic["score"](critics_data)
            total_costs += costs

            if name == "ObstaclesCritic":
                fail_flag = info["fail_flag"]

            if self.debug_:
                critics_info["costs_debug"][name] = costs
                critics_info = {**critics_info, **info}

        return total_costs, fail_flag, critics_info
