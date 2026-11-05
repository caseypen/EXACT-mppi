from typing import Tuple, Union
import jax
import jax.numpy as jnp
from enum import IntEnum
from functools import partial

from .logger import logger
from .models import OptimizerSettings, Rectangles, Polygons
from .tools.signed_distance import minimum_signed_distance_from_traj_to_obs_points
from .tools.point_flow_distance import minimum_signed_distance_from_point_flow
import exact_mppi.mppi_jax.tools.signed_distance_polygon as poly_sdf


class ValidationResult(IntEnum):
    SUCCESS = 0
    SOFT_RESET = 1
    FAILURE = 2


class OptimalTrajectoryValidator:
    """
    Abstract base class for validating optimal trajectories from MPPI optimization
    """

    def __init__(
        self,
        settings: OptimizerSettings,
        collision_lookahead_time: float = 2.0,
        collision_margin_distance: float = 0.1,
        point_flow_distance_enabled: bool = False,
        collision_substeps: int = 1,
        point_chunk_size: int = 32,
    ):
        if collision_substeps not in (1, 2, 4, 8):
            raise ValueError("collision_substeps must be one of 1, 2, 4, or 8")
        if point_chunk_size not in (16, 32, 64):
            raise ValueError("point_chunk_size must be one of 16, 32, or 64")
        self.model_dt_ = float(settings.model_dt)
        self.point_flow_distance_enabled_ = bool(point_flow_distance_enabled)
        self.collision_substeps_ = int(collision_substeps)
        self.point_chunk_size_ = int(point_chunk_size)
        self.collision_lookahead_time_ = collision_lookahead_time
        self.traj_samples_to_evaluate_ = int(
            self.collision_lookahead_time_ / settings.model_dt
        )
        if self.traj_samples_to_evaluate_ > settings.time_steps:
            self.traj_samples_to_evaluate_ = int(settings.time_steps)
            logger.warning(
                f"Collision lookahead time is greater than the number of trajectory samples, "
                f"setting it to the maximum number of samples ({self.traj_samples_to_evaluate_})."
            )

        self.collision_margin_distance_ = max(0.0, collision_margin_distance)

    def validateTrajectory(
        self,
        optimal_trajectory: jax.Array,  # (T, 3)
        obs_points: jax.Array,  # (N, 2)
        obs_points_mask: jax.Array,  # (N,)
        footprint: Union[Rectangles, Polygons],
        obs_velocities: jax.Array | None = None,  # (N, 2), same frame as points
        interval_vx: jax.Array | None = None,  # (T,)
        interval_wz: jax.Array | None = None,  # (T,)
        initial_pose: jax.Array | None = None,  # (3,)
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Validate the optimal trajectory from MPPI optimization
        Could be used to check for collisions, progress towards goal,
        distance from path, min distance from obstacles, dynamic feasibility, etc.

        Args:
            optimal_trajectory: The optimal trajectory to validate
            obs_points: Obstacle points
            obs_points_mask: Obstacle points mask
            footprint: Rectangle or polygon footprint of the robot
            obs_velocities: Optional obstacle-point velocities in the same frame
                as ``obs_points``. Used when point-flow validation is enabled.
            interval_vx: Nominal forward velocities for trajectory intervals.
            interval_wz: Nominal angular velocities for trajectory intervals.
            initial_pose: Initial pose used to interpolate nominal robot motion.

        Returns:
            Tuple of [jax.Array, jax.Array]: [validation result, minimum distance from obstacles]
        """
        # The Optimizer automatically ensures that we are within Kinematic
        # and dynamic constraints, no need to check for those again.

        # Check for collisions. This is highly unlikely to occur since the Obstacle/Cost Critics
        # penalize collisions severely, but it is still possible if those critics are not used or the
        # optimized trajectory is very near obstacles and the dynamic constraints cause invalidity.
        result, dist_min = self._validateTrajectory_jit(
            optimal_trajectory,
            obs_points,
            obs_points_mask,
            footprint,
            obs_velocities,
            interval_vx,
            interval_wz,
            initial_pose,
        )
        return result, dist_min

    @partial(jax.jit, static_argnames=("self",))
    def _validateTrajectory_jit(
        self,
        optimal_trajectory: jax.Array,  # (T, 3)
        obs_points: jax.Array,  # (N, 2)
        obs_points_mask: jax.Array,  # (N,)
        footprint: Union[Rectangles, Polygons],
        obs_velocities: jax.Array | None,
        interval_vx: jax.Array | None,
        interval_wz: jax.Array | None,
        initial_pose: jax.Array | None,
    ) -> Tuple[jax.Array, jax.Array]:
        
        subset = optimal_trajectory[: self.traj_samples_to_evaluate_]
        x = subset[:, 0]
        y = subset[:, 1]
        yaw = subset[:, 2]

        if self.point_flow_distance_enabled_:
            if obs_velocities is None:
                obs_velocities = jnp.zeros_like(obs_points)
            if interval_vx is None or interval_wz is None or initial_pose is None:
                raise ValueError(
                    "point-flow validation requires interval velocities and initial_pose"
                )
            if isinstance(footprint, Rectangles):
                flow_kwargs = {
                    "rect_centers": footprint.centers,
                    "rect_halfs": footprint.halfs,
                }
            elif isinstance(footprint, Polygons):
                flow_kwargs = {
                    "vertices": footprint.vertices,
                    "vertex_counts": footprint.vertex_counts,
                }
            else:
                raise TypeError(f"Unknown footprint type: {type(footprint)}")
            dist_t = minimum_signed_distance_from_point_flow(
                x[None, :],
                y[None, :],
                yaw[None, :],
                interval_vx[: self.traj_samples_to_evaluate_][None, :],
                interval_wz[: self.traj_samples_to_evaluate_][None, :],
                initial_pose,
                obs_points,
                obs_velocities,
                obs_points_mask,
                self.model_dt_,
                self.collision_substeps_,
                self.point_chunk_size_,
                **flow_kwargs,
            )
        elif isinstance(footprint, Rectangles):
            dist_t = minimum_signed_distance_from_traj_to_obs_points(
                x, y, yaw,
                obs_points,
                obs_points_mask,
                footprint.centers,
                footprint.halfs,
            )
        elif isinstance(footprint, Polygons):
            dist_t = poly_sdf.minimum_signed_distance_from_traj_to_obs_points(
                x, y, yaw,
                obs_points,
                obs_points_mask,
                vertices=footprint.vertices,
                vertex_counts=footprint.vertex_counts,
            )
        else:
             # Fallback, though static analysis or python tracing should catch this
             dist_t = jnp.full(x.shape, 1000.0)

        dist_min = jnp.min(dist_t)
        unsafe = jnp.any(dist_t < self.collision_margin_distance_)

        result = jnp.where(
            unsafe,
            jnp.int32(ValidationResult.SOFT_RESET),
            jnp.int32(ValidationResult.SUCCESS),
        )
        return result, dist_min
