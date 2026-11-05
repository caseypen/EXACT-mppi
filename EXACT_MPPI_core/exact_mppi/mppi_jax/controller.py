from typing import Optional, Tuple, List, Union
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .logger import logger

from .optimizer import Optimizer
from .models import Rectangles, Polygons
from .motion_models import MotionModelType
from .rangerminiv3 import RangerMiniV3Helper, RangerMiniMotionState


class MPPIController:

    def __init__(self, *args, **kwargs):
        self.optimizer_ = Optimizer(*args, **kwargs)

        self.max_obs_num_ = kwargs.get("max_obs_num", 100)
        self.wheelbase_ = kwargs.get("wheelbase", 1.6)
        self.optimal_trajectory_ = None

        if (
            self.optimizer_.getMotionModelType()
            == MotionModelType.RangerMiniV3MotionModel
        ):
            self.rangerminiv3_helper_ = RangerMiniV3Helper()
            self.last_motion_mode_ = self.rangerminiv3_helper_.motion_mode_

        logger.info("Configured MPPI Controller")

    def reset(self):
        self.optimizer_.reset()

        if hasattr(self, "rangerminiv3_helper_"):
            self.rangerminiv3_helper_.reset()

    def computeVelocityCommands(
        self,
        robot_pose: ArrayLike,
        robot_speed: ArrayLike,
        plan: ArrayLike,
        goal: ArrayLike,
        lidar_points: Optional[np.ndarray],
        lidar_velocities: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        obs_points, obs_velocities, obs_points_mask = self._pack_lidar_points(
            lidar_points, lidar_velocities, self.max_obs_num_
        )
        try:
            cmd, self.optimal_trajectory_ = self.optimizer_.evalControl(
                robot_pose=self._to_jnp(robot_pose),
                robot_speed=self._to_jnp(robot_speed),
                plan=self._to_jnp(plan),
                goal=self._to_jnp(goal),
                obs_points=self._to_jnp(obs_points),
                obs_velocities=self._to_jnp(obs_velocities),
                obs_points_mask=self._to_jnp(obs_points_mask),
                footprint=self.footprint_,
            )
        except RuntimeError as e:
            logger.error(e)

            cmd = jnp.array([0.0, 0.0, 0.0])
            self.optimal_trajectory_ = None

        if self.optimizer_.getMotionModelType() == MotionModelType.AckermannMotionModel:
            eps = 1.0e-3
            vx = cmd[0]
            v_safe = jnp.where(
                jnp.abs(vx) > eps, vx, eps * jnp.where(vx >= 0.0, 1.0, -1.0)
            )
            steer_cmd = jnp.arctan((self.wheelbase_ * cmd[2]) / v_safe)
            control_cmd = jnp.array([cmd[0], steer_cmd])

        elif self.optimizer_.getMotionModelType() == MotionModelType.OmniMotionModel:
            control_cmd = cmd

        elif self.optimizer_.getMotionModelType() == MotionModelType.OmniXYMotionModel:
            control_cmd = jnp.array([cmd[0], cmd[1], 0.0])

        elif (
            self.optimizer_.getMotionModelType()
            == MotionModelType.RangerMiniV3MotionModel
        ):
            vx = cmd[0]
            vy = cmd[1]
            wz = cmd[2]

            mode, cmd_vx, cmd_vy, cmd_wz = self.rangerminiv3_helper_.process_mppi_cmd(
                vx, vy, wz
            )
            if mode != self.last_motion_mode_:
                mode_name = RangerMiniMotionState(mode).name
                logger.info(f"Switched to: {mode_name}")
                self.last_motion_mode_ = mode

            control_cmd = jnp.array([cmd_vx, cmd_vy, cmd_wz])

        else:  # Differential drive motion model
            control_cmd = jnp.array([cmd[0], cmd[2]])

        return jax.device_get(control_cmd)

    def setRectangleFootprint(self, vertices: List[List[List[float]]]):
        rects = np.asarray(vertices, dtype=np.float32)
        mins = rects.min(axis=1)  # (P, 2)
        maxs = rects.max(axis=1)  # (P, 2)

        self.footprint_ = Rectangles(
            centers=self._to_jnp(0.5 * (mins + maxs)),
            halfs=self._to_jnp(0.5 * (maxs - mins)),
        )

    def setPolygonFootprint(self, polygon_list: List[List[List[float]]]):
        """
        polygon_list: list of polygons, where each polygon is a list of [x, y] vertices.
        """
        valid_polys = [p for p in polygon_list if len(p) > 0]
        if not valid_polys:
            # Fallback for empty
            self.footprint_ = Polygons(
                vertices=self._to_jnp(np.zeros((0, 0, 2), dtype=np.float32)),
                vertex_counts=self._to_jnp(np.zeros((0,), dtype=np.int32)),
            )
            return

        max_v = max(len(p) for p in valid_polys)
        num_polys = len(valid_polys)
        
        padded_vertices = np.zeros((num_polys, max_v, 2), dtype=np.float32)
        counts = np.zeros((num_polys,), dtype=np.int32)
        
        for i, poly in enumerate(valid_polys):
            count = len(poly)
            counts[i] = count
            padded_vertices[i, :count, :] = np.array(poly, dtype=np.float32)
        
        self.footprint_ = Polygons(
            vertices=self._to_jnp(padded_vertices),
            vertex_counts=self._to_jnp(counts, dtype=jnp.int32),
        )

    def getOptimalTrajectory(self) -> Optional[np.ndarray]:
        if self.optimal_trajectory_ is None:
            return None
        return jax.device_get(self.optimal_trajectory_)

    def getGeneratedTrajectories(self) -> np.ndarray:
        return jax.device_get(self.optimizer_.getGeneratedTrajectories())

    def getCosts(self) -> np.ndarray:
        return jax.device_get(self.optimizer_.getCosts())

    def getCostsDebug(self) -> dict:
        return jax.device_get(self.optimizer_.getCostsDebug())

    def getPathFollowPoint(self):
        return jax.device_get(self.optimizer_.getPathFollowPoint())

    @staticmethod
    def _pack_lidar_points(
        lidar_points: Optional[np.ndarray], lidar_velocities: Optional[np.ndarray] = None,
        Nmax: int = 100, dtype=jnp.float32
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        if lidar_points is None:
            obs_points = jnp.zeros((Nmax, 2), dtype=dtype)
            obs_velocities = jnp.zeros((Nmax, 2), dtype=dtype)
            obs_mask = jnp.zeros((Nmax,), dtype=dtype)
            return obs_points, obs_velocities, obs_mask

        points = np.asarray(lidar_points, dtype=np.float32)
        if points.size == 0:
            obs_points = jnp.zeros((Nmax, 2), dtype=dtype)
            obs_velocities = jnp.zeros((Nmax, 2), dtype=dtype)
            obs_points_mask = jnp.zeros((Nmax,), dtype=dtype)
            return obs_points, obs_velocities, obs_points_mask

        points = points.reshape((-1, 2))
        velocities = np.zeros_like(points)
        if lidar_velocities is not None:
            candidate = np.asarray(lidar_velocities, dtype=np.float32)
            if candidate.size == points.size:
                candidate = candidate.reshape((-1, 2))
                velocities = np.where(np.isfinite(candidate), candidate, 0.0)
        N = points.shape[0]

        if N > Nmax:
            r2 = np.einsum("ij,ij->i", points, points)
            idx = np.argpartition(r2, Nmax - 1)[:Nmax]
            points = points[idx]
            velocities = velocities[idx]
            N = Nmax

        obs_points = np.zeros((Nmax, 2), dtype=np.float32)
        obs_points[:N] = points
        obs_velocities = np.zeros((Nmax, 2), dtype=np.float32)
        obs_velocities[:N] = velocities

        obs_points_mask = np.zeros((Nmax,), dtype=np.float32)
        obs_points_mask[:N] = 1.0

        obs_points = jnp.asarray(obs_points, dtype=dtype)
        obs_velocities = jnp.asarray(obs_velocities, dtype=dtype)
        obs_points_mask = jnp.asarray(obs_points_mask, dtype=dtype)

        return obs_points, obs_velocities, obs_points_mask

    @staticmethod
    def _to_jnp(x, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray(x, dtype=dtype)
