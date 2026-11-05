from typing import Tuple, Optional
from functools import partial
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from dataclasses import replace

from .logger import logger

from .models import *
from .motion_models import *
from .critics import CriticData
from .critic_manager import CriticManager
from .optimal_trajectory_validator import OptimalTrajectoryValidator, ValidationResult
from .tools import utils


class Optimizer:
    def __init__(
        self,
        motion_model: str = "diff",
        model_dt: float = 0.05,
        time_steps: int = 56,
        batch_size: int = 1000,
        iteration_count: int = 1,
        temperature: float = 0.3,
        gamma: float = 0.015,
        vx_max: float = 0.5,
        vx_min: float = -0.35,
        vy_max: float = 0.5,
        wz_max: float = 1.9,
        ax_max: float = 3.0,
        ax_min: float = -3.0,
        ay_max: float = 3.0,
        ay_min: float = -3.0,
        az_max: float = 3.5,
        vx_std: float = 0.2,
        vy_std: float = 0.2,
        wz_std: float = 0.4,
        retry_attempt_limit: int = 1,
        open_loop: bool = False,
        seed: int = 0,
        debug: bool = False,
        **kwargs,
    ):
        self.debug_ = debug
        self.counter_ = 0
        self.last_command_vel_ = jnp.array([0.0, 0.0, 0.0])

        ax_max = abs(ax_max)
        if ax_min > 0.0:
            ax_min = -ax_min
            logger.warning(
                "Sign of the parameter ax_min is incorrect, consider setting it negative."
            )

        if ay_min > 0.0:
            ay_min = -ay_min
            logger.warning(
                "Sign of the parameter ay_min is incorrect, consider setting it negative."
            )

        constraints = ControlConstraints(
            vx_max=vx_max,
            vx_min=vx_min,
            vy=vy_max,
            wz=wz_max,
            ax_max=ax_max,
            ax_min=ax_min,
            ay_max=ay_max,
            ay_min=ay_min,
            az_max=az_max,
        )

        self.settings_ = OptimizerSettings(
            constraints=constraints,
            sampling_std=SamplingStd(vx=vx_std, vy=vy_std, wz=wz_std),
            model_dt=float(model_dt),
            temperature=float(temperature),
            gamma=float(gamma),
            batch_size=int(batch_size),
            time_steps=int(time_steps),
            iteration_count=int(iteration_count),
            shift_control_sequence=True,
            retry_attempt_limit=int(retry_attempt_limit),
            open_loop=open_loop,
        )

        self.setMotionModel(motion_model, **kwargs)

        critic_params = kwargs.get("Critics", {})
        self.critic_manager_ = CriticManager(critic_params, constraints, self.debug_)

        trajectory_validator_params = kwargs.get("TrajectoryValidator", {})
        obstacle_critic_params = critic_params.get("ObstaclesCritic", {})
        point_flow_distance_enabled = "collision_substeps" in obstacle_critic_params
        self.trajectory_validator_ = OptimalTrajectoryValidator(
            self.settings_,
            trajectory_validator_params.get("collision_lookahead_time", 2.0),
            trajectory_validator_params.get("collision_margin_distance", 0.1),
            point_flow_distance_enabled=point_flow_distance_enabled,
            collision_substeps=int(obstacle_critic_params.get("collision_substeps", 1)),
            point_chunk_size=int(obstacle_critic_params.get("point_chunk_size", 32)),
        )

        self.reset(seed)

    def reset(self, seed: int = 0):
        self.control_sequence_ = reset_ControlSequence(self.settings_.time_steps)
        self.control_history_ = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        if self.settings_.open_loop:
            self.last_command_vel_ = jnp.array([0.0, 0.0, 0.0])

        if self.debug_:
            self.generated_trajectories_ = reset_Trajectories(
                self.settings_.batch_size, self.settings_.time_steps
            )
            self.costs_ = jnp.zeros(self.settings_.batch_size)
            self.critics_info_ = {}

        self.key_ = jax.random.PRNGKey(seed)
        logger.info("Optimizer reset")

    def isHolonomic(self) -> bool:
        return self.motion_model_.isHolonomic()

    def evalControl(
        self,
        robot_pose: ArrayLike,
        robot_speed: ArrayLike,
        plan: ArrayLike,
        goal: ArrayLike,
        obs_points: jax.Array,
        obs_velocities: Optional[jax.Array],
        obs_points_mask: jax.Array,
        footprint: ControlConstraints,  # Should use proper type hint, likely Any
    ) -> Tuple[jax.Array, jax.Array]:
        # Normalize point velocities before calling the jitted numerical path.
        if obs_velocities is None:
            obs_velocities = jnp.zeros_like(obs_points)
        else:
            obs_velocities = jnp.asarray(obs_velocities)

        critics_data = self.prepare(
            robot_pose,
            robot_speed,
            plan,
            goal,
            obs_points,
            obs_velocities,
            obs_points_mask,
            footprint,
            self.last_command_vel_,
        )

        while True:
            (
                self.key_,
                self.control_sequence_,
                optimal_trajectory,
                fail_flag,
                generated_trajectories,
                costs,
                critics_info,
            ) = self._evalControl_optimize_jit(
                self.key_,
                self.control_sequence_,
                critics_data,
                robot_pose,
            )

            validation_result, min_dist = self.trajectory_validator_.validateTrajectory(
                optimal_trajectory,
                obs_points,
                obs_points_mask,
                footprint,
                obs_velocities=obs_velocities,
                interval_vx=self.control_sequence_.vx,
                interval_wz=self.control_sequence_.wz,
                initial_pose=jnp.asarray(robot_pose).reshape(-1),
            )

            if validation_result == ValidationResult.SOFT_RESET:
                trajectory_valid = False
                logger.warning(
                    "Soft reset triggered by trajectory validator "
                    f"(min_dist={min_dist:.3f} < margin={self.trajectory_validator_.collision_margin_distance_:.3f})"
                )
            elif validation_result == ValidationResult.FAILURE:
                raise RuntimeError(
                    "Trajectory validator failed to validate trajectory, hard reset triggered."
                )
            elif validation_result == ValidationResult.SUCCESS:
                trajectory_valid = True
            else:
                trajectory_valid = True

            if not self.fallback(bool(fail_flag) or (not trajectory_valid)):
                break

        (
            self.control_sequence_,
            self.control_history_,
            control,
        ) = self._evalControl_process_jit(
            self.control_sequence_,
            self.control_history_,
        )

        self.last_command_vel_ = control

        if self.debug_:
            self.generated_trajectories_ = generated_trajectories
            self.costs_ = costs
            self.critics_info_ = critics_info

        return control, optimal_trajectory

    @partial(jax.jit, static_argnames=("self",), donate_argnames=("control_sequence",))
    def _evalControl_optimize_jit(
        self,
        key: jax.Array,
        control_sequence: ControlSequence,
        critics_data: CriticData,
        pose: jax.Array,
    ):
        key, control_sequence, fail_flag, trajectories, costs, critics_info = (
            self.optimize(key, control_sequence, critics_data)
        )

        optimal_trajectory = self.getOptimizedTrajectory(control_sequence, pose)

        return (
            key,
            control_sequence,
            optimal_trajectory,
            fail_flag,
            trajectories,
            costs,
            critics_info,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
        donate_argnames=("control_sequence", "control_history"),
    )
    def _evalControl_process_jit(
        self,
        control_sequence: ControlSequence,
        control_history: jax.Array,
    ):
        control_sequence, control_history = utils.savitskyGolayFilter(
            control_sequence,
            control_history,
            self.settings_.shift_control_sequence,
        )

        control = self.getControlFromSequence(control_sequence)

        if self.settings_.shift_control_sequence:
            control_sequence = self.shiftControlSequence(control_sequence)

        return control_sequence, control_history, control

    def optimize(
        self,
        key: jax.Array,
        control_sequence: ControlSequence,
        critics_data: CriticData,
    ) -> Tuple[
        jax.Array,
        ControlSequence,
        jax.Array,
        Optional[Trajectories],
        Optional[jax.Array],
        Optional[dict],
    ]:
        if self.debug_:
            for i in range(self.settings_.iteration_count):
                key, state, trajectories = self.generateNoisedTrajectories(
                    key, critics_data.state, control_sequence
                )
                critics_data = replace(
                    critics_data, state=state, trajectories=trajectories
                )
                costs, fail_flag, critics_info = (
                    self.critic_manager_.evalTrajectoriesScores(critics_data)
                )
                control_sequence, costs = self.updateControlSequence(
                    control_sequence, state, costs
                )

            return (
                key,
                control_sequence,
                fail_flag,
                critics_data.trajectories,
                costs,
                critics_info,
            )
        else:

            def step(i, carry):
                key, control_sequence, critics_data, _ = carry

                key, state, trajectories = self.generateNoisedTrajectories(
                    key, critics_data.state, control_sequence
                )
                critics_data = replace(
                    critics_data, state=state, trajectories=trajectories
                )

                costs, fail_flag, _ = self.critic_manager_.evalTrajectoriesScores(
                    critics_data
                )

                control_sequence, costs = self.updateControlSequence(
                    control_sequence, state, costs
                )

                return key, control_sequence, critics_data, fail_flag

            key, control_sequence, critics_data, fail_flag = jax.lax.fori_loop(
                0,
                self.settings_.iteration_count,
                step,
                (
                    key,
                    control_sequence,
                    critics_data,
                    jnp.array(False),
                ),
            )

            return key, control_sequence, fail_flag, None, None, None

    def fallback(self, fail: bool) -> bool:
        if not fail:
            self.counter_ = 0
            return False

        self.reset()

        self.counter_ += 1
        if self.counter_ > self.settings_.retry_attempt_limit:
            self.counter_ = 0
            raise RuntimeError("Optimizer fail to compute path")

        return True

    @partial(jax.jit, static_argnames=("self",))
    def prepare(
        self,
        robot_pose: jax.Array,
        robot_speed: jax.Array,
        plan: jax.Array,
        goal: jax.Array,
        obs_points: jax.Array,
        obs_velocities: Optional[jax.Array],
        obs_points_mask: jax.Array,
        footprint: jax.Array, # Actually Union[Rectangles, Polygons]
        last_command_vel: jax.Array,
    ) -> CriticData:
        pose = robot_pose  # (3,) -> [x, y, yaw]
        speed = (
            last_command_vel if self.settings_.open_loop else robot_speed
        )  # (3,) -> [vx, vy, wz]
        path_array = plan  # (T, 3) -> [x, y, yaw]
        local_path_length = jnp.linalg.norm(
            path_array[1:, :2] - path_array[:-1, :2], axis=1
        ).sum()
        state = reset_State(self.settings_.batch_size, self.settings_.time_steps)
        state = replace(
            state, pose=pose, speed=speed, local_path_length=local_path_length
        )
        if obs_velocities is None:
            obs_velocities = jnp.zeros_like(obs_points)
        else:
            obs_velocities = jnp.asarray(obs_velocities)

        critics_data = CriticData(
            state=state,
            trajectories=reset_Trajectories(
                self.settings_.batch_size, self.settings_.time_steps
            ),
            path=Path(x=path_array[:, 0], y=path_array[:, 1], yaws=path_array[:, 2]),
            goal=goal,
            model_dt=self.settings_.model_dt,
            motion_model=MotionModelParams(
                model_type=self.motion_model_.model_type_.value,
                min_turning_rad=self.motion_model_.getMinTurningRadius(),
                is_holonomic=self.isHolonomic(),
            ),
            furthest_reached_path_point=0,
            path_pts_valid=jnp.ones(path_array.shape[0] - 1, dtype=jnp.bool_),
            obs_points=obs_points,
            obs_velocities=obs_velocities,
            obs_points_mask=obs_points_mask,
            footprint=footprint,
        )

        return critics_data

    def shiftControlSequence(
        self, control_sequence: ControlSequence
    ) -> ControlSequence:
        """
        Shift the optimal control sequence after processing for
        next iterations initial conditions after execution

        Args:
            control_sequence: Control sequence to shift

        Returns:
            ControlSequence: shifted control sequence
        """
        vx_shifted = jnp.concatenate(
            [control_sequence.vx[1:], control_sequence.vx[-1:]]
        )
        wz_shifted = jnp.concatenate(
            [control_sequence.wz[1:], control_sequence.wz[-1:]]
        )

        if self.isHolonomic():
            vy_shifted = jnp.concatenate(
                [control_sequence.vy[1:], control_sequence.vy[-1:]]
            )
        else:
            vy_shifted = control_sequence.vy

        return ControlSequence(
            vx=vx_shifted,
            vy=vy_shifted,
            wz=wz_shifted,
        )

    def generateNoisedTrajectories(
        self,
        key: jax.Array,
        state: State,
        control_sequence: ControlSequence,
    ) -> Tuple[jax.Array, State, Trajectories]:
        """
        Updates generated trajectories with noised trajectories
        from the last cycle's optimal control

        Args:
            key: Random key
            state: State to be updated
            control_sequence: Last cycle's optimal control

        Returns:
            Tuple of updated [key, state, trajectories]
        """
        K, T = self.settings_.batch_size, self.settings_.time_steps
        key, k1, k2, k3 = jax.random.split(key, 4)

        noise_vx = jax.random.normal(k1, (K, T)) * self.settings_.sampling_std.vx
        noise_wz = jax.random.normal(k2, (K, T)) * self.settings_.sampling_std.wz

        if self.isHolonomic():
            noise_vy = jax.random.normal(k3, (K, T)) * self.settings_.sampling_std.vy
        else:
            noise_vy = jnp.zeros((K, T))

        cvx_new = noise_vx + control_sequence.vx[None, :]
        cvy_new = noise_vy + control_sequence.vy[None, :]
        cwz_new = noise_wz + control_sequence.wz[None, :]
        state = replace(state, cvx=cvx_new, cvy=cvy_new, cwz=cwz_new)

        state = self.updataStateVelocities(state)
        trajectories = self.integrateStateVelocities(state)

        return key, state, trajectories

    def applyControlSequenceConstraints(
        self, control_sequence: ControlSequence
    ) -> ControlSequence:
        """
        Apply hard vehicle constraints on control sequence

        Args:
            control_sequence: Control sequence to apply constraints to

        Returns:
            ControlSequence: Control sequence with constraints applied
        """
        s = self.settings_

        max_delta_vx = s.model_dt * s.constraints.ax_max
        min_delta_vx = s.model_dt * s.constraints.ax_min
        max_delta_vy = s.model_dt * s.constraints.ay_max
        min_delta_vy = s.model_dt * s.constraints.ay_min
        max_delta_wz = s.model_dt * s.constraints.az_max

        vx = control_sequence.vx
        wz = control_sequence.wz
        vx_0 = jnp.clip(vx[0], s.constraints.vx_min, s.constraints.vx_max)
        wz_0 = jnp.clip(wz[0], -s.constraints.wz, s.constraints.wz)

        def step_vx_wz(carry, input):
            vx_last, wz_last = carry
            vx_curr, wz_curr = input

            vx_curr = jnp.clip(vx_curr, s.constraints.vx_min, s.constraints.vx_max)
            lo_vx = jnp.where(
                vx_last > 0.0, vx_last + min_delta_vx, vx_last - max_delta_vx
            )
            hi_vx = jnp.where(
                vx_last > 0.0, vx_last + max_delta_vx, vx_last - min_delta_vx
            )
            vx_curr = jnp.clip(vx_curr, lo_vx, hi_vx)

            wz_curr = jnp.clip(wz_curr, -s.constraints.wz, s.constraints.wz)
            wz_curr = jnp.clip(wz_curr, wz_last - max_delta_wz, wz_last + max_delta_wz)

            return (vx_curr, wz_curr), (vx_curr, wz_curr)

        (_, _), (vx_rest, wz_rest) = jax.lax.scan(
            step_vx_wz, (vx_0, wz_0), (vx[1:], wz[1:])
        )

        vx_new = jnp.concatenate([vx_0[None], vx_rest], axis=0)
        wz_new = jnp.concatenate([wz_0[None], wz_rest], axis=0)

        if self.isHolonomic():
            vy = control_sequence.vy
            vy_0 = jnp.clip(vy[0], -s.constraints.vy, s.constraints.vy)

            def step_vy(vy_last, vy_curr):
                vy_curr = jnp.clip(vy_curr, -s.constraints.vy, s.constraints.vy)
                lo_vy = jnp.where(
                    vy_last > 0.0, vy_last + min_delta_vy, vy_last - max_delta_vy
                )
                hi_vy = jnp.where(
                    vy_last > 0.0, vy_last + max_delta_vy, vy_last - min_delta_vy
                )
                vy_curr = jnp.clip(vy_curr, lo_vy, hi_vy)
                return vy_curr, vy_curr

            _, vy_rest = jax.lax.scan(step_vy, vy_0, vy[1:])
            vy_new = jnp.concatenate([vy_0[None], vy_rest], axis=0)
        else:
            vy_new = control_sequence.vy

        control_sequence = replace(control_sequence, vx=vx_new, vy=vy_new, wz=wz_new)

        control_sequence = self.motion_model_.applyConstraints(control_sequence)
        return control_sequence

    def updataStateVelocities(self, state: State) -> State:
        """
        Update velocities in state

        Args:
            state: fill state with velocities on each step

        Returns:
            State: filled state with velocities on each step
        """
        state = self.updateInitialStateVelocities(state)
        state = self.propagateStateVelocitiesFromInitials(state)
        return state

    def updateInitialStateVelocities(self, state: State) -> State:
        """
        Update initial velocity in state

        Args:
            state: fill state

        Returns:
            State: filled state
        """
        vx_new = state.vx.at[:, 0].set(state.speed[0])
        wz_new = state.wz.at[:, 0].set(state.speed[2])

        if self.isHolonomic():
            vy_new = state.vy.at[:, 0].set(state.speed[1])
        else:
            vy_new = state.vy

        return replace(state, vx=vx_new, vy=vy_new, wz=wz_new)

    def propagateStateVelocitiesFromInitials(self, state: State) -> State:
        """
        Predict velocities in state using model
        for time horizon equal to timesteps

        Args:
            state: fill state

        Returns:
            State: filled state
        """
        state = self.motion_model_.predict(state)
        return state

    def integrateSequenceVelocities(
        self, sequence: ControlSequence, pose: jax.Array
    ) -> jax.Array:
        """
        Rollout velocities in control sequence to poses

        Args:
            sequence: Control sequence to roll out
            pose: Initial pose to roll out from (3,)

        Returns:
            jax.Array: rolled out trajectory (T, 3)
        """
        initial_yaw = pose[2]

        vx = sequence.vx
        wz = sequence.wz

        def yaw_step(last_yaw, wz_t):
            last_yaw = last_yaw + wz_t * self.settings_.model_dt
            return last_yaw, last_yaw

        _, traj_yaws = jax.lax.scan(yaw_step, initial_yaw, wz)  # (T,)

        yaw_cos = jnp.cos(traj_yaws)
        yaw_sin = jnp.sin(traj_yaws)
        yaw_cos = jnp.concatenate([jnp.cos(initial_yaw)[None], yaw_cos[:-1]], axis=0)
        yaw_sin = jnp.concatenate([jnp.sin(initial_yaw)[None], yaw_sin[:-1]], axis=0)

        dx = vx * yaw_cos
        dy = vx * yaw_sin

        if self.isHolonomic():
            vy = sequence.vy
            dx = dx - vy * yaw_sin
            dy = dy + vy * yaw_cos

        def xy_step(carry, dxy_t):
            last_x, last_y = carry
            dx_t, dy_t = dxy_t
            last_x = last_x + dx_t * self.settings_.model_dt
            last_y = last_y + dy_t * self.settings_.model_dt
            return (last_x, last_y), (last_x, last_y)

        (_, _), (traj_xs, traj_ys) = jax.lax.scan(
            xy_step, (pose[0], pose[1]), (dx, dy)
        )  # (T,)

        trajectory = jnp.stack([traj_xs, traj_ys, traj_yaws], axis=1)  # (T, 3)
        return trajectory

    def integrateStateVelocities(self, state: State) -> Trajectories:
        """
        Rollout velocities in state to poses

        Args:
            state: fill state

        Returns:
            Trajectories: rolled out trajectories
        """
        initial_yaw = state.pose[2]
        K, T = state.wz.shape

        def yaw_step(last_yaws, wz_t):
            last_yaws = last_yaws + wz_t * self.settings_.model_dt  # (K,)
            return last_yaws, last_yaws

        init_last_yaws = jnp.full((K,), initial_yaw)
        _, yaws_T = jax.lax.scan(yaw_step, init_last_yaws, state.wz.T)  # (T, K)
        yaws = yaws_T.T  # (K, T)

        yaw_cos = jnp.cos(yaws)
        yaw_sin = jnp.sin(yaws)
        yaw_cos = jnp.concatenate([yaw_cos[:, :1], yaw_cos[:, :-1]], axis=1)
        yaw_sin = jnp.concatenate([yaw_sin[:, :1], yaw_sin[:, :-1]], axis=1)
        yaw_cos = yaw_cos.at[:, 0].set(jnp.cos(initial_yaw))
        yaw_sin = yaw_sin.at[:, 0].set(jnp.sin(initial_yaw))

        dx = state.vx * yaw_cos
        dy = state.vx * yaw_sin

        if self.isHolonomic():
            dx = dx - state.vy * yaw_sin
            dy = dy + state.vy * yaw_cos

        def xy_step(carry, dxy_t):
            last_x, last_y = carry  # (K,)
            dx_t, dy_t = dxy_t
            last_x = last_x + dx_t * self.settings_.model_dt
            last_y = last_y + dy_t * self.settings_.model_dt
            return (last_x, last_y), (last_x, last_y)

        init_last_x = jnp.full((K,), state.pose[0])
        init_last_y = jnp.full((K,), state.pose[1])
        (_, _), (xs_T, ys_T) = jax.lax.scan(
            xy_step, (init_last_x, init_last_y), (dx.T, dy.T)
        )  # (T, K)
        xs = xs_T.T  # (K, T)
        ys = ys_T.T  # (K, T)

        return Trajectories(x=xs, y=ys, yaws=yaws)

    def getOptimizedTrajectory(
        self, control_sequence: ControlSequence, pose: jax.Array
    ) -> jax.Array:
        """
        Get the optimal trajectory for a cycle for visualization

        Args:
            control_sequence: Control sequence to get the optimal trajectory for
            pose: Initial pose to roll out from (3,)

        Returns:
            jax.Array: Optimal trajectory
        """
        return self.integrateSequenceVelocities(control_sequence, pose)

    def updateControlSequence(
        self, control_sequence: ControlSequence, state: State, costs: jax.Array
    ) -> Tuple[ControlSequence, jax.Array]:
        """
        Update control sequence with state controls weighted by costs
        using softmax function

        Args:
            control_sequence: Control sequence to update
            state: State to update control sequence with
            costs: Costs to update control sequence with

        Returns:
            Tuple of updated [control sequence, costs]
        """

        is_holo = self.isHolonomic()
        s = self.settings_

        vx_T = control_sequence.vx[None, :]  # (1, T)
        bounded_noises_vx = state.cvx - vx_T  # (K, T)
        gamma_vx = s.gamma / (s.sampling_std.vx**2)
        costs = costs + gamma_vx * jnp.sum(bounded_noises_vx * vx_T, axis=1)  # (K,)

        if s.sampling_std.wz > 0.0:
            wz_T = control_sequence.wz[None, :]  # (1, T)
            bounded_noises_wz = state.cwz - wz_T  # (K, T)
            gamma_wz = s.gamma / (s.sampling_std.wz**2)
            costs = costs + gamma_wz * jnp.sum(bounded_noises_wz * wz_T, axis=1)  # (K,)

        if self.isHolonomic() and s.sampling_std.vy > 0.0:
            vy_T = control_sequence.vy[None, :]  # (1, T)
            bounded_noises_vy = state.cvy - vy_T  # (K, T)
            gamma_vy = s.gamma / (s.sampling_std.vy**2)
            costs = costs + gamma_vy * jnp.sum(bounded_noises_vy * vy_T, axis=1)  # (K,)

        costs_normalized = costs - jnp.min(costs)  # (K,)
        inv_temp = 1.0 / s.temperature
        softmaxes = jax.nn.softmax(-inv_temp * costs_normalized)  # (K,)

        vx_new = jnp.sum(state.cvx * softmaxes[:, None], axis=0)  # (T,)
        wz_new = jnp.sum(state.cwz * softmaxes[:, None], axis=0)  # (T,)

        if is_holo:
            vy_new = jnp.sum(state.cvy * softmaxes[:, None], axis=0)  # (T,)
        else:
            vy_new = control_sequence.vy

        control_sequence = ControlSequence(vx=vx_new, vy=vy_new, wz=wz_new)
        control_sequence = self.applyControlSequenceConstraints(control_sequence)

        return control_sequence, costs

    def getControlFromSequence(self, control_sequence: ControlSequence) -> jax.Array:
        """
        Convert control sequence to a command

        Args:
            control_sequence: Control sequence

        Returns:
            jax.Array: Command
        """
        offset = 1 if self.settings_.shift_control_sequence else 0

        vx = control_sequence.vx[offset]
        wz = control_sequence.wz[offset]

        if self.isHolonomic():
            vy = control_sequence.vy[offset]
        else:
            vy = 0.0

        return jnp.array([vx, vy, wz])

    def setMotionModel(self, model: str, **kwargs):
        """
        Set the motion model of the vehicle platform

        Args:
            model: Model string to use
        """
        if model.lower() == "diff" or model.lower() == "diffdrive":
            self.motion_model_ = DiffDriveMotionModel()
        elif model.lower() == "omni":
            self.motion_model_ = OmniMotionModel()
        elif model.lower() == "acker" or model.lower() == "ackermann":
            min_turning_r = kwargs.get("AckermannConstraints", {}).get(
                "min_turning_r", 0.2
            )
            self.motion_model_ = AckermannMotionModel(min_turning_r)
        elif model.lower() == "omni_xy":
            self.motion_model_ = OmniXYMotionModel()
        elif model.lower() == "rangerminiv3":
            self.motion_model_ = RangerMiniV3MotionModel()
        else:
            raise Exception(f"[MPPI] Model {model} is not valid!")

        self.motion_model_.initialize(
            self.settings_.constraints, self.settings_.model_dt
        )

    def getMotionModelType(self) -> MotionModelType:
        """
        Get motion model type

        Returns:
            MotionModelType: Motion model type
        """
        return self.motion_model_.model_type_

    def getOptimalControlSequence(self) -> ControlSequence:
        """
        Get the optimal control sequence for a cycle for visualization

        Returns:
            ControlSequence: Optimal control sequence
        """
        return self.control_sequence_

    def getGeneratedTrajectories(self) -> Optional[jax.Array]:
        """
        Get the trajectories generated in a cycle for visualization

        Returns:
            Trajectories: Set of trajectories evaluated in cycle
        """
        if not hasattr(self, "generated_trajectories_"):
            # logger.warning("Generated trajectories are only available in debug mode.")
            return None

        generated_trajectories = jnp.stack(
            (
                self.generated_trajectories_.x,
                self.generated_trajectories_.y,
                self.generated_trajectories_.yaws,
            ),
            axis=-1,
        )
        return generated_trajectories

    def getCosts(self) -> Optional[jax.Array]:
        """
        Get the total costs of the generated trajectories

        Returns:
            jax.Array: Total costs of the generated trajectories
        """
        if not hasattr(self, "costs_"):
            # logger.warning("Costs are only available in debug mode.")
            return None

        return self.costs_

    def getCostsDebug(self) -> Optional[dict]:
        """
        Get the debug costs of each critic for the generated trajectories

        Returns:
            Optional[dict]: Debug costs dictionary of each critic
        """
        try:
            return self.critics_info_.get("costs_debug")
        except AttributeError:
            # logger.warning("Debug costs are only available in debug mode.")
            return None

    def getPathFollowPoint(self) -> Optional[jax.Array]:
        """
        Get the path follow point for visualization

        Returns:
            Optional[jax.Array]: Path follow point
        """
        try:
            path_follow_point = self.critics_info_.get("path_follow_point")
            if bool(jnp.any(jnp.isnan(path_follow_point))):
                return None
            return path_follow_point
        except AttributeError:
            # logger.warning("Path follow point is only available in debug mode.")
            return None
