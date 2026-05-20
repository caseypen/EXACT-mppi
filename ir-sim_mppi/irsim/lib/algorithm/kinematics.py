"""
This file is the implementation of the kinematics for different robots.

Reference details omitted for anonymous review.
"""

from math import cos, sin, tan
import traceback
from typing import Optional, Tuple

import numpy as np

from irsim.util.random import rng
from irsim.util.util import WrapToPi


def differential_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Calculate the next state for a differential wheel robot.

    Args:
        state: A 3x1 vector [x, y, theta] representing the current position and orientation.
        velocity: A 2x1 vector [linear, angular] representing the current velocities.
        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.

    Returns:
        next_state: A 3x1 vector [x, y, theta] representing the next state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03]

    assert state.shape[0] >= 3
    assert velocity.shape[0] >= 2

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + rng.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    phi = state[2, 0]
    co_matrix = np.array([[cos(phi), 0], [sin(phi), 0], [0, 1]])
    next_state = state[0:3] + co_matrix @ real_velocity * step_time
    next_state[2, 0] = WrapToPi(next_state[2, 0])

    return next_state


def ackermann_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
    mode: str = "steer",
    wheelbase: float = 1,
) -> np.ndarray:
    """
    Calculate the next state for an Ackermann steering vehicle.

    Args:
        state: A 4x1 vector [x, y, theta, steer_angle] representing the current state.
        velocity: A 2x1 vector representing the current velocities, format depends on mode.
            For "steer" mode, [linear, steer_angle] is expected.
            For "angular" mode, [linear, angular] is expected.

        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.
        mode: The kinematic mode, either "steer" or "angular" (default "steer").
        wheelbase: The distance between the front and rear axles (default 1).

    Returns:
        new_state: A 4x1 vector representing the next state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03]

    assert state.shape[0] >= 4
    assert velocity.shape[0] >= 2

    phi = state[2, 0]
    psi = state[3, 0]

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + rng.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    if mode == "steer" or mode == "angular":
        co_matrix = np.array(
            [[cos(phi), 0], [sin(phi), 0], [tan(psi) / wheelbase, 0], [0, 1]]
        )

    d_state = co_matrix @ real_velocity
    new_state = state + d_state * step_time

    if mode == "steer":
        new_state[3, 0] = real_velocity[1, 0]

    new_state[2, 0] = WrapToPi(new_state[2, 0])

    return new_state


def omni_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Calculate the next position for an omnidirectional robot.

    Args:
        state: A 3x1 vector [x, y, theta] representing the current position.
        velocity: A 2x1 vector [vx, vy] or 3x1 vector [vx, vy, wz] representing the current velocities.
        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0.03]). alpha[0] is for x velocity, alpha[1] is for y velocity.

    Returns:
        new_position: A 2x1 vector [x, y] or 3x1 vector [x, y, theta] representing the next state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03]

    assert velocity.shape[0] >= 2
    assert state.shape[0] >= 2

    if velocity.shape[0] == 3:
        real_velocity = velocity
        if noise:
            assert len(alpha) >= 2
            std_vx = np.sqrt(alpha[0])
            std_vy = np.sqrt(alpha[-1])
            real_velocity[0, 0] += rng.normal(0, std_vx)
            real_velocity[1, 0] += rng.normal(0, std_vy)

        phi = state[2, 0]
        co_matrix = np.array(
            [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
        )
        next_state = state[0:3] + co_matrix @ real_velocity * step_time
        next_state[2, 0] = WrapToPi(next_state[2, 0])
        return next_state

    if noise:
        assert len(alpha) >= 2
        std_vx = np.sqrt(alpha[0])
        std_vy = np.sqrt(alpha[-1])
        real_velocity = velocity + rng.normal([[0], [0]], scale=[[std_vx], [std_vy]])
    else:
        real_velocity = velocity

    return state[0:2] + real_velocity * step_time


def tractor_trailer_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    noise: bool = False,
    alpha: Optional[list[float]] = None,
    mode: str = "steer",
    wheelbase: float = 1,
    trailer_length: float = 1.5,
    hitch_length: float = 1,
) -> np.ndarray:
    """
    Calculate the next state for an Tractor-Trailer (Ackermann steering) system.

    Args:
        state: A 5x1 vector [x, y, theta, phi, steer_angle] representing the current state.
            phi = (heading_trailer - heading_tractor), in radians.

        velocity: A 2x1 vector representing the current velocities, format depends on mode.
            For "steer" mode, [linear, steer_angle] is expected.
            For "angular" mode, [linear, angular] is expected.

        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.
        mode: The kinematic mode, either "steer" or "angular" (default "steer").
        wheelbase: The distance between the front and rear axles of the tractor (default 1).
        trailer_length: The length between the hitch point and the axle of the trailer (default 1.5).
        hitch_length: The length between the hitch point and the rear axle of the tractor (default 1).

    Returns:
        next_state: A 5x1 vector representing the next state.
    """
    if alpha is None:
        alpha = [0.03, 0, 0, 0.03]

    assert state.shape[0] >= 5
    assert velocity.shape[0] >= 2

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + np.random.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    theta = state[2, 0]
    phi = state[3, 0]
    psi = state[4, 0]

    if mode == "steer" or mode == "angular":
        co_matrix = np.array(
            [
                [cos(theta), 0],
                [sin(theta), 0],
                [tan(psi) / wheelbase, 0],
                [
                    -sin(phi) / trailer_length
                    - tan(psi) / wheelbase
                    - hitch_length * cos(phi) * tan(psi) / (wheelbase * trailer_length),
                    0,
                ],
                [0, 1],
            ]
        )

    d_state = co_matrix @ real_velocity
    next_state = state[0:3] + d_state * step_time

    if mode == "steer":
        next_state[4, 0] = real_velocity[1, 0]

    next_state[2, 0] = WrapToPi(next_state[2, 0])
    next_state[3, 0] = WrapToPi(next_state[3, 0])

    return next_state


def rangerminiv3_kinematics(
    state: np.ndarray, velocity: np.ndarray, step_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the next state for an Ranger Mini V3 robot.

    Args:
        state: A 3x1 vector [x, y, theta] representing the current position.
        velocity: A 3x1 vector [vx, vy, wz] representing the current velocities.
        step_time: The time step for the simulation.

    Returns:
        new_state: A 3x1 vector [x, y, theta] representing the next state.
        new_velocity: A 3x1 vector [vx, vy, wz] representing the next velocities.
    """
    track = 0.364  # m
    wheelbase = 0.494  # m

    max_linear_speed = 1.5  # m/s
    max_angular_speed = 4.8  # rad/s
    max_speed_cmd = 10.0  # m/s

    max_steer_angle_central = 0.4782  # ~=27.40 degree
    max_steer_angle_parallel = 1.570  # 180degree
    max_round_angle = 0.935671
    min_turn_radius = 0.4764

    eps = 1e-6
    
    assert state.shape[0] >= 3
    assert velocity.shape[0] >= 3
    
    def CalculateSteeringAngle(velocity):
        linear = abs(velocity[0, 0])
        angular = abs(velocity[2, 0])

        # Do not return nan radius
        if angular < eps:
            radius = float('inf')
        else:
            radius = linear / angular
        
        k = 1.0 if (velocity[2, 0] * velocity[0, 0] >= 0) else -1.0

        l = wheelbase
        # w = track
        # x = np.sqrt(radius **2 + (l / 2) **2)
        # phi_i = np.arctan((l / 2) / (x - w / 2))
        if radius == 0:
            phi = 0.0
        else:   
            phi = np.arctan((l / 2) / radius)
        return k * phi, radius

    def ConvertInnerAngleToCentral(angle):
        phi_i = abs(angle)

        phi = np.arctan(
            wheelbase
            * np.sin(phi_i)
            / (wheelbase * np.cos(phi_i) + track * np.sin(phi_i))
        )
        phi *= 1.0 if (angle >= 0) else -1.0
        return phi

    def ConvertCentralAngleToInner(angle):
        phi = abs(angle)

        phi_i = np.arctan(
            wheelbase * np.sin(phi) / (wheelbase * np.cos(phi) - track * np.sin(phi))
        )
        phi_i *= 1.0 if (angle >= 0) else -1.0
        return phi_i

    if velocity[1, 0] != 0:
        motion_mode = "MOTION_MODE_PARALLEL"
    else:
        steer_cmd, radius = CalculateSteeringAngle(velocity)
        # Use minimum turn radius to switch between dual ackerman and spinning mode
        if radius < min_turn_radius:
            motion_mode = "MOTION_MODE_SPINNING"
        else:
            motion_mode = "MOTION_MODE_DUAL_ACKERMANN"

    if motion_mode == "MOTION_MODE_DUAL_ACKERMANN":
        steer_cmd = np.clip(
            steer_cmd, -max_steer_angle_central, max_steer_angle_central
        )
        phi_i = ConvertCentralAngleToInner(steer_cmd)

        # Dual Ackermann Model
        v = velocity[0, 0]
        phi = ConvertInnerAngleToCentral(phi_i)
        dstate = np.array(
            [
                [v * np.cos(phi) * np.cos(state[2, 0])],
                [v * np.cos(phi) * np.sin(state[2, 0])],
                [2.0 * v * np.sin(phi) / wheelbase],
            ]
        )

        next_state = state[0:3] + dstate * step_time
        next_state[2, 0] = WrapToPi(next_state[2, 0])
        next_velocity = np.array([[v], [0.0], [2.0 * v * np.sin(phi) / wheelbase]])

    elif motion_mode == "MOTION_MODE_PARALLEL":
        if abs(velocity[0, 0]) < eps:
            steer_cmd = 0.0
        else:   
            steer_cmd = np.arctan(velocity[1, 0] / velocity[0, 0])

        if np.signbit(velocity[0, 0] and velocity[0, 0] == 0.0):
            steer_cmd = -steer_cmd
        else:
            steer_cmd = steer_cmd

        steer_cmd = np.clip(
            steer_cmd, -max_steer_angle_parallel, max_steer_angle_parallel
        )
        vel = 1.0 if (velocity[0, 0] > 0) else -1.0
        vel = vel * np.sqrt(velocity[0, 0] ** 2 + velocity[1, 0] ** 2)

        # Parallel Model
        v = vel
        phi = steer_cmd
        dstate = np.array(
            [
                [v * np.cos(state[2, 0] + phi)],
                [v * np.sin(state[2, 0] + phi)],
                [0.0],
            ]
        )

        next_state = state[0:3] + dstate * step_time
        next_state[2, 0] = WrapToPi(next_state[2, 0])
        next_velocity = np.array([[v * np.cos(phi)], [v * np.sin(phi)], [0.0]])

    elif motion_mode == "MOTION_MODE_SPINNING":
        a_v = velocity[2, 0]
        a_v = np.clip(a_v, -max_angular_speed, max_angular_speed)

        # Spinning Model
        w = a_v
        dstate = np.array([[0.0], [0.0], [w]])
        next_state = state[0:3] + dstate * step_time
        next_state[2, 0] = WrapToPi(next_state[2, 0])
        next_velocity = np.array([[0.0], [0.0], [w]])
    else:
        raise ValueError(f"Invalid motion mode: {motion_mode}")

    return next_state, next_velocity
