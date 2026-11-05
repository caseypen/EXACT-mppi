from typing import Tuple, Union
import jax
from jax import numpy as jnp

from ..models import *
from ..critics import CriticData


def normalize_angles(angles: jax.Array) -> jax.Array:
    """
    Normalizes the angle to be -M_PIF circle to +M_PIF circle
    It takes and returns radians.

    Args:
        angles:s Angles to normalize

    Returns:
        jax.Array: normalized angles
    """

    return (angles + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def shortest_angular_distance(from_angle: jax.Array, to_angle: jax.Array) -> jax.Array:
    """
    Given 2 angles, this returns the shortest angular
    difference.  The inputs and outputs are of course radians.

    The result
    would always be -pi <= result <= pi.  Adding the result
    to "from" will always get you an equivalent angle to "to".

    Args:
        from_angle: Start angle
        to_angle: End angle

    Returns:
        jax.Array: Shortest distance between angles
    """

    return normalize_angles(to_angle - from_angle)


def savitskyGolayFilter(
    control_sequence: ControlSequence,
    control_history: jax.Array,  # (4, 3)
    shift_control_sequence: Union[bool, jax.Array],
) -> Tuple[ControlSequence, jax.Array]:
    """
    Apply Savisky-Golay filter to optimal trajectory

    Args:
        control_sequence: Sequence to apply filter to
        control_history: Recent set of controls for edge-case handling
        shift_control_sequence: whether to shift the control sequence

    Returns:
        ControlSequence: filtered control sequence
        jax.Array: updated control history
    """

    # Savitzky-Golay Quadratic, 9-point Coefficients
    filter_coeffs = (
        jnp.array(
            [-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0],
        )
        / 231.0
    )

    num_sequences = control_sequence.vx.shape[0] - 1

    def do_nothing(_):
        # Too short to smmoth meaningfully
        return control_sequence, control_history

    def apply_filter(_):
        def applyFilterOverAxis(sequence, hist4):
            num_sequences = sequence.shape[0] - 1
            pt_last = sequence[-1]
            padded = jnp.concatenate(
                [hist4, sequence, jnp.full((4,), pt_last)],
                axis=0,
            )  # (T+8,)

            idxs = jnp.arange(num_sequences, dtype=jnp.int32)

            def applyFilter(idx):
                data = jax.lax.dynamic_slice(padded, (idx,), (9,))
                return jnp.sum(data * filter_coeffs)

            filtered_data = jax.vmap(applyFilter)(idxs)  # (T-1,)
            return sequence.at[:num_sequences].set(filtered_data)

        # Filter trajectories
        vx_filtered = applyFilterOverAxis(control_sequence.vx, control_history[:, 0])
        vy_filtered = applyFilterOverAxis(control_sequence.vy, control_history[:, 1])
        wz_filtered = applyFilterOverAxis(control_sequence.wz, control_history[:, 2])

        control_sequence_filtered = ControlSequence(
            vx=vx_filtered, vy=vy_filtered, wz=wz_filtered
        )

        # Update control history
        shift_control_sequence_ = jnp.asarray(
            shift_control_sequence, dtype=jnp.bool_
        ).reshape(())
        offset = jnp.where(shift_control_sequence_, jnp.int32(1), jnp.int32(0))
        new_control = jnp.stack(
            [
                control_sequence_filtered.vx[offset],
                control_sequence_filtered.vy[offset],
                control_sequence_filtered.wz[offset],
            ],
            axis=0,
        )  # (3,)

        control_history_updated = jnp.concatenate(
            [control_history[1:], new_control[None, :]], axis=0
        )  # (4, 3)

        return control_sequence_filtered, control_history_updated

    return jax.lax.cond(num_sequences < 20, do_nothing, apply_filter, operand=None)


def calculate_path_length(path: jax.Array) -> jax.Array:
    """
    Calculate the length of a path

    Args:
        path: Path to calculate the length of

    Returns:
        jax.Array: Length of the path
    """

    return jnp.linalg.norm(path[1:] - path[:-1], axis=-1).sum()


def findPathFurthestReachedPoint(
    data: CriticData, max_lookahead_distance: float = 5.0
) -> jax.Array:
    """
    Evaluate furthest point idx of data.path which is
    nearest to some trajectory in data.trajectories

    Args:
        data: Data to use

    Returns:
        jax.Array: Idx of furthest path point reached by a set of trajectories
    """

    traj_x = data.trajectories.x[:, -1]  # (K,)
    traj_y = data.trajectories.y[:, -1]  # (K,)

    path_x = data.path.x
    path_y = data.path.y

    # The path handler supplies a pruned local path whose first point is the
    # current path projection. Endpoints can be spatially close to a later
    # branch on a loop or self-intersection, so only search the bounded forward
    # arc from that projection.
    path_step = jnp.sqrt(jnp.diff(path_x) ** 2 + jnp.diff(path_y) ** 2)
    path_distance = jnp.concatenate(
        [jnp.zeros((1,), dtype=path_x.dtype), jnp.cumsum(path_step)]
    )
    in_forward_window = path_distance <= jnp.asarray(
        max_lookahead_distance, dtype=path_distance.dtype
    )

    dx = path_x[None, :] - traj_x[:, None]  # (K, P)
    dy = path_y[None, :] - traj_y[:, None]  # (K, P)
    dist2 = dx * dx + dy * dy
    dist2 = jnp.where(in_forward_window[None, :], dist2, jnp.inf)

    nearest_idx = jnp.argmin(dist2, axis=1)  # (K,)
    return jnp.max(nearest_idx).astype(jnp.int32)


def findClosestPathPt(
    vec: jax.Array,
    dist: Union[float, jax.Array],
    init: Union[int, jax.Array],
) -> jax.Array:
    """
    Compare to trajectory points to find closest path point along integrated distances

    Args:
        vec: Vect to check
        dist: Distance to look for
        init: Starting index to indec from

    Returns:
        jax.Array: Index of closest path point
    """

    dist = jnp.asarray(dist, dtype=jnp.float32).reshape(())
    init = jnp.asarray(init, dtype=jnp.int32).reshape(())

    # First is 0, no accumulated distance yet
    distm1 = jnp.where(init != 0, vec[init], jnp.array(0.0))
    size_ = jnp.int32(vec.shape[0])

    def cond_fun(val):
        i, distim1, done, result = val
        return (~done) & (i < size_)

    def body_fun(val):
        i, distim1, done, result = val
        disti = vec[i]
        crossed = disti > dist

        choose_prev = (i > 0) & (dist - distim1 < disti - dist)
        chosen_idx = jnp.where(choose_prev, i - 1, i)

        new_distim1 = jnp.where(crossed, distim1, disti)
        new_done = crossed
        new_result = jnp.where(crossed, chosen_idx, result)

        return (i + 1, new_distim1, new_done, new_result)

    init_val = (
        init + 1,
        distm1,
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
    )

    _, _, done, result = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return jnp.where(done, result, size_ - 1).astype(jnp.int32)


def posePointAngleXY(
    pose: jax.Array,
    point_x: Union[float, jax.Array],
    point_y: Union[float, jax.Array],
    forward_preference: Union[bool, jax.Array],
) -> jax.Array:
    """
    Evaluate angle from pose (have angle) to point (no angle)

    Args:
        pose: Pose
        point_x: Point to find angle relative to X axis
        point_y: Point to find angle relative to Y axis
        forward_preference: If reversing direction is valid

    Returns:
        jax.Array: Angle between two points
    """

    pose_x = pose[0]
    pose_y = pose[1]
    pose_yaw = pose[2]

    yaw = jnp.arctan2(point_y - pose_y, point_x - pose_x)

    return jax.lax.cond(
        jnp.asarray(forward_preference, dtype=jnp.bool_).reshape(()),
        lambda _: jnp.abs(shortest_angular_distance(yaw, pose_yaw)),
        lambda _: jnp.minimum(
            jnp.abs(shortest_angular_distance(yaw, pose_yaw)),
            jnp.abs(
                shortest_angular_distance(yaw, normalize_angles(pose_yaw + jnp.pi))
            ),
        ),
        operand=None,
    )


def posePointAngleXYYAW(
    pose: jax.Array,
    point_x: Union[float, jax.Array],
    point_y: Union[float, jax.Array],
    point_yaw: Union[float, jax.Array],
) -> jax.Array:
    """
    Evaluate angle from pose (have angle) to point (no angle)

    Args:
        pose: Pose
        point_x: Point to find angle relative to X axis
        point_y: Point to find angle relative to Y axis
        point_yaw: Yaw of the point to consider along Z axis

    Returns:
        jax.Array: Angle between two points
    """

    pose_x = pose[0]
    pose_y = pose[1]
    pose_yaw = pose[2]

    yaw = jnp.arctan2(point_y - pose_y, point_x - pose_x)

    yaw = jnp.where(
        jnp.abs(shortest_angular_distance(yaw, point_yaw)) > jnp.pi / 2.0,
        normalize_angles(yaw + jnp.pi),
        yaw,
    )

    return jnp.abs(shortest_angular_distance(yaw, pose_yaw))


def normalize_yaws_between_points(
    last_yaws: jax.Array, yaw_between_points: jax.Array
) -> jax.Array:
    """
    Normalize the yaws between points on the basis of final yaw angle of the trajectory.

    Args:
        last_yaws: Final yaw angles of the trajectories
        yaw_between_points: Yaw angles calculated between x and y coordinates of the trajectories.

    Returns:
        jax.Array: Normalized yaw between points.

    """

    yaws = jnp.abs(shortest_angular_distance(last_yaws, yaw_between_points))
    yaws_between_points_corrected = jnp.where(
        yaws < (jnp.pi / 2.0),
        yaw_between_points,
        normalize_angles(yaw_between_points + jnp.pi),
    )
    return yaws_between_points_corrected
