"""Point-flow footprint distances for the dynamic corridor experiment.

The helper is deliberately independent of the MPPI rollout integrator: it uses
the constrained interval velocities already stored in ``State`` to insert
collision-only samples between normal rollout endpoints.
"""

import jax
import jax.numpy as jnp

from .signed_distance import sdf_rect
from .signed_distance_polygon import sdf_polygon


def _internal_poses(trajectory_x, trajectory_y, trajectory_yaws, vx, wz, pose, dt, substeps):
    """Return differential-drive poses at m/substeps of each rollout interval."""
    k, t = trajectory_x.shape
    # Preserve the existing MPPI endpoint convention exactly in the baseline
    # mode. Its XY integrator is forward Euler, whereas the internal samples
    # below intentionally use the requested exact constant-velocity arc.
    if substeps == 1:
        return jnp.stack([trajectory_x, trajectory_y, trajectory_yaws], axis=-1)[:, :, None, :]
    tau = dt * jnp.arange(1, substeps + 1, dtype=trajectory_x.dtype) / substeps
    starts_x = jnp.concatenate(
        [jnp.full((k, 1), pose[0], dtype=trajectory_x.dtype), trajectory_x[:, :-1]], axis=1
    )
    starts_y = jnp.concatenate(
        [jnp.full((k, 1), pose[1], dtype=trajectory_y.dtype), trajectory_y[:, :-1]], axis=1
    )
    starts_yaw = jnp.concatenate(
        [jnp.full((k, 1), pose[2], dtype=trajectory_yaws.dtype), trajectory_yaws[:, :-1]], axis=1
    )

    theta = starts_yaw[:, :, None] + wz[:, :, None] * tau[None, None, :]
    turning_wz = jnp.abs(wz) > 1.0e-6
    # `jnp.where` evaluates both branches during tracing. Divide by a safe
    # denominator so the unused arc branch cannot create inf/nan at omega=0.
    safe_wz = jnp.where(turning_wz, wz, 1.0)
    v_over_w = jnp.where(turning_wz, vx / safe_wz, 0.0)[:, :, None]
    x_arc = starts_x[:, :, None] + v_over_w * (jnp.sin(theta) - jnp.sin(starts_yaw[:, :, None]))
    y_arc = starts_y[:, :, None] - v_over_w * (jnp.cos(theta) - jnp.cos(starts_yaw[:, :, None]))
    x_line = starts_x[:, :, None] + vx[:, :, None] * tau[None, None, :] * jnp.cos(starts_yaw[:, :, None])
    y_line = starts_y[:, :, None] + vx[:, :, None] * tau[None, None, :] * jnp.sin(starts_yaw[:, :, None])
    turning = turning_wz[:, :, None]
    return jnp.stack([jnp.where(turning, x_arc, x_line), jnp.where(turning, y_arc, y_line), theta], axis=-1)


def _rectangle_distance(points_body, centers, halfs):
    def process_rectangle(carry, rectangle):
        center, half = rectangle
        return jnp.minimum(carry, sdf_rect(points_body - center, half)), None

    initial = jnp.full(points_body.shape[:-1], jnp.inf, dtype=points_body.dtype)
    distances, _ = jax.lax.scan(process_rectangle, initial, (centers, halfs))
    return distances


def _polygon_distance(points_body, vertices, vertex_counts):
    def process_polygon(carry, polygon):
        polygon_vertices, count = polygon
        return jnp.minimum(carry, sdf_polygon(points_body, polygon_vertices, count)), None

    initial = jnp.full(points_body.shape[:-1], jnp.inf, dtype=points_body.dtype)
    distances, _ = jax.lax.scan(process_polygon, initial, (vertices, vertex_counts))
    return distances


def minimum_signed_distance_from_point_flow(
    trajectory_x,
    trajectory_y,
    trajectory_yaws,
    interval_vx,
    interval_wz,
    initial_pose,
    points,
    velocities,
    mask,
    model_dt,
    substeps,
    point_chunk_size,
    *,
    rect_centers=None,
    rect_halfs=None,
    vertices=None,
    vertex_counts=None,
):
    """Return `(K, T)` minimum SDFs using a shared fixed-frame point flow.

    ``substeps`` and ``point_chunk_size`` are Python integers captured by the
    obstacle critic's JIT-specialized closure. Points are predicted once for
    all `(T * substeps)` timestamps and never replicated over rollouts.
    """
    k, t = trajectory_x.shape
    poses = _internal_poses(
        trajectory_x, trajectory_y, trajectory_yaws, interval_vx, interval_wz,
        initial_pose, model_dt, substeps,
    ).reshape(k, t * substeps, 3)
    point_padding = (-points.shape[0]) % point_chunk_size
    if point_padding:
        points = jnp.pad(points, ((0, point_padding), (0, 0)))
        velocities = jnp.pad(velocities, ((0, point_padding), (0, 0)))
        mask = jnp.pad(mask, (0, point_padding))
    times = (
        model_dt * (jnp.arange(t, dtype=points.dtype)[:, None]
        + jnp.arange(1, substeps + 1, dtype=points.dtype)[None, :] / substeps)
    ).reshape(t * substeps)
    points_t = points[None, :, :] + times[:, None, None] * velocities[None, :, :]

    minimum = jnp.full((k, t * substeps), jnp.inf, dtype=points.dtype)
    for start in range(0, points.shape[0], point_chunk_size):
        point_chunk = points_t[:, start : start + point_chunk_size, :]
        mask_chunk = mask[start : start + point_chunk_size]
        dx = point_chunk[None, :, :, 0] - poses[:, :, None, 0]
        dy = point_chunk[None, :, :, 1] - poses[:, :, None, 1]
        cos_yaw = jnp.cos(poses[:, :, 2])[:, :, None]
        sin_yaw = jnp.sin(poses[:, :, 2])[:, :, None]
        points_body = jnp.stack(
            [dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw], axis=-1
        )
        if rect_centers is not None:
            distances = _rectangle_distance(points_body, rect_centers, rect_halfs)
        else:
            distances = _polygon_distance(points_body, vertices, vertex_counts)
        distances = jnp.where(mask_chunk[None, None, :] > 0.5, distances, jnp.inf)
        minimum = jnp.minimum(minimum, jnp.min(distances, axis=-1))

    return jnp.min(minimum.reshape(k, t, substeps), axis=-1)