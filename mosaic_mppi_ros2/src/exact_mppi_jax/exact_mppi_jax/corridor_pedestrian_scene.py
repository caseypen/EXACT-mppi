"""Generate randomized LIMO corridor pedestrian scenes for Gazebo."""

from __future__ import annotations

import math
import os
import random
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import yaml


Point2 = Tuple[float, float]

BASE_WORLD = "mppi_corridor_dynamic.world"
CORRIDOR_Y_MIN = 7.55
CORRIDOR_Y_MAX = 11.45
SIDEWALK_WIDTH = 0.9
SIDEWALK_X_DRIFT = 3.0
CORRIDOR_Y_RANGE = (7.75, 11.25)
STATIC_X_RANGE = (8.8, 19.2)
SIDEWALK_X_CENTER_RANGE = (6.0, 21.0)
MIN_SIDEWALK_CENTER_DISTANCE = 4.0
STATIC_COUNT = 8
MIN_PEDESTRIAN_DISTANCE = 0.8
MIN_STATIC_TO_DYNAMIC_DISTANCE = 0.9


@dataclass(frozen=True)
class ScenePaths:
    world_path: str
    dynamic_config_path: str
    seed: int


@dataclass(frozen=True)
class DynamicPedestrian:
    name: str
    start: Point2
    end: Point2
    speed: float
    start_delay: float


@dataclass(frozen=True)
class SidewalkSegment:
    name: str
    start: Point2
    end: Point2

    @property
    def center(self) -> Point2:
        return (
            (self.start[0] + self.end[0]) * 0.5,
            (self.start[1] + self.end[1]) * 0.5,
        )

    @property
    def length(self) -> float:
        return _distance(self.start, self.end)

    @property
    def yaw(self) -> float:
        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])

    def point(self, t: float, lateral_offset: float) -> Point2:
        direction = (self.end[0] - self.start[0], self.end[1] - self.start[1])
        length = math.hypot(direction[0], direction[1])
        unit = (direction[0] / length, direction[1] / length)
        normal = (-unit[1], unit[0])
        return (
            self.start[0] + direction[0] * t + normal[0] * lateral_offset,
            self.start[1] + direction[1] * t + normal[1] * lateral_offset,
        )


def generate_corridor_pedestrian_scene(pkg_dir: str, seed_text: str = "") -> ScenePaths:
    """Generate a temporary world and dynamic obstacle config for one launch."""

    seed = _resolve_seed(seed_text)
    rng = random.Random(seed)

    sidewalks = _sample_sidewalks(rng)
    dynamic_pedestrians = _sample_dynamic_pedestrians(rng, sidewalks)
    static_points = _sample_static_pedestrians(rng, dynamic_pedestrians, sidewalks)

    temp_dir = tempfile.mkdtemp(prefix=f"mppi_corridor_pedestrians_{seed}_")
    world_path = os.path.join(temp_dir, "mppi_corridor_pedestrians.world")
    dynamic_config_path = os.path.join(temp_dir, "mppi_corridor_dynamic.yaml")

    base_world_path = os.path.join(pkg_dir, "worlds", BASE_WORLD)
    _write_world(base_world_path, world_path, static_points, dynamic_pedestrians, sidewalks)
    _write_dynamic_config(dynamic_config_path, dynamic_pedestrians)

    return ScenePaths(world_path=world_path, dynamic_config_path=dynamic_config_path, seed=seed)


def _resolve_seed(seed_text: str) -> int:
    stripped = str(seed_text).strip()
    if stripped:
        return int(stripped, 0)
    return random.SystemRandom().randrange(0, 2**32)


def _sample_sidewalks(rng: random.Random) -> List[SidewalkSegment]:
    max_attempts = 200
    names = ("diagonal_sidewalk_start", "diagonal_sidewalk_goal")

    for _ in range(max_attempts):
        centers = [rng.uniform(*SIDEWALK_X_CENTER_RANGE) for _ in names]
        if abs(centers[0] - centers[1]) < MIN_SIDEWALK_CENTER_DISTANCE:
            continue
        return [
            _make_sidewalk(name, center_x)
            for name, center_x in zip(names, centers)
        ]

    raise RuntimeError(
        "Could not place diagonal sidewalks with "
        f"{MIN_SIDEWALK_CENTER_DISTANCE:.1f}m minimum center spacing after "
        f"{max_attempts} attempts"
    )


def _make_sidewalk(name: str, center_x: float) -> SidewalkSegment:
    return SidewalkSegment(
        name=name,
        start=(center_x - SIDEWALK_X_DRIFT * 0.5, CORRIDOR_Y_MIN),
        end=(center_x + SIDEWALK_X_DRIFT * 0.5, CORRIDOR_Y_MAX),
    )


def _sample_dynamic_pedestrians(
    rng: random.Random,
    sidewalks: Iterable[SidewalkSegment],
) -> List[DynamicPedestrian]:
    pedestrians: List[DynamicPedestrian] = []
    for index, sidewalk in enumerate(sidewalks):
        span = rng.uniform(0.28, 0.45)
        start_t = rng.uniform(0.10, 0.90 - span)
        end_t = start_t + span
        lateral = rng.uniform(-0.18, 0.18)
        start = sidewalk.point(start_t, lateral)
        end = sidewalk.point(end_t, lateral)
        if rng.random() < 0.5:
            start, end = end, start
        pedestrians.append(
            DynamicPedestrian(
                name=f"dynamic_pedestrian_{index:02d}",
                start=start,
                end=end,
                speed=rng.uniform(0.1, 0.11),
                start_delay=0.4 * index,
            )
        )
    return pedestrians


def _sample_static_pedestrians(
    rng: random.Random,
    dynamic_pedestrians: Iterable[DynamicPedestrian],
    sidewalks: Iterable[SidewalkSegment],
) -> List[Point2]:
    dynamic_segments = [(ped.start, ped.end) for ped in dynamic_pedestrians]
    sidewalk_segments = list(sidewalks)
    points: List[Point2] = []
    max_attempts = 4000

    for _ in range(max_attempts):
        if len(points) >= STATIC_COUNT:
            return points

        candidate = (
            rng.uniform(*STATIC_X_RANGE),
            rng.uniform(*CORRIDOR_Y_RANGE),
        )
        if (
            _distance_to_any_sidewalk(candidate, sidewalk_segments)
            < SIDEWALK_WIDTH * 0.5 + 0.25
        ):
            continue
        if any(_distance(candidate, point) < MIN_PEDESTRIAN_DISTANCE for point in points):
            continue
        if any(
            _point_to_segment_distance(candidate, start, end) < MIN_STATIC_TO_DYNAMIC_DISTANCE
            for start, end in dynamic_segments
        ):
            continue
        points.append(candidate)

    raise RuntimeError(
        f"Could not place {STATIC_COUNT} static pedestrians after {max_attempts} attempts"
    )


def _write_world(
    base_world_path: str,
    output_path: str,
    static_points: List[Point2],
    dynamic_pedestrians: List[DynamicPedestrian],
    sidewalks: List[SidewalkSegment],
) -> None:
    tree = ET.parse(base_world_path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise RuntimeError(f"No <world> element found in {base_world_path}")

    for include in list(world.findall("include")):
        name = include.findtext("name", "")
        if name.startswith("cross_clone_"):
            world.remove(include)

    for sidewalk in sidewalks:
        _append_sidewalk(world, sidewalk)
    for index, point in enumerate(static_points):
        _append_pedestrian_include(
            world,
            name=f"static_pedestrian_{index:02d}",
            point=point,
            yaw=random_yaw_from_point(point),
        )
    for pedestrian in dynamic_pedestrians:
        _append_pedestrian_include(
            world,
            name=pedestrian.name,
            point=pedestrian.start,
            yaw=math.atan2(
                pedestrian.end[1] - pedestrian.start[1],
                pedestrian.end[0] - pedestrian.start[0],
            ),
        )

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def _write_dynamic_config(output_path: str, dynamic_pedestrians: List[DynamicPedestrian]) -> None:
    config = {
        "dynamic_obstacles": {
            "update_rate": 30.0,
            "reference_frame": "world",
            "default_z": 0.0,
            "default_yaw": 0.0,
            "start_paused": True,
            "start_topic": "/start_dynamic_obstacles",
            "reset_on_start": True,
            "service_names": ["/gazebo/set_entity_state", "/set_entity_state"],
            "obstacles": [
                {
                    "name": pedestrian.name,
                    "type": "linear",
                    "speed": round(pedestrian.speed, 3),
                    "loop": "ping_pong",
                    "yaw_mode": "tangent",
                    "start_delay": round(pedestrian.start_delay, 3),
                    "start": [_round(pedestrian.start[0]), _round(pedestrian.start[1]), 0.0],
                    "end": [_round(pedestrian.end[0]), _round(pedestrian.end[1]), 0.0],
                }
                for pedestrian in dynamic_pedestrians
            ],
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _append_sidewalk(world: ET.Element, sidewalk: SidewalkSegment) -> None:
    center = sidewalk.center
    model = ET.SubElement(world, "model", {"name": sidewalk.name})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = (
        f"{_round(center[0])} {_round(center[1])} 0.011 0 0 {_round(sidewalk.yaw)}"
    )
    link = ET.SubElement(model, "link", {"name": "link"})
    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(ET.SubElement(geometry, "box"), "size").text = (
        f"{_round(sidewalk.length)} {SIDEWALK_WIDTH:.3f} 0.02"
    )
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.78 0.78 0.72 0.55"
    ET.SubElement(material, "diffuse").text = "0.78 0.78 0.72 0.55"


def _append_pedestrian_include(
    world: ET.Element,
    name: str,
    point: Point2,
    yaw: float,
) -> None:
    include = ET.SubElement(world, "include")
    ET.SubElement(include, "uri").text = "model://pedestrian_obstacle"
    ET.SubElement(include, "name").text = name
    ET.SubElement(include, "pose").text = (
        f"{_round(point[0])} {_round(point[1])} 0 0 0 {_round(yaw)}"
    )


def _distance_to_any_sidewalk(
    point: Point2,
    sidewalks: Iterable[SidewalkSegment],
) -> float:
    return min(
        _point_to_segment_distance(point, sidewalk.start, sidewalk.end)
        for sidewalk in sidewalks
    )


def _point_to_segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return _distance(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    projection = (start[0] + t * dx, start[1] + t * dy)
    return _distance(point, projection)


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def random_yaw_from_point(point: Point2) -> float:
    # Deterministic visual variety tied to placement, independent of sample order.
    return math.atan2(math.sin(point[0] * 1.7), math.cos(point[1] * 1.3))


def _round(value: float) -> float:
    return round(float(value), 4)
