"""
Reference path generation for MPPI trajectory tracking.

Adapted from neupan InitialPath to provide reference trajectory generation
for MPPI cost function, matching NRMP's interface.
"""

import numpy as np
from math import tan, inf, cos, sin, sqrt
import math
from typing import Union, List, Optional
from .path_search import PathSearch

from pathfinding.core.grid import Grid

# Import curve generator from gctl (same as neupan)
try:
    from gctl import curve_generator
    HAS_GCTL = True
except ImportError:
    HAS_GCTL = False
    print("Warning: gctl not available, only line paths will work")


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    if isinstance(angle, (np.ndarray, float, int)):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)


class ReferencePath:
    """
    Generate reference path for MPPI trajectory tracking.
    
    Adapted from neupan's InitialPath to provide nom_s, ref_s, ref_us
    for MPPI cost calculation, matching NRMP's interface.
    """
    
    def __init__(
        self,
        receding: int,
        step_time: float,
        ref_speed: float,
        robot,
        curve_style: str = "line",
        grid_map: Optional[Grid] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            receding: Planning horizon steps
            step_time: Time step (dt)
            ref_speed: Reference speed (m/s)
            robot: RobotConfig object
            curve_style: 'line', 'dubins', 'reeds_shepp', 'astar'
            grid_map: Optional Grid object for A* pathfinding
        """
        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed
        self.robot = robot
        self.curve_style = curve_style
        self.min_radius = kwargs.get("min_radius", self.default_turn_radius())
        self.interval = kwargs.get("interval", self.dt * self.ref_speed)
        self.arrive_threshold = kwargs.get("arrive_threshold", 0.1)
        self.close_threshold = kwargs.get("close_threshold", 0.1)
        self.ind_range = kwargs.get("ind_range", 10)
        
        # Path state
        self.initial_path = None
        self.curve_list = None
        self.curve_index = 0
        self.point_index = 0
        
        # Initialize curve generator (same as neupan)
        if HAS_GCTL and curve_style in ["line", "dubins", "reeds_shepp"]:
            self.cg = curve_generator()
        else:
            self.cg = None

        # Initialize path searcher if grid_map provided
        if grid_map is not None:
            # Read grid parameters from kwargs or use default values
            grid_resolution = kwargs.get("grid_resolution", 0.5)
            grid_origin = kwargs.get("grid_origin", [0.0, 0.0])
            self.path_searcher = PathSearch(
                grid_map, 
                resolution=grid_resolution, 
                origin=grid_origin,
                curve_style=self.curve_style,
                vehicle_polygons=robot.vertices_list
            )
        else:
            self.path_searcher = None
        
    def set_path_from_points(
        self,
        start: Union[np.ndarray, List],
        goal: Union[np.ndarray, List],
    ):
        """
        Set path from start to goal.
        
        Args:
            start: [x, y, yaw] starting pose
            goal: [x, y, yaw] goal pose
        """
        start = np.asarray(start).flatten()[:3]
        goal = np.asarray(goal).flatten()[:3]
        
        # Convert to column vectors for compatibility
        start_vec = start.reshape(3, 1)
        goal_vec = goal.reshape(3, 1)
        
        waypoints = [start_vec, goal_vec]
        
        # Use curve generator for all path types (same as neupan)
        if self.cg is not None:
            print(f"[ref_path] using gctl curve generator: style={self.curve_style}")
            self.initial_path = self.cg.generate_curve(
                self.curve_style, waypoints, self.interval, self.min_radius, True
            )
            
            # Ensure consistent angles for line curve
            if self.curve_style == 'line':
                self._ensure_consistent_angles()
        elif self.path_searcher is not None:
            print(f"[ref_path] using grid path searcher: style={self.curve_style}")
            # Use A* path search
            grid_path, runs = self.path_searcher.find_initial_path(start, goal)
            
            if grid_path and len(grid_path) > 0:
                print(f"✓ A* path found: {len(grid_path)} grid points, {runs} iterations")
                
                self.initial_path = self.path_searcher.path_to_world_coords(
                    grid_path, 
                    interval=self.interval 
                )
            else:
                print("Warning: grid path search failed, falling back to line path")
                self.initial_path = self._generate_line_path(waypoints)
        else:
            # No curve generator (e.g., gctl missing) and no grid map for A*.
            # Ensure we still produce a valid path for downstream costs/visualization.
            print(
                "Warning: no curve generator or grid map available; "
                "falling back to line path"
            )
            self.initial_path = self._generate_line_path(waypoints)
        
        self.split_path_with_gear()
        self.curve_index = 0
        self.point_index = 0

    def _generate_line_path(
        self,
        waypoints: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Generate straight line path between waypoints.
        
        Args:
            waypoints: List of (3, 1) arrays [x, y, yaw]
            
        Returns:
            path: List of (4, 1) arrays [x, y, yaw, gear]
        """
        path = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i].flatten()
            end = waypoints[i + 1].flatten()
            
            # Calculate number of points based on interval
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = np.sqrt(dx**2 + dy**2)
            
            num_points = max(int(segment_length / self.interval), 2)
            
            # Linear interpolation for x, y
            x_interp = np.linspace(start[0], end[0], num_points)
            y_interp = np.linspace(start[1], end[1], num_points)
            
            # Calculate heading from line direction
            path_yaw = np.arctan2(dy, dx)
            
            # Interpolate yaw
            yaw_diff = wrap_to_pi(end[2] - start[2])
            
            for j in range(num_points):
                alpha = j / (num_points - 1)
                
                # Blend yaw from path direction to goal yaw
                if alpha < 0.7:
                    yaw = path_yaw
                else:
                    blend = (alpha - 0.7) / 0.3
                    yaw = path_yaw + blend * wrap_to_pi(end[2] - path_yaw)
                
                # Forward gear (1.0)
                point = np.array([[x_interp[j]], [y_interp[j]], [yaw], [1.0]])
                path.append(point)
        
        return path
    
    def generate_nom_ref_state(
        self,
        state: np.ndarray,
        cur_vel_array: np.ndarray,
        ref_speed: float
    ):
        """
        Generate nominal and reference states for MPPI cost.
        
        This matches NRMP's interface exactly.
        
        Args:
            state: Current state (3, 1) or (5, 1) - [x, y, yaw, ...]
            cur_vel_array: Current velocity array (2, T) - [v, delta]
            ref_speed: Reference speed (m/s)
            
        Returns:
            nom_s: Nominal states (3, T+1) - predicted states
            nom_u: Nominal control (2, T) - same as cur_vel_array
            ref_s: Reference states (3, T+1) - reference from path
            ref_us: Reference speeds (T,) - speed along path
        """
        state = state[:3]
        
        if self.initial_path is None:
            # No path set, return current state repeated
            nom_s = np.tile(state, (1, self.T + 1))
            nom_u = cur_vel_array
            ref_s = nom_s.copy()
            ref_us = np.full(self.T, ref_speed)
            return nom_s, nom_u, ref_s, ref_us
        
        ref_state = self.cur_point[0:3].copy()
        ref_index = self.point_index
        pre_state = state.copy()
        
        state_pre_list = [pre_state]
        state_ref_list = [ref_state]
        state_ref_list[0][2, 0] = pre_state[2, 0] + wrap_to_pi(ref_state[2, 0] - pre_state[2, 0])
        
        assert self.cur_point.shape[0] >= 4
        gear_list = [self.cur_point[-1, 0]] * self.T
        
        ref_speed_forward = ref_speed * self.dt
        
        for t in range(self.T):
            # Predict next state using motion model
            pre_state = self.motion_predict_model(
                pre_state, cur_vel_array[:, t : t + 1], self.robot.wheelbase, self.dt
            )
            state_pre_list.append(pre_state)
            
            # Find reference point along path
            if ref_speed_forward >= self.interval:
                inc_index = int(ref_speed_forward / self.interval)
                ref_index = ref_index + inc_index
                
                if ref_index > len(self.cur_curve) - 1:
                    ref_index = len(self.cur_curve) - 1
                    gear_list[t] = 0
                
                ref_state = self.cur_curve[ref_index][0:3]
            else:
                ref_state, ref_index = self.find_interaction_point(
                    ref_state, ref_index, ref_speed_forward
                )
                
                if ref_index > len(self.cur_curve) - 1:
                    gear_list[t] = 0
            
            diff = ref_state[2, 0] - pre_state[2, 0]
            ref_state[2, 0] = pre_state[2, 0] + wrap_to_pi(diff)
            state_ref_list.append(ref_state)
        
        nom_s = np.hstack(state_pre_list)  # (3, T+1)
        nom_u = cur_vel_array  # (2, T)
        ref_s = np.hstack(state_ref_list)  # (3, T+1)
        
        gear_array = np.array(gear_list)
        ref_us = gear_array * ref_speed  # (T,)
        
        return nom_s, nom_u, ref_s, ref_us
    
    def find_interaction_point(self, ref_state, ref_index, length):
        """Find interaction point between circle and path segment."""
        circle = np.squeeze(ref_state[0:2])
        
        while True:
            if ref_index > len(self.cur_curve) - 2:
                end_point = self.cur_curve[-1]
                end_point[2] = wrap_to_pi(end_point[2])
                return end_point[0:3], ref_index
            
            cur_point = self.cur_curve[ref_index]
            next_point = self.cur_curve[ref_index + 1]
            segment = [np.squeeze(cur_point[0:2]), np.squeeze(next_point[0:2])]
            interaction_point = self.range_cir_seg(circle, length, segment)
            
            if interaction_point is not None:
                diff = wrap_to_pi(next_point[2, 0] - cur_point[2, 0])
                theta = wrap_to_pi(cur_point[2, 0] + diff / 2)
                state_ref = np.append(interaction_point, theta).reshape((3, 1))
                return state_ref, ref_index
            else:
                ref_index += 1
    
    def range_cir_seg(self, circle, r, segment):
        """Find intersection point between circle and line segment."""
        assert (
            circle.shape == (2,)
            and segment[0].shape == (2,)
            and segment[1].shape == (2,)
        )
        
        sp = segment[0]
        ep = segment[1]
        d = ep - sp
        
        if np.linalg.norm(d) == 0:
            return None
        
        f = sp - circle
        a = d @ d
        b = 2 * f @ d
        c = f @ f - r**2
        
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            return None
        else:
            t2 = (-b + sqrt(discriminant)) / (2 * a)
            
            if t2 >= 0 and t2 <= 1:
                int_point = sp + t2 * d
                return int_point
            
            return None
    
    def closest_point(self, state, threshold=0.1, ind_range=10):
        """Find closest point on path to current state."""
        min_dis = inf
        cur_index = self.point_index
        
        start = max(cur_index, 0)
        end = min(cur_index + ind_range, len(self.cur_curve))
        
        for index in range(start, end):
            dis = distance(state[0:2], self.cur_curve[index][0:2])
            
            if dis < min_dis:
                min_dis = dis
                self.point_index = index
                if dis < threshold:
                    break
        
        return min_dis
    
    def split_path_with_gear(self):
        """Split path into curves by gear (forward/backward)."""
        if self.initial_path is None:
            return
        
        self.curve_list = []
        current_curve = []
        current_gear = self.initial_path[0][-1]
        
        for point in self.initial_path:
            if point[-1] != current_gear:
                self.curve_list.append(current_curve)
                current_curve = []
                current_gear = point[-1]
            
            current_curve.append(point)
        
        # Append the last curve
        if current_curve:
            self.curve_list.append(current_curve)
    
    def motion_predict_model(self, robot_state, vel, wheel_base, sample_time):
        """Predict next state using kinematic model."""
        if hasattr(self.robot, 'kinematics') and self.robot.kinematics == "diff":
            return self.diff_model(robot_state, vel, sample_time)
        else:
            return self.ackermann_model(robot_state, vel, wheel_base, sample_time)
    
    def ackermann_model(self, car_state, vel, wheel_base, sample_time):
        """Ackermann kinematic model."""
        assert car_state.shape == (3, 1) and vel.shape == (2, 1)
        
        phi = car_state[2, 0]
        v = vel[0, 0]
        psi = vel[1, 0]
        
        ds = np.array([[v * cos(phi)], [v * sin(phi)], [v * tan(psi) / wheel_base]])
        next_state = car_state + ds * sample_time
        
        return next_state
    
    def diff_model(self, robot_state, vel, sample_time):
        """Differential drive kinematic model."""
        assert robot_state.shape == (3, 1) and vel.shape == (2, 1)
        
        phi = robot_state[2, 0]
        v = vel[0, 0]
        w = vel[1, 0]
        
        ds = np.array([[v * cos(phi)], [v * sin(phi)], [w]])
        next_state = robot_state + ds * sample_time
        
        return next_state
    
    def default_turn_radius(self):
        """Calculate default turning radius from robot parameters."""
        if hasattr(self.robot, 'max_speed') and len(self.robot.max_speed) > 1:
            max_psi = float(self.robot.max_speed[1])
            wheelbase = float(self.robot.wheelbase)
            if abs(tan(max_psi)) > 1e-6:
                return wheelbase / tan(max_psi)
        return 0.0
    
    def _ensure_consistent_angles(self):
        """
        Ensure that all points in the initial path have consistent angles.
        For line curves, angles should represent the direction of travel.
        (From neupan InitialPath)
        """
        if self.initial_path is None or len(self.initial_path) < 2:
            return
        
        for i in range(len(self.initial_path) - 1):
            current_point = self.initial_path[i]
            next_point = self.initial_path[i + 1]
            
            dx = next_point[0, 0] - current_point[0, 0]
            dy = next_point[1, 0] - current_point[1, 0]
            
            theta = math.atan2(dy, dx)
            
            current_point[2, 0] = theta
        
        if len(self.initial_path) >= 2:
            self.initial_path[-1][2, 0] = self.initial_path[-2][2, 0]
    
    @property
    def cur_curve(self):
        """Get current curve segment."""
        if self.curve_list is None or self.curve_index >= len(self.curve_list):
            return self.initial_path if self.initial_path else []
        return self.curve_list[self.curve_index]
    
    @property
    def cur_point(self):
        """Get current point on path."""
        if len(self.cur_curve) == 0:
            return np.zeros((4, 1))
        if self.point_index >= len(self.cur_curve):
            return self.cur_curve[-1]
        return self.cur_curve[self.point_index]
    
    @property
    def path(self) -> Optional[List[np.ndarray]]:
        """Get initial path as list of points."""
        return self.initial_path
