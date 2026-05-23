import numpy as np
import heapq
import math
import cv2
import time
from typing import List, Union, Tuple, Optional, Dict

# Mock classes for standalone testing
try:
    from pathfinding.core.grid import Grid
    from pathfinding.core.node import GridNode
    from pathfinding.finder.a_star import AStarFinder
    from pathfinding.core.diagonal_movement import DiagonalMovement
except ImportError:
    class GridNode:
        def __init__(self, x, y, walkable=True):
            self.x, self.y, self.walkable = x, y, walkable
    class Grid:
        def __init__(self, width, height, matrix=None):
            self.width, self.height = width, height
            self.nodes = [[GridNode(x, y, matrix[y][x] > 0) for x in range(width)] for y in range(height)] if matrix else []
        def node(self, x, y): return self.nodes[y][x] if 0 <= x < self.width and 0 <= y < self.height else None
        def cleanup(self): pass
    class DiagonalMovement: only_when_no_obstacle = 1
    class AStarFinder:
        def __init__(self, diagonal_movement): pass
        def find_path(self, start, end, grid): return [], 0

class PathSearch:
    """
    Grid-based path search utility for SE(2) Asymmetric A*.
    Supports arbitrary vehicle shapes defined by polygons.
    """

    def __init__(self, grid_map: Grid, resolution: float = 0.5, 
                 origin: Optional[List[float]] = None, curve_style: str = "astar",
                 vehicle_polygons: Optional[List[List[List[float]]]] = None):
        
        if grid_map is None: raise ValueError("A valid Grid map must be provided.")
        
        self.grid_map = grid_map
        self.resolution = resolution
        self.origin = origin if origin is not None else [0.0, 0.0]
        self.curve_style = curve_style
        
        # 1. Convert Grid to NumPy array (0=Obstacle, 1=Walkable)
        self.np_grid = np.array([
            [1 if node.walkable else 0 for node in row] 
            for row in self.grid_map.nodes
        ], dtype=np.uint8)
        
        self.height, self.width = self.np_grid.shape
        self.angle_resolution = 10  # Degrees
        
        if vehicle_polygons is None:
            vehicle_polygons = [[[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]]
            
        self.vehicle_masks = self._precompute_vehicle_masks(vehicle_polygons)

    def _precompute_vehicle_masks(self, vehicle_polygons: List[List[List[float]]]) -> Dict[int, Tuple[np.ndarray, int, int]]:
        masks = {}
        max_dist = 0.0
        for poly in vehicle_polygons:
            for pt in poly:
                dist = np.hypot(pt[0], pt[1])
                if dist > max_dist: max_dist = dist
        
        canvas_dim = int(np.ceil(max_dist / self.resolution) * 2 + 8)
        center = canvas_dim // 2
        
        for angle_deg in range(0, 360, self.angle_resolution):
            mask = np.zeros((canvas_dim, canvas_dim), dtype=np.uint8)
            theta = np.radians(angle_deg)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            all_polys_grid = []
            for poly in vehicle_polygons:
                pts_world = np.array(poly)
                pts_rot = pts_world @ R.T 
                pts_grid = np.round(pts_rot / self.resolution).astype(np.int32)
                pts_grid[:, 0] += center
                pts_grid[:, 1] += center
                all_polys_grid.append(pts_grid)
            
            cv2.fillPoly(mask, all_polys_grid, 1)
            
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                masks[angle_deg] = (np.array([[1]], dtype=np.uint8), 0, 0)
                continue
                
            rmin, rmax = np.min(ys), np.max(ys)
            cmin, cmax = np.min(xs), np.max(xs)
            cropped = mask[rmin:rmax+1, cmin:cmax+1]
            masks[angle_deg] = (cropped, center - cmin, center - rmin) # mask, offset_x, offset_y
            
        return masks

    # --- Coordinate Conversions ---
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        gx = max(0, min(gx, self.width - 1))
        gy = max(0, min(gy, self.height - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = self.origin[0] + gx * self.resolution + self.resolution / 2
        wy = self.origin[1] + gy * self.resolution + self.resolution / 2
        return wx, wy

    # --- Collision Checks ---
    def _check_collision_fast(self, gx: int, gy: int, angle_deg: int) -> bool:
        """ Static state collision check """
        mask_tuple = self.vehicle_masks.get(angle_deg)
        if not mask_tuple: return True
        mask, ox, oy = mask_tuple
        h, w = mask.shape
        y1, x1 = gy - oy, gx - ox
        y2, x2 = y1 + h, x1 + w
        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height: return True
        return np.any((mask == 1) & (self.np_grid[y1:y2, x1:x2] == 0))

    def _check_path_collision(self, x1, y1, theta1_deg, x2, y2, theta2_deg) -> bool:
        """ Interpolated Swept Volume Check. """
        dist = math.hypot(x2 - x1, y2 - y1)
        angle_diff = (theta2_deg - theta1_deg + 180) % 360 - 180
        linear_steps = int(dist / 0.5) 
        angular_steps = int(abs(angle_diff) / 10.0) # slightly coarser for speed
        steps = max(linear_steps, angular_steps, 1)
        
        for i in range(1, steps + 1):
            t = i / steps
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            ith = (theta1_deg + t * angle_diff) % 360
            gx = int(round(ix))
            gy = int(round(iy))
            gth_idx = int(round(ith / self.angle_resolution)) * self.angle_resolution % 360
            if self._check_collision_fast(gx, gy, gth_idx):
                return True
        return False

    # --- Heuristic Function ---
    def _heuristic(self, x1, y1, angle1, x2, y2, angle2, use_angle_heuristic=False):
        """
        Estimate the cost from the current state to the goal.
        """
        dist_cost = math.hypot(x1 - x2, y1 - y2)
        
        angle_cost = 0.0
        if use_angle_heuristic:
            diff = abs(angle1 - angle2)
            diff = min(diff, 360 - diff)
            
            angle_cost = (diff / 180.0) * 5.0  # 180 deg is weighted as 5 m.
            
        return dist_cost + angle_cost

    # --- Main Interface ---
    def find_initial_path(self, start: Union[np.ndarray, List], goal: Union[np.ndarray, List]) -> Tuple[List[Tuple[int, int]], int]:
        self.last_start = start
        self.last_goal = goal
        t0 = time.time()
        if self.curve_style == "astar":
            path, runs = self._plan_with_astar(start, goal)
        elif self.curve_style == "asymmetric_astar":
            path, runs = self._plan_with_asymmetric_astar(start, goal)
        else:
            raise NotImplementedError(f"Style {self.curve_style} not supported.")

        # Normalize path format to (x, y, theta_idx) for downstream consumers
        path = self._attach_yaw_to_grid_path(path)

        elapsed_ms = (time.time() - t0) * 1000.0
        print(f"[path_search] completed style={self.curve_style} len={len(path)} runs={runs} time={elapsed_ms:.1f}ms")
        return path, runs

    def _plan_with_astar(self, start, goal):
        self.grid_map.cleanup()
        start_gx, start_gy = self.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.world_to_grid(goal[0], goal[1])
        start_node = self.grid_map.node(start_gx, start_gy)
        end_node = self.grid_map.node(goal_gx, goal_gy)
        if not start_node or not end_node or not start_node.walkable or not end_node.walkable: return [], 0
        path, runs = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle).find_path(start_node, end_node, self.grid_map)

        # Normalize path to list of (x, y)
        if path and not isinstance(path[0], (tuple, list)):
            try:
                path = [(n.x, n.y) for n in path]
            except Exception:
                path = []

        return path, runs

    def _attach_yaw_to_grid_path(self, grid_path: List[Tuple]) -> List[Tuple[int, int, int]]:
        """Ensure a grid path contains yaw indices; infer yaw from neighbors for plain A* paths."""
        if not grid_path:
            return []

        # Already contains yaw info
        first = grid_path[0]
        if hasattr(first, "x") and hasattr(first, "y"):
            # Convert any remaining GridNode-like items defensively
            grid_path = [(n.x, n.y) if hasattr(n, "x") else n for n in grid_path]

        if len(grid_path[0]) == 3:
            return grid_path

        enriched = []
        n = len(grid_path)
        for i, (gx, gy) in enumerate(grid_path):
            if i + 1 < n:
                nx, ny = grid_path[i + 1]
            elif i > 0:
                nx, ny = grid_path[i - 1]
            else:
                nx, ny = gx + 1, gy  # degenerate single-point path

            yaw_rad = math.atan2(ny - gy, nx - gx)
            yaw_deg = np.degrees(yaw_rad) % 360
            yaw_idx = int(round(yaw_deg / self.angle_resolution)) * self.angle_resolution % 360
            enriched.append((gx, gy, yaw_idx))

        return enriched

    def _plan_with_asymmetric_astar(self, start: Union[np.ndarray, List], goal: Union[np.ndarray, List]) -> Tuple[List[Tuple[int, int]], int]:
        """
        Run SE(2) A* with optional goal yaw constraints.
        """
        start_gx, start_gy = self.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.world_to_grid(goal[0], goal[1])
        
        start_theta = start[2] if len(start) > 2 else 0.0
        start_angle_idx = int(np.round(np.degrees(start_theta) / self.angle_resolution)) * self.angle_resolution % 360

        use_goal_yaw = (len(goal) > 2)
        goal_angle_idx = 0
        if use_goal_yaw:
            goal_theta = goal[2]
            goal_angle_idx = int(np.round(np.degrees(goal_theta) / self.angle_resolution)) * self.angle_resolution % 360
            print(f"[path_search] Goal Yaw constraint enabled: {np.degrees(goal_theta):.1f} deg -> Index {goal_angle_idx}")
        
        open_list = []
        heapq.heappush(open_list, (0, 0, start_gx, start_gy, start_angle_idx))
        
        came_from = {}
        cost_so_far = {}
        start_state = (start_gx, start_gy, start_angle_idx)
        came_from[start_state] = None
        cost_so_far[start_state] = 0
        
        runs = 0
        # 8-Connectivity (x, y, cost)
        motions = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
                   (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        # BFS Search Order for angles
        angle_search_order = [0]
        max_search_depth = 6 
        for i in range(1, max_search_depth + 1):
            angle_search_order.append(i * self.angle_resolution)
            angle_search_order.append(-i * self.angle_resolution)

        path_found = False
        final_state = None

        while open_list:
            runs += 1
            _, current_cost, cx, cy, c_angle = heapq.heappop(open_list)
            
            dist_to_goal = abs(cx - goal_gx) + abs(cy - goal_gy) # Manhattan check is faster
            
            if dist_to_goal == 0:
                if use_goal_yaw:
                    if c_angle == goal_angle_idx:
                        final_state = (cx, cy, c_angle)
                        path_found = True
                        break
                else:
                    final_state = (cx, cy, c_angle)
                    path_found = True
                    break
            
            for dx, dy, move_cost in motions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height): continue
                
                valid_angle = -1
                step_penalty = 0.0
                found_valid = False
                
                for da in angle_search_order:
                    test_angle = (c_angle + da) % 360
                    
                    if not self._check_path_collision(cx, cy, c_angle, nx, ny, test_angle):
                        valid_angle = test_angle
                        found_valid = True
                        if da != 0: step_penalty = abs(da) * 0.05 
                        break 
                
                if not found_valid: continue
                
                new_cost = current_cost + move_cost + step_penalty
                new_state = (nx, ny, valid_angle)
                
                if new_state not in cost_so_far or new_cost < cost_so_far[new_state]:
                    cost_so_far[new_state] = new_cost
                    
                    priority = new_cost + self._heuristic(
                        nx, ny, valid_angle, 
                        goal_gx, goal_gy, goal_angle_idx, 
                        use_angle_heuristic=use_goal_yaw
                    )
                    
                    heapq.heappush(open_list, (priority, new_cost, nx, ny, valid_angle))
                    came_from[new_state] = (cx, cy, c_angle)

        if not path_found: return [], runs
        
        path = []
        curr = final_state
        while curr is not None:
            path.append((curr[0], curr[1], curr[2])) 
            curr = came_from[curr]
        return path[::-1], runs

    def path_to_world_coords(self, grid_path: List[Tuple[int, int, int]], 
                             interval: float = 0.3) -> List[np.ndarray]:
        """
        Convert grid path (x, y, theta_idx) to world path (x, y, yaw).
        Includes interpolation for both position and yaw.
        """
        if not grid_path:
            return []
        
        keyframes = []
        for i, pt in enumerate(grid_path):
            gx, gy, gth_idx = pt
            
            # If it's the start or end point, use the exact world coordinates if available
            if i == 0 and hasattr(self, 'last_start') and self.last_start is not None:
                wx, wy = self.last_start[0], self.last_start[1]
                yaw_rad = self.last_start[2] if len(self.last_start) > 2 else np.radians(gth_idx)
            elif i == len(grid_path) - 1 and hasattr(self, 'last_goal') and self.last_goal is not None:
                wx, wy = self.last_goal[0], self.last_goal[1]
                yaw_rad = self.last_goal[2] if len(self.last_goal) > 2 else np.radians(gth_idx)
            else:
                wx, wy = self.grid_to_world(gx, gy)
                yaw_rad = np.radians(gth_idx)
            
            keyframes.append((wx, wy, yaw_rad))
            
        dense_path = self._densify_path_with_yaw(keyframes, interval)
        
        world_path = []
        for pt in dense_path:
            # pt is (x, y, yaw)
            # Format: [x, y, yaw, gear] (gear=1.0 default)
            point = np.array([[pt[0]], [pt[1]], [pt[2]], [1.0]])
            world_path.append(point)
            
        return world_path

    def _densify_path_with_yaw(self, keyframes: List[Tuple[float, float, float]], 
                               interval: float) -> List[Tuple[float, float, float]]:
        """
        Linearly interpolate position (x,y) and Slerp (linear angle) for yaw.
        """
        if len(keyframes) < 2:
            return keyframes
            
        dense_points = []
        
        for i in range(len(keyframes) - 1):
            x1, y1, th1 = keyframes[i]
            x2, y2, th2 = keyframes[i + 1]
            
            dense_points.append((x1, y1, th1))
            
            dist = math.hypot(x2 - x1, y2 - y1)
            
            # Wrap yaw to the shortest turn.
            yaw_diff = th2 - th1
            while yaw_diff > np.pi: yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi: yaw_diff += 2 * np.pi
            
            dist_steps = int(np.ceil(dist / interval))
            angle_steps = int(np.ceil(abs(yaw_diff) / 0.1))
            n_insert = max(dist_steps, angle_steps)
            
            if n_insert > 1:
                for j in range(1, n_insert):
                    t = j / n_insert
                    
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    
                    th = th1 + t * yaw_diff
                    
                    dense_points.append((x, y, th))
        
        dense_points.append(keyframes[-1])
        
        return dense_points

# Test Logic
if __name__ == "__main__":
    w, h = 50, 50
    matrix = np.ones((h, w), dtype=np.int8) 
    
    # Narrow corridor fixture.
    matrix[10:40, 20] = 0
    matrix[10:40, 24] = 0
    
    grid = Grid(width=w, height=h, matrix=matrix)
    
    # Long rectangular robot.
    robot_parts = [[[1.0, 0.25], [-1.0, 0.25], [-1.0, -0.25], [1.0, -0.25]]]
    
    searcher = PathSearch(grid, resolution=0.2, curve_style="asymmetric_astar", vehicle_polygons=robot_parts)
    
    # Case 1: Start Horizontal (0 deg), Goal Inside Corridor Horizontal (0 deg)
    start = [5.0, 5.0, 0.0]  
    goal = [4.4, 6.0, np.radians(90)]
    
    print(f"Planning with Goal Yaw Constraint...")
    path, runs = searcher.find_initial_path(start, goal)
    
    if len(path) > 0:
        print(f"Success! Final Node: {path[-1]}")
        final_yaw_idx = path[-1][2]
        print(f"Goal Yaw Index: {final_yaw_idx} (Should be close to 90deg index)")
    else:
        print("Failed to find path")
