import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Environment2D:
    def __init__(self, width, height, num_obstacles=5, obstacle_size_range=(5, 10), seed=None):
        """
        Create a 2D grid environment with sparse, larger obstacles.

        Args:
            width (int): number of cells in the x-direction.
            height (int): number of cells in the y-direction.
            num_obstacles (int): number of rectangular obstacles to place.
            obstacle_size_range (tuple): (min_size, max_size) for the obstacle dimensions.
            seed (int or None): seed for reproducibility. If None, randomness is not fixed.
        """
        self.width = width
        self.height = height
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # Initialize grid: 0 = free, 1 = obstacle.
        self.grid = np.zeros((height, width), dtype=np.int32)
        
        # Place obstacles as random rectangles.
        for _ in range(num_obstacles):
            obs_width = random.randint(obstacle_size_range[0], obstacle_size_range[1])
            obs_height = random.randint(obstacle_size_range[0], obstacle_size_range[1])
            top_left_x = random.randint(1, width - obs_width - 1)
            top_left_y = random.randint(1, height - obs_height - 1)
            self.grid[top_left_y:top_left_y+obs_height, top_left_x:top_left_x+obs_width] = 1

        # Set the boundaries as obstacles.
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def place_robot(self):
        """
        Randomly places the robot in a free (non-obstacle) cell and assigns a random orientation.
        
        Returns:
            robot_position (tuple): (x, y) continuous coordinates of the robot.
            robot_orientation (float): orientation in radians.
        """
        free_cells = np.argwhere(self.grid == 0)
        idx = np.random.choice(len(free_cells))
        cell = free_cells[idx]
        # Center of the cell in continuous coordinates.
        robot_position = (cell[1] + 0.5, cell[0] + 0.5)
        # Random orientation between 0 and 2*pi
        robot_orientation = np.random.uniform(0, 2 * np.pi)
        return robot_position, robot_orientation

    def raycast(self, start, angle, max_range):
        """
        Perform a grid-based raycast (using DDA) from the start position in the given angle.
        
        Args:
            start (tuple): (x, y) continuous starting coordinates.
            angle (float): angle (in radians) for the ray.
            max_range (float): maximum range of the sensor.
        
        Returns:
            distance (float): distance at which the ray hit an obstacle (or max_range if none).
            beam_points (list): list of continuous (x, y) points along the ray.
            hit (bool): True if an obstacle was hit.
        """
        x, y = start
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Current grid cell.
        grid_x = int(x)
        grid_y = int(y)
        
        # Pre-calculate step and initial tMax/tDelta for x.
        if dx == 0:
            tDeltaX = float('inf')
            tMaxX = float('inf')
            step_x = 0
        else:
            step_x = 1 if dx > 0 else -1
            tDeltaX = 1 / abs(dx)
            if dx > 0:
                tMaxX = (grid_x + 1 - x) / dx
            else:
                tMaxX = (x - grid_x) / abs(dx)
        
        # Pre-calculate step and initial tMax/tDelta for y.
        if dy == 0:
            tDeltaY = float('inf')
            tMaxY = float('inf')
            step_y = 0
        else:
            step_y = 1 if dy > 0 else -1
            tDeltaY = 1 / abs(dy)
            if dy > 0:
                tMaxY = (grid_y + 1 - y) / dy
            else:
                tMaxY = (y - grid_y) / abs(dy)
        
        distance = 0.0
        beam_points = []
        hit = False
        
        # Continue stepping until we exceed max_range.
        while distance < max_range:
            if tMaxX < tMaxY:
                distance = tMaxX
                grid_x += step_x
                tMaxX += tDeltaX
            else:
                distance = tMaxY
                grid_y += step_y
                tMaxY += tDeltaY

            # Compute the continuous intersection point.
            px = x + dx * distance
            py = y + dy * distance
            beam_points.append((px, py))
            
            # Check bounds.
            if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
                break

            # Check for obstacle hit.
            if self.grid[grid_y, grid_x] == 1:
                hit = True
                break

        return min(distance, max_range), beam_points, hit

    def laser_scan(self, robot_position, robot_orientation, num_beams=360, max_range=10.0):
        """
        Simulate a 2D laser scan from the robot's position using grid raycasting over a 360° field-of-view.
        
        For each beam, the ray is cast using DDA until it:
          - hits an obstacle,
          - goes out-of-bounds, or
          - reaches the maximum range.
          
        The endpoint of each beam is always marked in the cost map.
        If the beam terminated because an obstacle was hit, that cell is assigned a cost of 1.0.
        Otherwise, the cost is computed as the squared ratio of the beam length to max_range.
        
        Args:
            robot_position (tuple): (x, y) coordinates of the robot.
            robot_orientation (float): the robot's facing direction in radians.
            num_beams (int): number of rays to cast (covering 360 degrees).
            max_range (float): maximum sensor range (default 10 units).
        
        Returns:
            measurements (list): distance measured for each beam.
            cost_map (np.array): 2D array (same size as grid) with cost values.
        """
        angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
        measurements = []
        cost_map = np.zeros_like(self.grid, dtype=np.float32)
        
        for angle in angles:
            distance, beam_points, hit = self.raycast(robot_position, angle, max_range)
            measurements.append(distance)
            
            if beam_points:
                # Get the endpoint of the beam.
                px, py = beam_points[-1]
                ix = int(px)
                iy = int(py)
                # Compute cost: if hit, mark as 1.0; if not, use squared ratio.
                cost = 1.0 if hit else (distance / max_range) ** 2
                cost_map[iy, ix] = max(cost_map[iy, ix], cost)
        
        return measurements, cost_map

# Example usage:
if __name__ == "__main__":
    # Create a sparse environment with larger obstacles; seed is None for random behavior.
    env = Environment2D(width=50, height=50, num_obstacles=5, obstacle_size_range=(5, 10), seed=None)
    
    # Place the robot and assign a random orientation.
    robot_pos, robot_orientation = env.place_robot()
    measurements, cost_map = env.laser_scan(robot_pos, robot_orientation, num_beams=360, max_range=10.0)
    
    # Plot occupancy grid and cost map side by side.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Occupancy Grid (Environment Map)
    axs[0].imshow(env.grid, cmap='gray_r', origin='lower')
    axs[0].scatter([robot_pos[0]], [robot_pos[1]], c='blue', label='Robot')
    # Draw an arrow for the robot's orientation.
    arrow_length = 3.0
    arrow_dx = arrow_length * math.cos(robot_orientation)
    arrow_dy = arrow_length * math.sin(robot_orientation)
    axs[0].arrow(robot_pos[0], robot_pos[1], arrow_dx, arrow_dy, head_width=0.5, head_length=0.7, fc='blue', ec='blue')
    axs[0].set_title("Occupancy Grid")
    axs[0].legend()
    
    # Cost Map from Laser Scan
    im = axs[1].imshow(cost_map, cmap='hot', origin='lower')
    axs[1].set_title("Cost Map from Laser Scan")
    fig.colorbar(im, ax=axs[1], label='Cost')
    
    plt.tight_layout()
    plt.show()
