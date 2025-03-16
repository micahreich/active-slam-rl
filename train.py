import argparse
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# ==============================================================
# Environment2D
# ==============================================================
class Environment2D:
    """
    A 50x50 grid environment with random rectangular obstacles.
    0 = free, 1 = obstacle. We do a 360° laser scan for a cost map.
    """
    def __init__(self, width=50, height=50, num_obstacles=5, obstacle_size_range=(5, 10), seed=None):
        self.width = width
        self.height = height
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Initialize grid: 0 = free, 1 = obstacle
        self.grid = np.zeros((height, width), dtype=np.int32)
        
        # Place obstacles as random rectangles
        for _ in range(num_obstacles):
            obs_width = random.randint(obstacle_size_range[0], obstacle_size_range[1])
            obs_height = random.randint(obstacle_size_range[0], obstacle_size_range[1])
            top_left_x = random.randint(1, width - obs_width - 1)
            top_left_y = random.randint(1, height - obs_height - 1)
            self.grid[top_left_y:top_left_y+obs_height, top_left_x:top_left_x+obs_width] = 1
        
        # Set the boundaries as obstacles
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def place_robot(self):
        """
        Randomly places the robot in a free cell and returns (x, y) plus orientation in [0, 2π].
        """
        free_cells = np.argwhere(self.grid == 0)
        if len(free_cells) == 0:
            raise ValueError("No free cells in the environment.")
        idx = np.random.choice(len(free_cells))
        cell = free_cells[idx]
        robot_position = (cell[1] + 0.5, cell[0] + 0.5)
        robot_orientation = np.random.uniform(0, 2 * np.pi)
        return robot_position, robot_orientation

    def raycast(self, start, angle, max_range):
        """
        A grid-based raycast (DDA). 
        Returns:
         distance: float,
         beam_points: list of (x, y) along the ray,
         hit: bool (True if obstacle was hit).
        """
        x, y = start
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        grid_x = int(x)
        grid_y = int(y)
        
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
        
        while distance < max_range:
            if tMaxX < tMaxY:
                distance = tMaxX
                grid_x += step_x
                tMaxX += tDeltaX
            else:
                distance = tMaxY
                grid_y += step_y
                tMaxY += tDeltaY
            
            px = x + dx * distance
            py = y + dy * distance
            beam_points.append((px, py))
            
            if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
                break
            
            if self.grid[grid_y, grid_x] == 1:
                hit = True
                break
        
        return min(distance, max_range), beam_points, hit

    def laser_scan(self, robot_position, robot_orientation, num_beams=360, max_range=10.0):
        """
        Cast laser rays in 360°, produce distance readings plus a cost map.
        cost_map is updated with 1.0 if obstacle was hit, or (distance/max_range)^2 if not.
        """
        angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
        measurements = []
        cost_map = np.zeros_like(self.grid, dtype=np.float32)
        
        for angle in angles:
            distance, beam_points, hit = self.raycast(robot_position, angle, max_range)
            measurements.append(distance)
            if beam_points:
                px, py = beam_points[-1]
                ix = int(px)
                iy = int(py)
                if 0 <= iy < self.height and 0 <= ix < self.width:
                    cost = 1.0 if hit else (distance / max_range) ** 2
                    cost_map[iy, ix] = max(cost_map[iy, ix], cost)
        
        return measurements, cost_map


def compute_obstacle_polygons(env):
    """
    Convert each obstacle cell into a 1x1 square shapely Polygon.
    """
    polygons = []
    height, width = env.grid.shape
    for y in range(height):
        for x in range(width):
            if env.grid[y, x] == 1:
                corners = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
                cell_poly = Polygon(corners)
                polygons.append(cell_poly)
    return polygons

def compute_visible_region(env, robot_pos, max_range=10.0, num_beams=360):
    """
    1) Gather laser endpoints -> hull
    2) Subtract all obstacle polygons from that hull => "safe region".
    """
    from shapely.ops import unary_union
    
    angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
    endpoints = []
    for angle in angles:
        dist, beam_points, hit = env.raycast(robot_pos, angle, max_range)
        if beam_points:
            endpoints.append(beam_points[-1])
    if len(endpoints) == 0:
        return None
    
    hull = Polygon(endpoints).convex_hull
    # Subtract obstacles
    obs_polys = compute_obstacle_polygons(env)
    all_obstacles = unary_union(obs_polys)
    
    safe_region = hull.difference(all_obstacles)
    if safe_region.is_empty:
        return None
    return safe_region


# ==============================================================
# Flow-based policy for (v,omega)
# ==============================================================
class RealNVPCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(RealNVPCoupling, self).__init__()
        self.dim = dim
        self.split = dim // 2
        self.scale_net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split),
            nn.Tanh()  # bounded output
        )
        self.translate_net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split)
        )
    
    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split], x[:, self.split:]
        if not reverse:
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
            y = torch.cat([y1, y2], dim=1)
            return y, log_det
        else:
            s = self.scale_net(x[:, :self.split])
            t = self.translate_net(x[:, :self.split])
            y1 = x[:, :self.split]
            y2 = (x[:, self.split:] - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
            y = torch.cat([y1, y2], dim=1)
            return y, log_det

class LambdaLayer(nn.Module):
    """A simple permutation layer: reverses the order of features."""
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class FlowPolicy(nn.Module):
    def __init__(self, latent_dim, num_layers=2):
        super(FlowPolicy, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RealNVPCoupling(latent_dim))
            self.layers.append(LambdaLayer(lambda x: x.flip(dims=[1])))
    
    def forward(self, z):
        log_det_total = 0
        out = z
        for layer in self.layers:
            if isinstance(layer, RealNVPCoupling):
                out, log_det = layer(out, reverse=False)
                log_det_total += log_det
            else:
                out = layer(out)
        return out, log_det_total

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim)
        a, log_det = self.forward(z)
        return a, log_det


# ==============================================================
# Differential Drive & Reward
# ==============================================================
def simulate_differential_drive(controls, start_pos, start_theta):
    """
    controls: (num_steps, 2) => [v, omega].
    Returns (waypoints, orientations).
    """
    num_steps = controls.shape[0]
    waypoints = [torch.tensor(start_pos, dtype=torch.float32, device=controls.device)]
    orientations = [torch.tensor(start_theta, dtype=torch.float32, device=controls.device)]
    
    for t in range(num_steps):
        v, omega = controls[t]
        theta = orientations[-1]
        new_x = waypoints[-1][0] + v * torch.cos(theta)
        new_y = waypoints[-1][1] + v * torch.sin(theta)
        new_theta = theta + omega
        waypoints.append(torch.stack([new_x, new_y]))
        orientations.append(new_theta)
    
    waypoints = torch.stack(waypoints)
    orientations = torch.stack(orientations)
    return waypoints, orientations

def compute_reward(controls, start_pos, start_theta,
                   cost_map_tensor, env, safe_region,
                   penalty_factor=100.0, smoothness_weight=1.0):
    """
    - Simulate differential drive
    - Reward = displacement - cost_penalty - hull_penalty - smooth_penalty
    - hull_penalty: if waypoint is outside safe_region => add penalty
    """
    from shapely.geometry import Point
    
    waypoints, orientations = simulate_differential_drive(controls, start_pos, start_theta)
    start_tensor = torch.tensor(start_pos, dtype=torch.float32, device=controls.device)
    displacement = torch.norm(waypoints[-1] - start_tensor)
    
    # Cost penalty from cost_map
    norm_waypoints = waypoints.clone()
    norm_waypoints[:, 0] = (norm_waypoints[:, 0] / (env.width - 1)) * 2 - 1
    norm_waypoints[:, 1] = (norm_waypoints[:, 1] / (env.height - 1)) * 2 - 1
    grid = norm_waypoints.view(1, -1, 1, 2)
    sampled_cost = F.grid_sample(cost_map_tensor, grid, align_corners=True)
    cost_penalty = sampled_cost.view(-1).sum()
    
    # Hull penalty: if a waypoint is outside safe_region => add penalty
    hull_penalty = 0.0
    for wp in waypoints:
        px, py = wp[0].item(), wp[1].item()
        if not safe_region.contains(Point(px, py)):
            hull_penalty += penalty_factor
    
    # Smoothness penalty
    smooth_penalty = 0.0
    if controls.shape[0] > 1:
        diffs = controls[1:] - controls[:-1]
        smooth_penalty = smoothness_weight * (diffs ** 2).sum()
    
    reward = displacement - cost_penalty - hull_penalty - smooth_penalty
    return reward, waypoints, orientations


# ==============================================================
# Pool Sampling Approach
# ==============================================================
def create_env_and_data():
    """
    Creates a single environment, places the robot, does laser scan,
    builds cost map, computes safe region.
    Returns:
     env, robot_pos, robot_orientation, cost_map_tensor, safe_region
    or None if invalid.
    """
    env = Environment2D(width=50, height=50, num_obstacles=5, obstacle_size_range=(5,10), seed=None)
    robot_pos, robot_orientation = env.place_robot()
    _, cost_map_np = env.laser_scan(robot_pos, robot_orientation, num_beams=360, max_range=10.0)
    cost_map_tensor = torch.tensor(cost_map_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    safe_region = compute_visible_region(env, robot_pos, max_range=10.0, num_beams=360)
    if safe_region is None or safe_region.is_empty:
        return None
    return (env, robot_pos, robot_orientation, cost_map_tensor, safe_region)

def sample_controls(policy, batch_size=1, num_steps=10):
    """
    Sample from the policy => shape (num_steps, 2).
    Clip v in [0,1], omega in [-1,1].
    """
    control_dim = 2
    a, _ = policy.sample(batch_size)
    a = a.view(batch_size, num_steps, control_dim)
    v = torch.clamp(a[:, :, 0], min=0.0, max=1.0)
    omega = torch.clamp(a[:, :, 1], min=-1.0, max=1.0)
    controls = torch.stack([v, omega], dim=2)
    return controls[0]

def main(args):
    num_steps = 10
    control_dim = 2
    trajectory_dim = num_steps * control_dim
    
    # Create policy
    policy = FlowPolicy(latent_dim=trajectory_dim, num_layers=2)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Build initial environment pool
    pool = []
    for _ in range(args.pool_size):
        data = create_env_and_data()
        while data is None:
            data = create_env_and_data()
        pool.append(data)  # each item is (env, robot_pos, robot_orientation, cost_map_tensor, safe_region)
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # If we are refreshing occasionally
        if args.refresh_interval > 0 and epoch % args.refresh_interval == 0 and epoch > 0:
            # Refresh a fraction of the pool (or all).
            # Example: refresh half of them
            half_to_refresh = max(1, args.pool_size // 2)
            for i in range(half_to_refresh):
                idx_to_refresh = random.randrange(args.pool_size)
                data = create_env_and_data()
                while data is None:
                    data = create_env_and_data()
                pool[idx_to_refresh] = data
        
        # Pick a random environment from the pool
        env_data = random.choice(pool)
        env, robot_pos, robot_orientation, cost_map_tensor, safe_region = env_data
        
        # Sample controls from policy
        controls = sample_controls(policy, batch_size=1, num_steps=num_steps)
        
        # Compute reward
        reward, _, _ = compute_reward(controls,
                                      robot_pos,
                                      robot_orientation,
                                      cost_map_tensor,
                                      env,
                                      safe_region,
                                      penalty_factor=args.penalty_factor,
                                      smoothness_weight=args.smoothness_weight)
        loss = -reward
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.3f}, Reward: {reward.item():.3f}")
    
    # Save model
    torch.save(policy.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")
    
    # Visualization with the last environment in the pool
    # (or we could pick a random one).
    last_env_data = pool[-1]
    env, robot_pos, robot_orientation, cost_map_tensor, safe_region = last_env_data
    with torch.no_grad():
        controls = sample_controls(policy, batch_size=1, num_steps=num_steps)
        final_reward, waypoints, orientations = compute_reward(
            controls, robot_pos, robot_orientation, cost_map_tensor, env, safe_region,
            penalty_factor=args.penalty_factor, smoothness_weight=args.smoothness_weight
        )
    waypoints = waypoints.detach().cpu().numpy()
    orientations = orientations.detach().cpu().numpy()
    
    plt.figure(figsize=(6,6))
    plt.imshow(env.grid, cmap='gray_r', origin='lower')
    plt.plot(waypoints[:, 0], waypoints[:, 1], '-o', color='green', label='Trajectory')
    plt.scatter([robot_pos[0]], [robot_pos[1]], color='blue', label='Robot Start')
    # Plot orientation arrows
    for i, (wp, theta) in enumerate(zip(waypoints, orientations)):
        dx = 1.0 * math.cos(theta)
        dy = 1.0 * math.sin(theta)
        plt.arrow(wp[0], wp[1], dx, dy, head_width=0.3, head_length=0.3, fc='cyan', ec='cyan')
    
    # Plot the safe region
    if not safe_region.is_empty:
        if safe_region.geom_type == 'Polygon':
            x_hull, y_hull = safe_region.exterior.xy
            plt.plot(x_hull, y_hull, color='red', linestyle='--', label='Safe Region')
        else:
            # Possibly MultiPolygon
            for geom in safe_region.geoms:
                x_hull, y_hull = geom.exterior.xy
                plt.plot(x_hull, y_hull, color='red', linestyle='--', label='Safe Sub-Region')
    
    plt.title("Differential Drive Policy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDrive + Flow-based policy with environment pool sampling.")
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs.')
    parser.add_argument('--pool_size', type=int, default=5, help='Number of environments in the pool.')
    parser.add_argument('--refresh_interval', type=int, default=500, help='Refresh a fraction of the pool every X epochs (0 = never).')
    parser.add_argument('--penalty_factor', type=float, default=100.0, help='Penalty factor if waypoint is outside the safe region.')
    parser.add_argument('--smoothness_weight', type=float, default=1.0, help='Weight for control smoothness penalty.')
    parser.add_argument('--save_model', type=str, default='dp.pth', help='Path to save the model.')
    args = parser.parse_args()
    
    main(args)
