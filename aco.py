import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib.animation import FuncAnimation
import rasterio
from rasterio.transform import from_origin

# Parameters
NUM_ANTS = 75
MAX_ITERATIONS = 250
ALPHA = 1.0  # Pheromone importance
BETA = 2.0   # Heuristic importance
RHO = 0.1    # Pheromone evaporation rate
Q = 10       # Pheromone deposit amount when ants reach the goal
GRID_SIZE = 50  # Size of the grid cells
PHEROMONE_INITIAL = 1.0  # Initial pheromone level
RESET_INTERVAL = 30  # Number of iterations after which ants reset

# Load elevation map
with rasterio.open('images/processed_imgs/49 (1).jpg') as src:
    elevation_map = src.read(1)  # Read the first band
    transform = src.transform
    crs = src.crs

height, width = elevation_map.shape
res_x, res_y = transform[0], -transform[4]

x_coords = np.arange(transform[2], transform[2] + width * res_x, res_x)
y_coords = np.arange(transform[5], transform[5] - height * res_y, -res_y)

# Compute slope map
def compute_slope(elevation_map):
    gradient_x, gradient_y = np.gradient(elevation_map)
    slope_map = np.sqrt(gradient_x**2 + gradient_y**2)
    slope_map = np.clip(slope_map, 0, 1)
    return slope_map

slope_map = compute_slope(elevation_map)

# Initialize pheromone grid
pheromone_grid = np.ones_like(elevation_map) * PHEROMONE_INITIAL

# Define start and goal positions
goal_position = np.array([random.choice(x_coords), random.choice(y_coords)])
start_position = np.array([random.choice(x_coords), random.choice(y_coords)])

# Utility functions
def coord_to_index(x, y):
    col = int((x - transform[2]) / res_x)
    row = int((transform[5] - y) / res_y)
    return row, col

def index_to_coord(row, col):
    x = transform[2] + col * res_x
    y = transform[5] - row * res_y
    return x, y

def get_pheromone(x, y):
    row, col = coord_to_index(x, y)
    if 0 <= row < height and 0 <= col < width:
        return pheromone_grid[row, col]
    return 0

def update_pheromone(x, y, value):
    row, col = coord_to_index(x, y)
    if 0 <= row < height and 0 <= col < width:
        pheromone_grid[row, col] += value

def evaporate_pheromones():
    global pheromone_grid
    pheromone_grid *= (1 - RHO)

def get_elevation(x, y):
    row, col = coord_to_index(x, y)
    if 0 <= row < height and 0 <= col < width:
        return elevation_map[row, col]
    return 0

def get_slope(x, y):
    row, col = coord_to_index(x, y)
    if 0 <= row < height and 0 <= col < width:
        return slope_map[row, col]
    return 0

# Ant class
class Ant:
    def __init__(self, start, goal, follow_pheromone=False):
        self.position = start.copy()
        self.path = [start.copy()]
        self.visited = set()
        self.reached_goal = False
        self.has_complete_path = False
        self.complete_path = None
        self.goal = goal
        self.follow_pheromone = follow_pheromone

    def move(self):
        if self.reached_goal:
            return

        possible_moves = self.get_possible_moves()
        if not possible_moves:
            return

        next_move = self.select_next_move(possible_moves)
        self.position = next_move
        self.path.append(next_move.copy())
        self.visited.add(tuple(next_move))

        if np.linalg.norm(self.position - self.goal) < GRID_SIZE:
            for point in self.path:
                update_pheromone(point[0], point[1], Q)
            self.reached_goal = True
            self.complete_path = self.path.copy()

    def get_possible_moves(self):
        possible_moves = []
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x = self.position[0] + GRID_SIZE * math.cos(rad)
            y = self.position[1] + GRID_SIZE * math.sin(rad)
            if self.is_valid_move(x, y) and (x, y) not in self.visited:
                possible_moves.append(np.array([x, y]))
        return possible_moves

    def is_valid_move(self, x, y):
        row, col = coord_to_index(x, y)
        return 0 <= row < height and 0 <= col < width

    def select_next_move(self, possible_moves):
        if self.follow_pheromone:
            probabilities = []
            total_prob = 0.0
            for move in possible_moves:
                pheromone = get_pheromone(move[0], move[1])
                distance = np.linalg.norm(move - self.goal)
                elevation = get_elevation(move[0], move[1])
                slope = get_slope(move[0], move[1])

                speed_factor = 1.0 / (1.0 + elevation) if elevation > 0 else 1.0 + abs(elevation)
                adjusted_distance = distance / (1 + slope)
                heuristic = 1.0 / adjusted_distance if adjusted_distance > 0 else 1.0
                probability = (pheromone**ALPHA) * (heuristic**BETA) * speed_factor
                probabilities.append(probability)
                total_prob += probability

            if total_prob == 0:
                return random.choice(possible_moves)

            rand_value = random.uniform(0, total_prob)
            cumulative_prob = 0.0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if cumulative_prob >= rand_value:
                    return possible_moves[i]

            return random.choice(possible_moves)
        else:
            return random.choice(possible_moves)

# Initialize ants
def initialize_ants(follow_pheromone=False):
    return [Ant(start_position, goal_position, follow_pheromone=follow_pheromone) for _ in range(NUM_ANTS // 2)], \
           [Ant(goal_position, start_position, follow_pheromone=follow_pheromone) for _ in range(NUM_ANTS // 2)]

forward_ants, backward_ants = initialize_ants()

# Set up the plot
fig, ax1 = plt.subplots(figsize=(9, 6))

# Subplot: Optimal Path
ax1.set_xlim(x_coords.min(), x_coords.max())
ax1.set_ylim(y_coords.min(), y_coords.max())
ax1.set_title("Optimal Path")
goal_dot, = ax1.plot(*goal_position, 'ro', label="Goal Position")
start_dot, = ax1.plot(*start_position, 'go', label="Start Position")
optimal_path_line, = ax1.plot([], [], 'b-', linewidth=2, label="Optimal Path")

elevation_contour = ax1.contour(x_coords, y_coords, elevation_map, cmap='coolwarm', alpha=0.4)
slope_contour = ax1.contour(x_coords, y_coords, slope_map, cmap='Greens', alpha=0.4)

ax1.legend()

# Optimization variables
best_path = None
best_path_length = float('inf')
iteration_count = 0

# Update best path logic
def update_best_path(forward_ant, backward_ant):
    global best_path, best_path_length
    meeting_point = None
    forward_index = -1
    backward_index = -1

    for i, f_pos in enumerate(forward_ant.path):
        for j, b_pos in enumerate(backward_ant.path):
            if np.linalg.norm(f_pos - b_pos) < GRID_SIZE:
                meeting_point = f_pos
                forward_index = i
                backward_index = j
                break
        if meeting_point is not None:
            break

    if meeting_point is not None:
        forward_path = forward_ant.path[:forward_index + 1]
        backward_path = backward_ant.path[:backward_index][::-1]
        complete_path = forward_path + backward_path

        path_length = np.sum(np.linalg.norm(np.diff(complete_path, axis=0), axis=1))
        if path_length < best_path_length:
            best_path_length = path_length
            best_path = np.array(complete_path)
            optimal_path_line.set_data(best_path[:, 0], best_path[:, 1])
            return True
    return False

# Reset ants
def reset_ants():
    global forward_ants, backward_ants
    forward_ants, backward_ants = initialize_ants(follow_pheromone=True)

# Animation update function
def update(frame):
    global best_path, best_path_length, iteration_count
    path_found = False

    for ant in forward_ants + backward_ants:
        ant.move()

    for f_ant in forward_ants:
        for b_ant in backward_ants:
            if update_best_path(f_ant, b_ant):
                path_found = True

    iteration_count += 1
    if iteration_count >= RESET_INTERVAL:
        iteration_count = 0
        reset_ants()

    evaporate_pheromones()

    if iteration_count == 0 or path_found:
        print(f"Frame {frame}: Best path length: {best_path_length:.2f}")

    return [optimal_path_line]

# Run animation
ani = FuncAnimation(fig, update, interval=0.001, frames=MAX_ITERATIONS, blit=True, repeat=False)
plt.show()

# //change reduction in remote sensing smart city application (satellite image dataset, case studyc)
