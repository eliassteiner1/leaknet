"""
Synthetic scene generation.

A "scene" is a 2D site in a [-1, 1] box with a bunch of possible sources, of which a few are actually leaking, a handful
of sensors, and a wind time series. For every timestep each sensor gets a concentration value from a (handmade!) plume
formula. Collections of scenes are written to one hdf5 file, one group per scene.
"""

import random
import time
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


### plume --------------------------------------------------------------------------------------------------------------
def atmospheric_dispersion(x, y, Q, u):
    """
    Main atmospheric dispersion model. Takes in some coordinates, a leak strength Q and a wind speed u. Then spits out
    the concentration C @ (x, y). Note: this is meant to be in the wind coordinate frame where x is downwind and y is
    crosswind. Transform points accordingly.

    This is not a physical gaussian plume model, just a formula that was tuned by eye until the plumes looked plausible:
    a round blob around the leak in calm conditions that turns into a 1/x downwind plume with increasing wind.

    Args:
        x: x coordinate (downwind)
        y: y coordinate (crosswind)
        Q: leak strength
        u: wind speed
    """

    def w(u):
        # weighting function for high wind influence
        result = np.where(u >= 0, (20 * u * np.exp(-3 * u)) / (2.15 + u), 0)
        return result

    def lo(x, u, Q):
        # x-axis distribution in low wind conditions
        result = np.exp((-(x**2)) / (10 ** (-5) * (0.5 + u) ** (-14))) * (Q / (u + 0.1))
        return result

    def hi(x, u, Q):
        # x-axis distribution in high wind conditions
        result = np.where(x >= 0, (1 / (6 * x + 1)) * (Q / (u + 0.1)), lo(x, u, Q))
        return result

    def C_x(x, u, Q):
        # x-axis weighted combination of high and low wind conditions
        result = lo(x, u, Q) + w(u) * hi(x, u, Q)
        return result

    def sigma(x, u):
        # balance function for transitioning from upwind (x<0) to downwind (x>0) + at small u's basically only the
        # upwind condition (gaussian round plume) should be used and at high u's only the downwind 1/x plume.
        result = np.where(
            x > 0, np.where(u > 0, 1 / (1 + np.exp(7 * (7 * np.abs(x) ** 0.1 + 10 * np.abs(u) ** 0.8 - 9))), 1), 1
        )
        return result

    def cross(x, y):
        # y-axis crosswind function (gaussian bell that gets progressively wider and lower)
        a = 0.02
        result = np.where(x > 0, np.exp(-(y**2) / (a * x**2)), 0)
        return result

    def C(x, y, u, Q):
        # total combination. Returns the full concentration function
        result = np.where(
            x <= 0,
            np.sqrt(C_x(x, u, Q) * lo(y, u, Q)),
            np.sqrt(C_x(x, u, Q) * (sigma(x, u) * lo(y, u, Q) + cross(x, y))),
        )
        return result

    # the unused np.where branches divide by zero / overflow here and there, the selected values are fine
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return C(x, y, u, Q)


def coordinate_transform_scalar(x, y, px, py, wind_direc):
    """
    Takes in some global coordinates and transforms them into the local wind frame with wind direction and leak
    location marking the new 0, 0.

    Args:
        x, y: global point
        px, py: leak position (in global)
        wind_direc: wind direction angle (in global rad, from x+ = 0)
    """

    # vector from leak to x, y
    vec = np.array([x - px, y - py])
    # wind normal (!) vector
    wind = np.array([np.cos(wind_direc), np.sin(wind_direc)])
    # downwind x is just projection of vec onto wind direction
    x_local = np.dot(vec, wind)
    # crosswind y is just the other remaining perpendicular part
    y_local = np.linalg.norm(vec - x_local * wind)

    return x_local, y_local


def coordinate_transform_array(X, Y, px, py, wind_direc):
    """
    Does the same thing as coordinate_transform_scalar but for whole meshgridded arrays of points.
    """

    # elements of path vector from leak to every global point
    X = X - px
    Y = Y - py
    # stack both elements to form all the vectors @ every point in the input grid
    XY = np.stack([X, Y])
    # also create the wind vector at every location (constant)
    WIND = np.zeros(XY.shape)
    WIND[0, :, :] = np.cos(wind_direc)
    WIND[1, :, :] = np.sin(wind_direc)
    # x_local is projection of path vector onto wind vector = np.dot(vec, wind). (taking manual dot product here)
    X_local = XY[0, :, :] * WIND[0, :, :] + XY[1, :, :] * WIND[1, :, :]
    # y_local is the remaining perpendicular part = np.linalg.norm(vec - x_local*wind)
    Y_local = np.linalg.norm(XY - X_local * WIND, axis=0)

    return X_local, Y_local


### site layout --------------------------------------------------------------------------------------------------------
def my_normalize(coords, padding_factor):
    """
    Custom function to scale-shift normalize a bunch of coordinates to be within a box of xy [-1, 1]. Padding factor
    makes the bounding box of the points smaller than [-1, 1] (if padding_factor > 1)
    """

    x_max = np.max(coords[:, 0])
    x_min = np.min(coords[:, 0])
    y_max = np.max(coords[:, 1])
    y_min = np.min(coords[:, 1])
    x_delta = x_max - x_min
    y_delta = y_max - y_min
    x_mean = (x_max + x_min) / 2
    y_mean = (y_max + y_min) / 2

    coords_normalized = np.zeros(coords.shape)
    coords_normalized[:, 0] = (coords[:, 0] - x_mean) / (0.5 * x_delta * padding_factor)
    coords_normalized[:, 1] = (coords[:, 1] - y_mean) / (0.5 * y_delta * padding_factor)

    return coords_normalized


def optimized_poisson_disk_sampling(min_points, max_points, k=60, radius_multiplier=1.2, padding=0.05):
    """
    A function to generate more nicely distributed random sensor patterns using Poisson disk sampling. Returns one
    collection of n sensors (where each n_sensors is equally likely but random)

    Args:
        min_points: minimal amount of sensors
        max_points: maximal amount of sensors
        k: number of retries when finding next point (in poisson disk sampling)
        radius_multiplier: radius modifier. larger (>1) results in better spread but needs more retries
        padding: border from the edge of [-1, 1] box of sensor bounding box
    """

    # Set the width and height for the square from -1 to 1
    width, height = 2, 2

    # Randomly choose the number of points within the specified range
    num_points = np.random.randint(min_points, max_points + 1)

    # Calculate approximate radius based on desired number of points and apply multiplier
    base_radius = np.sqrt((width * height) / (num_points * np.pi))
    radius = base_radius * radius_multiplier

    points = []

    # Grid size based on calculated radius
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    # Create grid with empty cells
    grid = -np.ones((grid_width, grid_height), dtype=int)

    # Generate the first point
    initial_point = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
    points.append(initial_point)
    grid[int((initial_point[0] + 1) / cell_size), int((initial_point[1] + 1) / cell_size)] = 0

    # List of active points
    active_points = [0]

    while active_points and len(points) < num_points:
        # Choose a random index from active points
        idx = np.random.choice(active_points)
        base_point = points[idx]

        found_new_point = False
        for _ in range(k):
            # Generate a new point around the base point within [radius, 2*radius]
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(radius, 2 * radius)
            new_point = base_point + distance * np.array([np.cos(angle), np.sin(angle)])

            # Check if the new point is within bounds
            if not (-1 <= new_point[0] <= 1 and -1 <= new_point[1] <= 1):
                continue

            # Calculate the grid cell of the new point
            grid_x, grid_y = int((new_point[0] + 1) / cell_size), int((new_point[1] + 1) / cell_size)

            # Check neighboring cells for minimum distance constraint
            nearby_points = []
            for i in range(max(grid_x - 2, 0), min(grid_x + 3, grid_width)):
                for j in range(max(grid_y - 2, 0), min(grid_y + 3, grid_height)):
                    if grid[i, j] != -1:
                        nearby_points.append(points[grid[i, j]])

            if nearby_points:
                # Use a KDTree for efficient distance checking
                tree = cKDTree(nearby_points)
                if np.min(tree.query([new_point], k=1)[0]) < radius:
                    continue

            # Add new point to list and grid
            points.append(new_point)
            active_points.append(len(points) - 1)
            grid[grid_x, grid_y] = len(points) - 1
            found_new_point = True
            break

        if not found_new_point:
            # No valid points found; deactivate the point
            active_points.remove(idx)

    points = np.array(points[:num_points])

    # Stretch and translate points to fit within the padded area of [-1, 1]
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()

    effective_width = width - 2 * padding
    effective_height = height - 2 * padding

    points[:, 0] = (points[:, 0] - min_x) / (max_x - min_x) * effective_width + (-1 + padding)
    points[:, 1] = (points[:, 1] - min_y) / (max_y - min_y) * effective_height + (-1 + padding)

    return points


def generate_sensor_patterns(n_sensor_patterns, n_sensor_min, n_sensor_max):
    """
    This function generates a certain amount of fixed random sensor patterns. Returns a list of arrays (coords).
    Because choosing n_sensors truly randomly for only very few patterns (<20) can result in pretty skewed
    distributions, n_sensors is just cycled through to make it pretty uniform. Points are still sampled randomly.
    """

    delta = n_sensor_max - n_sensor_min + 1
    pos_sensors = []
    for i in range(n_sensor_patterns):
        lim = n_sensor_min + i % delta
        pos_sensors.append(np.array(optimized_poisson_disk_sampling(min_points=lim, max_points=lim)))

    return pos_sensors


### wind ---------------------------------------------------------------------------------------------------------------
def generate_wind(num_steps, speed_update_interval, bias_pull_strength=0.3):
    """
    This function generates some patterned random wind speed and direction series. Primarily a random walk is used for
    both, speed and direction. However, every few steps a number is drawn and the random walk will be biased around
    that number for an interval. For wind speed this number is weibull and for direction it is uniform.

    Args:
        num_steps: number of timesteps to simulate
        speed_update_interval: length of interval for keeping one bias
        bias_pull_strength: how strongly to pull towards that bias

    Returns:
        wind_speed [0, 1], wind_direction [0, 2pi)
    """

    wind_speed = np.zeros(num_steps)
    wind_direction = np.zeros(num_steps)

    speed_std_dev = 0.02  # reduced speed variability for a smoother random walk
    direction_std_dev = 15  # increased direction variability for looser random walk

    # Initial bias values
    speed_bias = np.random.weibull(a=2) / 2.2  # scale to be (mostly) between 0 and 1
    direction_bias = np.random.uniform(0, 360)

    for t in range(num_steps):
        # Update the bias every `speed_update_interval` steps
        if t % speed_update_interval == 0 and t != 0:
            speed_bias = np.random.weibull(a=2) / 2.2
            direction_bias = np.random.uniform(0, 360)

        # Wind speed smoother random walk around bias with bounds
        wind_speed[t] = (
            wind_speed[t - 1] + bias_pull_strength * (speed_bias - wind_speed[t - 1]) + np.random.normal(0, speed_std_dev)
            if t > 0
            else speed_bias
        )
        wind_speed[t] = np.clip(wind_speed[t], 0, 1)

        # Wind direction looser random walk around bias with wrapping around circle
        wind_direction[t] = (
            wind_direction[t - 1]
            + bias_pull_strength * (direction_bias - wind_direction[t - 1])
            + np.random.normal(0, direction_std_dev)
            if t > 0
            else direction_bias
        )
        wind_direction[t] %= 360

    wind_direction = (wind_direction / 360) * 2 * np.pi
    return wind_speed, wind_direction


def calculate_covariance_blob(wind_direc, wind_speed):
    """
    This function creates a wind statistic over a whole timeframe. The idea is to start with a certain size covariance
    blob. This blob is then subsequently squished (shrunk in crosswind direction) depending on the wind strength. This
    serves as a metric to judge along which axis it is easy to predict the leak and along which it might be harder.
    Used as the shape of the target blobs in the heatmap losses. Returns just a covariance matrix that can be made
    into a pdf using val = multivariate_normal(mean, covar).pdf(pos xy)
    """

    def create_scale_mtx(s):
        return np.array([[1, 0], [0, s]])

    def create_rotation_mtx(alpha):
        return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    def apply_covariance_update(cov, s, alpha):
        # s is the amount that the covariance matrix should be shrunken by ACROSS the wind direction
        # alpha is the wind direction
        scl = create_scale_mtx(s)
        rot = create_rotation_mtx(alpha)
        return rot @ scl @ rot.T @ cov @ rot @ scl @ rot.T

    stddv = 0.15
    covariance = np.array([[stddv, 0], [0, stddv]])

    covariance_new = covariance
    covariance_old = covariance

    for direc, speed in zip(wind_direc, wind_speed):
        scale_weight = 1 - 0.05 * min(1, speed * 2)
        covariance_new = apply_covariance_update(covariance_new, scale_weight, direc)

    covariance = 0.03 * covariance_old + covariance_new

    return covariance


### scenes -------------------------------------------------------------------------------------------------------------
def generate_sample(n_timesteps, sensor_patterns, n_sensor_max, n_source_min, n_source_max, n_leak_min, n_leak_max):
    """Main generation function for ONE scene of n timesteps. Returns a dict of (padded) arrays."""

    # random number of sources and source locations (x, y)
    N_sources = np.random.randint(n_source_min, n_source_max + 1)
    pos_sources = np.random.randn(N_sources, 2)
    pos_sources = my_normalize(pos_sources, padding_factor=1.6)

    # random sensor locations (x, y): just choose from pregenerated set (or make a new one if there is no set)
    pos_sensors = sensor_patterns() if callable(sensor_patterns) else random.choice(sensor_patterns)

    # random number of leaks and random leak locations (x, y)
    n_leaks = np.random.randint(n_leak_min, n_leak_max + 1)
    pos_leaks = pos_sources[:n_leaks, :]  # should be fine like this, because source pos are generated randomly each time

    # time vector
    timesteps = np.linspace(0, n_timesteps - 1, n_timesteps, dtype=int)

    # random wind-speed [0, 1] and wind-direction [0, 2pi] vectors (samples for each time step)
    wind_speed, wind_direc = generate_wind(n_timesteps, speed_update_interval=20, bias_pull_strength=0.3)

    # random emission strengths (currently constant but different for each leak!)
    Q = np.zeros((max(n_leaks, 1), n_timesteps))
    if n_leaks != 0:
        Q = np.array([(el + 1) * np.random.rand() for el in Q[:]])

    # get all the sensor concentration values (instead of getting it at EVERY point xD)
    concentration_at_sensors = np.zeros((n_timesteps, len(pos_sensors)))
    for t in timesteps:
        for sensor_idx, (sx, sy) in enumerate(pos_sensors):
            concentration = 0
            for leak_idx, (lx, ly) in enumerate(pos_leaks):
                # local coords also mean "how much is each point x,y downwind and crosswind" calculate C from that now
                x_local, y_local = coordinate_transform_scalar(sx, sy, lx, ly, wind_direc[t])
                # add up all the concentration from different leaks
                concentration += atmospheric_dispersion(x_local, y_local, Q[leak_idx, t], wind_speed[t])
            concentration_at_sensors[t, sensor_idx] = concentration

    # normalize the concentrations to be within 0 and 1 (might need adjustment!)
    # all padded concentrations are set to -1! (makes it easier for the model to discern!)
    concentration_at_sensors = concentration_at_sensors.round(decimals=5)
    non_zero_mask = concentration_at_sensors != 0
    concentration_at_sensors[non_zero_mask] = (np.log10(concentration_at_sensors[non_zero_mask]) + 5.2) / 6
    concentration_at_sensors[~non_zero_mask] = 0  # 0 stay 0 (necessary??)

    # compute wind statistics, from this the distr is found with: val = multivariate_normal(mean, covar).pdf(pos xy)
    wind_statistics_covariance = calculate_covariance_blob(wind_direc, wind_speed)

    n_sensors = len(pos_sensors)
    sample = {
        "n_timesteps": n_timesteps,
        "time": timesteps,
        "n_sensors": n_sensors,
        "pos_sensors": np.pad(pos_sensors, ((0, n_sensor_max - n_sensors), (0, 0)), constant_values=0),
        "con_sensors": np.pad(concentration_at_sensors, ((0, 0), (0, n_sensor_max - n_sensors)), constant_values=-1),
        "msk_sensors": np.concatenate([np.zeros([n_sensors]), np.ones([n_sensor_max - n_sensors])]),
        "n_sources": len(pos_sources),
        "pos_sources": np.pad(pos_sources, ((0, n_source_max - len(pos_sources)), (0, 0)), constant_values=0),
        "n_leaks": n_leaks,
        "Q_leaks": np.pad(Q.T, ((0, 0), (0, n_leak_max - max(1, n_leaks))), constant_values=0),
        "pos_leaks": np.pad(pos_leaks, ((0, n_leak_max - n_leaks), (0, 0)), constant_values=0),
        "msk_leaks": np.concatenate([np.zeros([n_leaks]), np.ones([n_leak_max - n_leaks])]),
        "wind_speed": wind_speed,
        "wind_direc": wind_direc,
        "wind_covar": wind_statistics_covariance,
    }

    return sample


def concentration_field(sample, resolution=128):
    """
    Raw (not log-normalized) concentration over the whole [-1, 1] plane for every timestep of a scene. Only needed for
    looking at scenes, the models never see this. Returns x, y, Z with Z of shape [n_timesteps, resolution, resolution].
    """

    x, y = np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(sample["time"]), resolution, resolution))

    for t in sample["time"]:
        for idx in range(int(sample["n_leaks"])):
            px, py = sample["pos_leaks"][idx]
            X_local, Y_local = coordinate_transform_array(X, Y, px, py, sample["wind_direc"][t])
            Z[t] += atmospheric_dispersion(X_local, Y_local, sample["Q_leaks"][t, idx], sample["wind_speed"][t])

    return x, y, Z


### collections --------------------------------------------------------------------------------------------------------
def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def _generate_chunk(args):
    n_samples, seed, kwargs = args
    set_seed(seed)
    return [generate_sample(**kwargs) for _ in range(n_samples)]


def generate_collection(
    n_samples,
    n_timesteps=100,
    n_sensor_patterns=50,
    n_sensor_min=6,
    n_sensor_max=10,
    n_source_min=15,
    n_source_max=30,
    n_leak_min=1,
    n_leak_max=1,
    seed=None,
    n_workers=1,
    chunk_size=100,
    progress=True,
):
    """
    Generates a list of scenes that all draw their sensor layout from the same small set of pregenerated patterns
    (n_sensor_patterns = 0: a new layout for every scene).
    The generation of one scene is a slow python loop, so the work can be spread over a few processes.
    """

    set_seed(seed)
    if n_sensor_patterns:
        sensor_patterns = generate_sensor_patterns(n_sensor_patterns, n_sensor_min, n_sensor_max)
    else:
        # no fixed set: every scene gets its own new layout
        sensor_patterns = partial(optimized_poisson_disk_sampling, n_sensor_min, n_sensor_max)

    kwargs = dict(
        n_timesteps=n_timesteps,
        sensor_patterns=sensor_patterns,
        n_sensor_max=n_sensor_max,
        n_source_min=n_source_min,
        n_source_max=n_source_max,
        n_leak_min=n_leak_min,
        n_leak_max=n_leak_max,
    )

    # every chunk gets its own seed, so the result does not depend on the number of workers
    sizes = [chunk_size] * (n_samples // chunk_size) + ([n_samples % chunk_size] if n_samples % chunk_size else [])
    seeds = [None if seed is None else seed + 1 + i for i in range(len(sizes))]
    jobs = [(size, chunk_seed, kwargs) for size, chunk_seed in zip(sizes, seeds)]

    collection = []
    with tqdm(total=n_samples, desc="Generating Data", ncols=120, disable=not progress) as progress_bar:
        if n_workers > 1:
            with Pool(n_workers) as pool:
                for chunk in pool.imap(_generate_chunk, jobs):
                    collection.extend(chunk)
                    progress_bar.update(len(chunk))
        else:
            for job in jobs:
                chunk = _generate_chunk(job)
                collection.extend(chunk)
                progress_bar.update(len(chunk))

    return collection


def default_filename(n_samples):
    return f"{time.strftime('%Y-%m-%d--%H-%M-%S')}--X{n_samples:05}.h5"


def write_h5(collection, path):
    """One group per scene (item_0, item_1, ...), one dataset per key."""
    with h5py.File(path, "w") as f:
        for i, item in enumerate(tqdm(collection, desc="Writing h5", ncols=120)):
            group = f.create_group(f"item_{i}")
            for key, array in item.items():
                group.create_dataset(key, data=array)
