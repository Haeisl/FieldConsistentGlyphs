import numpy as np

from SharedData import shared_data


def generate_grid_points(bounds, grid_dims):
    min_x, max_x, min_y, max_y = bounds[0], bounds[1], bounds[2], bounds[3]
    rows, cols = int(grid_dims[0]), int(grid_dims[1])

    x = np.linspace(min_x + (max_x - min_x) / (cols + 1), max_x - (max_x - min_x) / (cols + 1), cols)
    y = np.linspace(min_y + (max_y - min_y) / (rows + 1), max_y - (max_y - min_y) / (rows + 1), rows)

    xv, yv = np.meshgrid(x,y)

    grid_points = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(rows * cols)])
    return grid_points


def clip_point(point):
    bounds = shared_data.get("bounds")
    point = np.clip(point, bounds[::2], bounds[1::2])
    point[2] = 0.0

    return point


def bilinear_interpolation(point):
    # Get image data object from shared data
    image_data = shared_data.get("image_data")
    # Get essential grid properties
    dimensions = np.array(image_data.GetDimensions())
    spacing = np.array(image_data.GetSpacing())
    origin = np.array(image_data.GetOrigin())

    # Clip point to be within the bounds
    point = clip_point(point)

    # Calculate the indices for the corners of the cell containing the point
    indices = np.floor((point[:2] - origin[:2]) / spacing[:2]).astype(int)
    indices = np.clip(indices, 0, dimensions[:2] - 2)

    # Compute the fractional part within the cell
    t = (point[:2] - (origin[:2] + indices * spacing[:2])) / spacing[:2]

    data_array = np.array(image_data.GetPointData().GetArray(0))
    index_flat = lambda idx: idx[1] * dimensions[0] + idx[0]

    q11 = data_array[index_flat(indices)]
    q21 = data_array[index_flat(indices + [1, 0])]
    q12 = data_array[index_flat(indices + [0, 1])]
    q22 = data_array[index_flat(indices + [1, 1])]

    # Perform bilinear interpolation weighted by t
    interpolated = (q11 * (1 - t[0]) * (1 - t[1]) +
                    q21 * t[0] * (1 - t[1]) +
                    q12 * (1 - t[0]) * t[1] +
                    q22 * t[0] * t[1])

    # Append zero to the interpolated result to form a 3D vector
    return np.append(interpolated, 0)


def get_orthogonal2D(vector):
    # return the orthogonal of a 2D vector (in 3D space, with z component = 0)
    return np.array([vector[1], -vector[0], 0.])


def magnitude(vector):
    return np.linalg.norm(vector)


def normalize_if_requested(vector):
    normalize_requested = shared_data.get("normalize")
    mag = magnitude(vector)

    if not normalize_requested or mag == 0:
        return vector

    return vector / mag


def rk4_step(cur, invert, ortho=False):
    stepsize = shared_data.get("stepsize")
    scaling = shared_data.get("scaling")

    k1 = bilinear_interpolation(cur)

    mid_point_1 = cur + k1 * 0.5
    k2 = bilinear_interpolation(mid_point_1)

    mid_point_2 = cur + k2 * 0.5
    k3 = bilinear_interpolation(mid_point_2)

    end_point = cur + k3
    k4 = bilinear_interpolation(end_point)

    delta_v = (-1 if invert else 1) * (k1 + 2*k2 + 2*k3 + k4) / 6.

    if ortho:
        delta_v = get_orthogonal2D(delta_v)

    mag = magnitude(delta_v)
    delta_v = normalize_if_requested(delta_v)

    next = cur + delta_v * stepsize * scaling
    next = clip_point(next)

    return next, mag


def euler_step(cur, invert, ortho=False):
    stepsize = shared_data.get("stepsize")
    scaling = shared_data.get("scaling")

    delta_v = (-1 if invert else 1) * bilinear_interpolation(cur)

    if ortho:
        delta_v = get_orthogonal2D(delta_v)

    mag = magnitude(delta_v)
    delta_v = normalize_if_requested(delta_v)

    next = cur + delta_v * stepsize * scaling
    next = clip_point(next)

    return next, mag


def calculate_arc_length(points):
    points = np.array(points)
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    return np.sum(distances)


def bezier_points(steps, p1, p1_v, p2, p2_v):
    curve_points = []

    for t in steps:
        next_p = (1 - t)**3 * p1 + 3 *(1 - t)**2 * t * p1_v + 3 * (1 - t) * t**2 * p2_v + t**3 * p2
        curve_points.append(next_p)

    return curve_points