import numpy as np
import re
import random
from scipy.spatial.distance import pdist, squareform
import math

def matches_layout_pattern(layout: str) -> bool:
    """
    Checks if the layout string matches the pattern

    Args:
        layout (str): the layout string

    Returns:
        bool: True if the layout string matches the pattern, False otherwise
    """

    return bool(re.match(r'^([sdchtr])(\d*)$', layout))


def rotate(rot_center, coords, angle):
    """ Rotate a set of 2D points counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        
        args:
        rot_center: tuple, the center of the rotation
        coords: (n, 2) np.array, the coordinates of the points to rotate
        angle: float, the angle of the rotation in radians
    """
    # create the rotation matrix
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    
    # translate the point to the origin
    new_coords = np.array(coords) - np.array(rot_center)
    
    # rotate the points around the origin
    new_coords = np.matmul(R, new_coords.T).T
    
    # translate the points back
    new_coords += np.array(rot_center)

    return new_coords

def layout_gen(layout: str, D: float) -> np.ndarray:
    """
    Generates a layout based on the given layout string and spacing

    Args:
        layout (str): the layout string; can be s (square), d (diamond), c (circle), h (hexagon), t (triangle) or r (random)
        D (float): the spacing between turbines

    Returns:
        np.ndarray: the layout
    """

    match = re.match(r'^([sdchtr])(\d*)$', layout)

    if not match:
        raise ValueError('Invalid layout string; must be of form [s|d|c|h|t|r][1-99]')
    else:
        shape, n = match.groups()
        n = int(n)

        if n > 99 or n < 1:
            raise ValueError('Invalid layout string; must be of form [s|d|c|h|t|r][1-99]')

        if shape == 's':

            h = n * D / 2
            layout = np.array([[i*D-h, j*D-h] for i in range(n) for j in range(n)])

        elif shape == 'd':

            spacing = np.sqrt(0.5 * D**2)
            points = []

            for i in range(1, 2*n):
                w = i if i < n else 2*n-i
                y = (n - i) * spacing
                x_offset = spacing / 2
                n_s, n_f = divmod(w-1, 2)
                left_limit = -n_s * spacing - n_f * x_offset
                for j in range(w):
                    x = left_limit + j * spacing
                    points.append([x, y])

            layout = np.array(points)

        elif shape == 'c':

            center_point = np.array([0, 0])
            layout = np.array([center_point])

            n_r = n - 1

            for i in range(1, n_r+1):
                n_p = 6 * i
                dr = D * i
                dtheta = 2 * np.pi / n_p
                for j in range(n_p):
                    angle = j * dtheta
                    point = np.array([dr * np.cos(angle), dr * np.sin(angle)])
                    layout = np.vstack([layout, point])

        elif shape == 'h':

            spacing = np.sqrt(0.5 * D**2)
            points = []

            for i in range(0, 2*n-1):
                w = n + i if i < n else 3*n-2 - i
                y = (n - 1 - i) * spacing
                x_offset = spacing / 2
                n_s, n_f = divmod(w-1, 2)
                left_limit = -n_s * spacing - n_f * x_offset
                for j in range(w):
                    x = left_limit + j * spacing
                    points.append([x, y])

            layout = np.array(points)

        elif shape == 't':

            spacing = np.sqrt(0.5 * D**2)
            points = []

            for i in range(0, n):
                w = n - i
                y = i * spacing
                x_offset = spacing / 2
                n_s, n_f = divmod(w-1, 2)
                left_limit = -n_s * spacing - n_f * x_offset
                for j in range(w):
                    x = left_limit + j * spacing
                    points.append([x, y])

            layout = np.array(points)

        elif shape == 'r':

            min_dist = D
            farm_lw_ratio = np.random.uniform(0.5, 2)
                
            num_y = np.int32(np.sqrt(n / farm_lw_ratio))
            num_x = np.int32(np.ceil(n / num_y))
            width = np.ceil((num_y - 1) * 2.5 * min_dist)
            length = np.ceil((num_x - 1) * 2.5 * min_dist)

            # create regularly spaced points for the boundary of the overall rectangular domain
            x = np.linspace(0., length, num_x, dtype=np.float32)
            y = np.linspace(0., width, num_y, dtype=np.float32)

            base_coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
            
            # delete random points from the base coordinates if length exceeds n_points
            if len(base_coords) > n:
                # pick random indices to delete
                indices = np.random.choice(len(base_coords), len(base_coords) - n, replace=False)
                base_coords = np.delete(base_coords, indices, axis=0)

            # Perturb all points with a random noise
            r = min_dist * np.sqrt(np.random.uniform(0, 1, (len(base_coords), 1)))
            theta = np.random.uniform(0, 2 * np.pi, (len(base_coords), 1))
            perturbations = np.concatenate((r * np.cos(theta), r * np.sin(theta)), axis=1)
            base_coords += perturbations

            # compute the resulting spacing between points
            # min_proximity = np.min(pdist(base_coords))
            proximities_matrix = squareform(pdist(base_coords))
            np.fill_diagonal(proximities_matrix, np.inf)
            closest_proximities = np.min(proximities_matrix, axis=1)
            min_proximity = np.mean(closest_proximities)

            # compute scaling factor
            factor = min_dist / min_proximity

            # scale the coordinates
            base_coords *= factor
            width *= factor
            length *= factor
            
            # randomly rotate the rectangle around (0,0)
            alpha = random.uniform(-np.pi/4, np.pi/4)
            layout = rotate((0, 0), base_coords, alpha)

        return layout
