"""Verify the fixed fundamental-sector triangulation.

Requires SciPy and Shapely 2.1 or newer.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from shapely import coverage_is_valid, polygons


MAX_EXACT_FLOAT_INTEGER = 2**53


def exact_integer_matrix(values, rows, name):
    """Convert a `rows`-by-n MAT array `values` named `name` to exact integers."""
    if values.ndim != 2 or values.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains a non-finite value")
    if np.any(np.abs(values) > MAX_EXACT_FLOAT_INTEGER):
        raise ValueError(f"{name} contains an integer outside the exact Float64 range")
    if not np.all(values == np.trunc(values)):
        raise ValueError(f"{name} contains a non-integer value")
    return values.astype(np.int64).T


def load_triangulation(path):
    """Load `path`; return integer nodes, zero-based triangles, and lattice size.

    Triangle indices in the MAT file start counting at one.
    """
    data = loadmat(path)
    nodes = exact_integer_matrix(data["node_keys"], 2, "node_keys")
    triangles = exact_integer_matrix(data["triangles"], 3, "triangles")
    lattice = exact_integer_matrix(data["lattice_size"], 1, "lattice_size")
    if lattice.shape != (1, 1):
        raise ValueError("lattice_size must be one integer")
    if np.any(triangles < 1) or np.any(triangles > len(nodes)):
        raise ValueError("triangle index outside the node array")
    return nodes, triangles - 1, int(lattice[0, 0])


def check_lattice_data(nodes, triangles, lattice_size):
    """Check exact sector containment and node uniqueness; return nothing.

    `nodes` and zero-based `triangles` use lattice size `lattice_size`.
    """
    if len(np.unique(nodes, axis=0)) != len(nodes):
        raise ValueError("duplicate lattice node")
    x, y = nodes[:, 0], nodes[:, 1]
    inside = (0 <= x) & (x <= lattice_size) & (-x <= y) & (y <= 0)
    if not np.all(inside):
        raise ValueError("node outside the fundamental sector")
    canonical = np.sort(triangles, axis=1)
    if len(np.unique(canonical, axis=0)) != len(triangles):
        raise ValueError("duplicate triangle")


def check_exact_areas(nodes, triangles, lattice_size):
    """Check exact positive orientations and total sector area; return nothing."""
    total_area = 0
    for index, (a, b, c) in enumerate(triangles):
        ax, ay = map(int, nodes[a])
        bx, by = map(int, nodes[b])
        cx, cy = map(int, nodes[c])
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if area <= 0:
            raise ValueError(f"triangle {index + 1} is not counterclockwise")
        total_area += area
    if total_area != lattice_size**2:
        raise ValueError("triangle areas do not sum to the sector area")


def check_geometric_conformity(nodes, triangles):
    """Check that indexed `nodes` cover the sector; return nothing."""
    triangle_polygons = polygons(nodes[triangles])
    if not coverage_is_valid(triangle_polygons):
        raise ValueError("triangles are not a conforming polygonal coverage")


def check_radial_boundary_symmetry(nodes, triangles):
    """Check exact reflection symmetry of the two radial boundary meshes."""
    upper_vertices, lower_vertices = set(), set()
    for triangle in triangles:
        for vertex in triangle:
            x, y = nodes[vertex]
            if y == 0:
                upper_vertices.add(x)
            if x + y == 0:
                lower_vertices.add(x)
    if upper_vertices != lower_vertices:
        raise ValueError("radial boundary meshes are not reflection-symmetric")


def main():
    """Verify the hardcoded triangulation and print its dimensions."""
    path = Path(__file__).with_name("triangulation.mat")
    nodes, triangles, lattice_size = load_triangulation(path)
    check_lattice_data(nodes, triangles, lattice_size)
    check_exact_areas(nodes, triangles, lattice_size)
    check_geometric_conformity(nodes, triangles)
    check_radial_boundary_symmetry(nodes, triangles)
    print(f"verified {len(nodes)} nodes and {len(triangles)} triangles")


if __name__ == "__main__":
    main()
