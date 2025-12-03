from typing import Optional, Any, Dict, Iterable, Sequence, Union
import matplotlib

import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)

from mokap.geometry.backend import xp, ArrayLike

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mokap.geometry import intersect_rays, unproject

CUSTOM_COLORS = ['#9B5DE5', '#EF476F', '#FFD166', '#00BBF9', '#00F5D4', '#118ab2', '#073b4c', '#ee6c4d']


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    # From https://stackoverflow.com/a/18926541
    import matplotlib.colors as colors
    return colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                                                    cmap(np.linspace(minval, maxval, n)))


def plot_box_3d(
        centre: ArrayLike,
        size: ArrayLike,
        color: str = 'k',
        alpha: float = 0.1,
        ax: Optional[Axes3D] = None,
) -> Axes3D:

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    coords = np.indices((2, 2, 2)).reshape(3, -1).T
    v = (coords - 0.5) * np.array(size) + np.array(centre)

    faces_idx = np.array([
        [0, 1, 3, 2],  # bottom
        [4, 5, 7, 6],  # top
        [0, 1, 5, 4],  # back
        [2, 3, 7, 6],  # front
        [0, 2, 6, 4],  # left
        [1, 3, 7, 5],  # right
    ])
    faces = v[faces_idx]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.1, edgecolors=color, alpha=alpha))

    return ax


def plot_ellipsoid_3d(
        centre: ArrayLike,
        size:   ArrayLike,
        color:  str = 'k',
        alpha:  float = 0.1,
        resolution: int = 30,
        ax:     Optional[Axes3D] = None,
) -> Axes3D:

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    centre = np.asarray(centre)
    radii = np.asarray(size) / 2.0

    # Generate the surface points of a unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale and translate to create the ellipsoid
    x = radii[0] * x + centre[0]
    y = radii[1] * y + centre[1]
    z = radii[2] * z + centre[2]

    # Plot the surface
    ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=4, cstride=4, linewidth=0)

    return ax


def plot_cameras_3d(
        T_c2w:              ArrayLike,
        K:                  ArrayLike,
        D:                  ArrayLike,
        imsizes:            ArrayLike = np.array([1440, 1080]),
        cameras_names:      Optional[Sequence[Any]] = None,
        depth:              Optional[Union[float, ArrayLike]] = None,
        depth_ratio:        float = 0.75,
        colors:             Optional[Sequence[str]] = None,
        trust_volume:       Optional[Dict[str, ArrayLike]] = None,
        ax:                 Optional[Axes3D] = None,
) -> Axes3D:
    """ Matplotlib 3D plot for viewing C cameras, with their frustums, and the global focal point """

    T_c2w = xp.asarray(T_c2w)
    K = xp.asarray(K)
    D = xp.asarray(D)

    if K.ndim != 3 or D.ndim != 2:
        raise ValueError('This function should be called for C cameras!')

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = CUSTOM_COLORS

    C = K.shape[0]

    if cameras_names is None:
        cameras_names = [f'Cam #{c}' for c in range(C)]

    images_sizes = np.asarray(imsizes)
    if images_sizes.ndim == 1:
        images_sizes = np.vstack([images_sizes] * C)

    unit_coords = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],         # need to repeat the first one for Poly3DCollection
        [0.5, 0.5],     # centre point
    ], dtype=np.float32)
    frustums_2d = (images_sizes[:, None, :] * unit_coords[None, :, :]).astype(np.float32)

    # Plot the axes arrows
    axes_length = 10
    ax.quiver(*[0, 0, 0], *[axes_length, 0, 0], color='r', alpha=0.5)
    ax.quiver(*[0, 0, 0], *[0, axes_length, 0], color='g', alpha=0.5)
    ax.quiver(*[0, 0, 0], *[0, 0, axes_length], color='b', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # First, find the shared focal point. The calculation is independent of the initial depth
    # used for back-projection, we only need the normalized direction vectors
    # so we use a dummy depth of 1.0 to get the directions
    frustums_for_direction = unproject(
        frustums_2d, xp.ones(C), K, T_c2w, D, distortion_model='full'
    )
    # TODO: Use new raycasting function

    tvecs_c2w = T_c2w[..., :3, 3]
    directions = frustums_for_direction[:, -1] - tvecs_c2w
    directions_normalised = directions / np.linalg.norm(directions, axis=1)[:, None]
    focal_point = intersect_rays(tvecs_c2w, directions_normalised)

    # Determine the plotting depths
    if depth is None:
        # Automatic mode: depth is 3/4 the distance from each camera to the focal point
        distances_to_focal = xp.linalg.norm(tvecs_c2w - focal_point, axis=1)
        plot_depths = distances_to_focal * depth_ratio
    else:
        # Manual override: use the fixed depth for all cameras
        plot_depths = xp.array([depth] * C)

    # Calculate the final frustums for plotting using the determined depths
    frustums_3d = unproject(
        frustums_2d,
        plot_depths,
        K,
        T_c2w,
        D,
        distortion_model='full'
    )

    for n in range(C):
        col = colors[n]

        # Cameras positions (optical centres)
        ax.scatter(*tvecs_c2w[n], color=col, label=cameras_names[n], alpha=1.0)

        # Frustum plans
        ax.add_collection3d(
            Poly3DCollection([frustums_3d[n, :-1]], facecolors=col, edgecolors=col, linewidths=1, linestyles='-',
                             alpha=0.05))

        # Frustum lines
        for corner in frustums_3d[n, :-2]:
            ax.plot(*np.stack([tvecs_c2w[n], corner]).T, color=col, linestyle='-', linewidth=0.25, alpha=0.5)

        # Optical axis
        ax.plot(*np.stack([tvecs_c2w[n], frustums_3d[n, -1]]).T, color=col, linestyle='--', linewidth=1.0, alpha=0.5)

    ax.scatter(*focal_point, marker='*', color='k', s=25)

    if trust_volume is not None:
        # Check the format of the trust_volume dict to determine behavior
        first_value = next(iter(trust_volume.values()))

        if np.isscalar(first_value):
            # Values are extents (sizes) so we center the box on the shared focal point
            volume_centre = focal_point
            volume_size = [trust_volume['x'], trust_volume['y'], trust_volume['z']]
        else:
            # Values are ranges (min, max). Calculate the box's own center and size
            x_min, x_max = trust_volume['x']
            y_min, y_max = trust_volume['y']
            z_min, z_max = trust_volume['z']
            volume_centre = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            volume_size = [x_max - x_min, y_max - y_min, z_max - z_min]

        ax = plot_box_3d(centre=volume_centre, size=volume_size, ax=ax, color='#96d895', alpha=0.05)
        ax.scatter(*volume_centre, marker='s', color='#96d895', s=25)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    return ax


def plot_points_3d(
        points3d:       ArrayLike,
        points_names:   Optional[Sequence[Any]] = None,
        errors:         Optional[ArrayLike] = None,
        color:          str = 'k',
        label:          str = '3D points',
        ax:             Optional[Axes3D] = None,
) -> Axes3D:
    """ Matplotlib 3D plot for points, their names and the associated errors """

    points3d = np.asarray(points3d)

    if points3d.ndim != 2:
        raise ValueError('This function should be called for N 3D points!')

    if errors is not None:
        errors = np.asarray(errors)
        assert points3d.shape[0] == errors.shape[0]

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = points3d.T

    if errors is not None:
        colormap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)
        pts_scatter = ax.scatter(xs, ys, zs,
                                 c=errors, cmap=colormap, norm=normalize,
                                 marker='o', label=label, alpha=0.5)
    else:
        pts_scatter = ax.scatter(xs, ys, zs,
                                 color=color,
                                 marker='o', label=label, alpha=0.5)

    if points_names is not None:
        for p, name in enumerate(points_names):
            if errors is not None:
                c = pts_scatter.to_rgba(errors[p]) if np.isfinite(errors[p]) else color
            else:
                c = color
            ax.text(xs[p], ys[p], zs[p], f"  {name}", c=c, alpha=0.8, fontweight='bold')

    ax.legend()

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def plot_object_3d(
        object_points:  ArrayLike,
        object_pose:    ArrayLike,
        color:          str = 'blue',
        label:          str = 'Object Ground Truth',
        ax:             Optional[Axes3D] = None,
) -> Axes3D:
    """
    Plots a 3D object (like a calibration pattern) given its local points and its pose in the world

    Args:
        object_points: The (N, 3) points of the board in its own local coordinate system (often with z=0)
        object_pose: The b2w object pose as a T matrix
        color: The color for the board points
        label: The legend label for the board points
        ax: Optional existing Matplotlib Axes3D object
    """

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    # Convert local points to homogeneous coordinates
    local_points_hom = np.hstack([
        np.asarray(object_points),
        np.ones((np.asarray(object_points).shape[0], 1))
    ])

    # Apply the transformation to get the points in world coordinates
    world_points_hom = (object_pose @ local_points_hom.T).T
    world_points_3d = world_points_hom[:, :3]

    ax = plot_points_3d(
        points3d=world_points_3d,
        color=color,
        label=label,
        ax=ax
    )

    return ax


def plot_points2d_3d(
        points2d:     ArrayLike,
        T_c2w:        ArrayLike,
        K:            ArrayLike,
        D:            ArrayLike,
        depth:        float = 10.0,
        points_names: Optional[Iterable[Any]] = None,
        errors:       Optional[ArrayLike] = None,
        colors:       Optional[str] = None,
        ax:           Optional[Axes3D] = None,
) -> Axes3D:

    points2d = np.asarray(points2d)
    T_c2w = xp.asarray(T_c2w)
    K = xp.asarray(K)
    D = xp.asarray(D)

    if points2d.ndim != 3 or K.ndim != 3 or D.ndim != 2:
        raise ValueError('This function should be called for CxN 2D points!')

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

    C = points2d.shape[0]

    if colors is None:
        colors = CUSTOM_COLORS

    points2d_3d = unproject(points2d, xp.asarray([depth] * C), K, T_c2w, D, distortion_model='full')

    for n in range(C):

        xs, ys, zs = points2d_3d[n].T

        if errors is not None:
            colormap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)

            pts_scatter = ax.scatter(xs, ys, zs, s=10,
                                     c=errors[n], cmap=colormap, norm=normalize,
                                     marker='.', label='3D points', alpha=0.5)
        else:
            pts_scatter = ax.scatter(xs, ys, zs, s=10,
                                     c=colors[n],
                                     marker='.', label='3D points', alpha=0.5)

        if points_names is not None:
            for p, name in enumerate(points_names):
                if errors is not None:
                    c = pts_scatter.to_rgba(errors[p]) if np.isfinite(errors[p]) else colors[n]
                else:
                    c = colors[n]
                ax.text(xs[p], ys[p], zs[p], f"  {name}", c=c, alpha=0.8, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def plot_triangulation_scene(
        points3d:           ArrayLike,
        points2d:           ArrayLike,
        T_c2w:              ArrayLike,
        K:                  ArrayLike,
        D:                  ArrayLike,
        visibility_mask:    Optional[ArrayLike] = None,
        points_names:       Optional[Sequence[Any]] = None,
        errors:             Optional[ArrayLike] = None,
        cameras_names:      Optional[Sequence[Any]] = None,
        imsizes:            ArrayLike = np.array([1440, 1080]),
        frustums_depth:     float = 0.9,
        detections_depth:   float = 0.95,
        colors:             Optional[Sequence[str]] = None,
        trust_volume:       Optional[Dict[str, ArrayLike]] = None,
        object_pose:        Optional[ArrayLike] = None,
        object_points:      Optional[ArrayLike] = None,
        ax:                 Optional[Axes3D] = None,
        worst_point_idx:    Optional[int] = None,
        camera_stats:       Optional[Dict[str, Dict[str, float]]] = None,
) -> Axes3D:
    """
    Comprehensive 3D plot of a triangulation scene
    """

    points3d = xp.asarray(points3d)
    points2d = xp.asarray(points2d)
    T_c2w = xp.asarray(T_c2w)
    K = xp.asarray(K)
    D = xp.asarray(D)

    if ax is None:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = CUSTOM_COLORS

    points2d_plot = points2d.copy()
    if visibility_mask is not None:
        points2d_plot[~xp.asarray(visibility_mask)] = xp.nan

    # Plot cameras and frustums
    ax = plot_cameras_3d(
        T_c2w, K, D,
        cameras_names=cameras_names,
        imsizes=imsizes,
        depth_ratio=frustums_depth,
        trust_volume=trust_volume,
        colors=colors,
        ax=ax
    )

    # Plot final triangulated 3D points
    ax = plot_points_3d(
        points3d,
        points_names=points_names,
        errors=errors,
        color='black',
        label='Triangulated points',
        ax=ax
    )

    # Optional ground truth object
    if all(arg is not None for arg in [object_pose, object_points]):
        ax = plot_object_3d(
            object_points=object_points,
            object_pose=object_pose,
            color='blue',
            label='Ground truth',
            ax=ax
        )

    # Back-projected rays
    tvecs_c2w = T_c2w[..., :3, 3]
    cam_to_point_vectors = points3d[None, :, :] - tvecs_c2w[:, None, :]
    depths_to_3d_points = xp.linalg.norm(cam_to_point_vectors, axis=2)
    plot_depths = depths_to_3d_points * detections_depth

    points2d_in_3d = unproject(points2d_plot, plot_depths, K, T_c2w, D, distortion_model='simple')
    points2d_in_3d = np.asarray(points2d_in_3d)

    C, N, _ = points2d_in_3d.shape
    for c in range(C):
        for n in range(N):
            if np.all(np.isfinite(points2d_in_3d[c, n, :])):
                start_point = tvecs_c2w[c]
                end_point = points2d_in_3d[c, n, :]

                # Highlight worst point
                is_worst = (worst_point_idx is not None) and (n == worst_point_idx)
                if is_worst:
                    ax.scatter(*end_point, c='red', marker='x', s=25, linewidth=2, zorder=10)
                    ax.plot(*np.stack([start_point, end_point]).T, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
                else:
                    ax.scatter(*end_point, c=colors[c], marker='.', alpha=0.7, s=20)
                    ax.plot(*np.stack([start_point, end_point]).T, color=colors[c], linestyle=':', linewidth=0.7, alpha=0.6)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [Line2D([0], [0], marker='.', color='gray', label='Back-projected rays',
                              markerfacecolor='gray', markersize=8, linestyle='None')]
    if worst_point_idx is not None:
        legend_elements.append(Line2D([0], [0], marker='x', color='red', label='Worst error',
                                      markerfacecolor='red', markersize=10, linestyle='None'))
    ax.legend(handles=handles + legend_elements)

    # Overlay stats
    if camera_stats is not None:
        stats_text = "Per-camera errors (px):\n"
        for cam_name, metrics in camera_stats.items():
            if np.isfinite(metrics['mean']):
                stats_text += f"{cam_name:>10}: Mean={metrics['mean']:.2f}, Max={metrics['max']:.2f}\n"
            else:
                stats_text += f"{cam_name:>10}: No Data\n"

        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                  fontsize=9, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    return ax