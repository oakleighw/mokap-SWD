from typing import Optional, Any, Dict, Sequence, Tuple, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from mokap.geometry.backend import xp, ArrayLike
from mokap.geometry.projective import unproject
from mokap.geometry import intersect_rays, transform_points, homogenize

CUSTOM_COLORS = ['#9B5DE5', '#EF476F', '#FFD166', '#00BBF9', '#00F5D4', '#118ab2', '#073b4c', '#ee6c4d']


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    import matplotlib.colors as colors
    return colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )


def init_3d_plot(ax: Optional[Axes3D] = None, figsize: Tuple[int, int] = (12, 12)) -> Axes3D:
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    return ax


def _set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# Two tiny geometry helpers

def scene_focal_point(T_c2w: ArrayLike) -> np.ndarray:
    T = xp.asarray(T_c2w)
    centers = T[:, :3, 3]
    axes = T[:, :3, 2]  # +Z axis
    return np.asarray(intersect_rays(centers, axes))


def calculate_depths(T_c2w: ArrayLike, focal_point: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Calculates the depth for each camera frustum relative to the focal point.
    """
    centers = xp.asarray(T_c2w)[:, :3, 3]
    # (C, 3) - (3,) -> (C, 3)
    diffs = centers - xp.asarray(focal_point)
    dists = xp.linalg.norm(diffs, axis=1)

    depths = dists * scale
    depths = xp.where(depths < 1e-3, 1.0, depths)
    return np.asarray(depths)


# Plotting functions

def draw_trust_volume(
        trust_volume: Dict[str, Any],
        ax: Axes3D,
        center_point: Optional[np.ndarray] = None,
        view_transform: Optional[np.ndarray] = None,
        color: str = '#96d895',
        alpha: float = 0.05
) -> Axes3D:

    if np.isscalar(list(trust_volume.values())[0]):
        # Extents provided: Center around the provided center_point (focal point)
        if center_point is not None:
            cx, cy, cz = center_point
        else:
            cx, cy, cz = 0, 0, 0

        sx, sy, sz = trust_volume['x'], trust_volume['y'], trust_volume['z']
        min_x, max_x = cx - sx / 2, cx + sx / 2
        min_y, max_y = cy - sy / 2, cy + sy / 2
        min_z, max_z = cz - sz / 2, cz + sz / 2
    else:
        # Min/max tuples provided: absolute coordinates
        min_x, max_x = trust_volume['x']
        min_y, max_y = trust_volume['y']
        min_z, max_z = trust_volume['z']

    corners = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z],
    ])

    corners = transform_points(corners, view_transform)

    faces_idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 7, 3, 0]]
    verts = [[corners[i] for i in face] for face in faces_idx]

    poly = Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors=color, alpha=alpha)
    ax.add_collection3d(poly)

    # Draw center marker
    center = np.mean(corners, axis=0)
    ax.scatter(*center, color=color, marker='+', s=50, alpha=0.5)

    return ax


def draw_cameras(
        T_c2w: ArrayLike,
        K: ArrayLike,
        D: ArrayLike,
        depths: ArrayLike,
        ax: Axes3D,
        view_transform: Optional[np.ndarray] = None,
        cameras_names: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (1440, 1080)
) -> Axes3D:

    T_c2w = np.asarray(T_c2w)
    K = np.asarray(K)
    D = np.asarray(D)
    depths = np.asarray(depths)

    T_c2w_viz = view_transform @ T_c2w if view_transform is not None else T_c2w

    C = T_c2w.shape[0]
    if colors is None:
        colors = CUSTOM_COLORS * (C // len(CUSTOM_COLORS) + 1)
    if cameras_names is None:
        cameras_names = [f"Cam {i}" for i in range(C)]

    w, h = image_size
    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], dtype=np.float32)
    corners_2d_batch = np.tile(corners_2d[None, ...], (C, 1, 1))

    # Unproject to camera's *local* frame
    T_identity = np.eye(4)[None, ...].repeat(C, axis=0)

    frustum_pts_local = unproject(
        corners_2d_batch, depths, K, T_identity, D, distortion_model='standard'
    )

    for i in range(C):
        col = colors[i]
        cam_center = T_c2w_viz[i, :3, 3]

        R = T_c2w_viz[i, :3, :3]
        pts_local = frustum_pts_local[i]
        pts_world = (R @ pts_local.T).T + cam_center

        ax.scatter(*cam_center, color=col, s=20)
        ax.text(*cam_center, s=f" {cameras_names[i]}", color=col, fontsize=8)

        face = Poly3DCollection([pts_world], facecolors=col, alpha=0.1, linewidths=1.2, edgecolors=col)
        ax.add_collection3d(face)

        for p in pts_world[:-1]:
            ax.plot(*zip(cam_center, p), color=col, linestyle='-', linewidth=1.0, alpha=0.3)

    return ax


def draw_observations(
        points2d: ArrayLike,
        T_c2w: ArrayLike,
        K: ArrayLike,
        D: ArrayLike,
        depths: ArrayLike,
        ax: Axes3D,
        view_transform: Optional[np.ndarray] = None,
        visibility_mask: Optional[ArrayLike] = None,
        colors: Optional[Sequence[str]] = None
) -> Axes3D:

    points2d = np.asarray(points2d)
    T_c2w = np.asarray(T_c2w)
    K = np.asarray(K)
    D = np.asarray(D)
    depths = np.asarray(depths)

    C, N = points2d.shape[:2]
    T_c2w_viz = view_transform @ T_c2w if view_transform is not None else T_c2w

    T_identity = np.eye(4)[None, ...].repeat(C, axis=0)
    pts_local = unproject(points2d, depths, K, T_identity, D, distortion_model='standard')

    if colors is None:
        colors = CUSTOM_COLORS

    for c in range(C):
        col = colors[c % len(colors)]
        R_viz = T_c2w_viz[c, :3, :3]
        t_viz = T_c2w_viz[c, :3, 3]
        cam_pts = pts_local[c]

        mask = np.isfinite(points2d[c, :, 0])
        if visibility_mask is not None:
            mask = mask & visibility_mask[c]

        valid_pts = cam_pts[mask]

        if len(valid_pts) > 0:
            pts_world = (R_viz @ valid_pts.T).T + t_viz
            ax.scatter(pts_world[:, 0], pts_world[:, 1], pts_world[:, 2],
                       color=col, marker='x', s=10, alpha=0.9, depthshade=False)
    return ax


def draw_points(
        points3d: ArrayLike,
        ax: Axes3D,
        view_transform: Optional[np.ndarray] = None,
        errors: Optional[ArrayLike] = None,
        default_color: str = 'k',
        worst_point_idx: Optional[int] = None
) -> Axes3D:

    points3d = np.asarray(points3d)
    if errors is not None:
        errors = np.asarray(errors)

    pts_viz = transform_points(points3d, view_transform)
    xs, ys, zs = pts_viz.T

    if errors is not None:
        if errors.ndim == 2:
            errors = np.nanmean(errors, axis=0)

        cmap = truncate_colormap(plt.cm.brg, 0.45, 1.0).reversed()
        norm = matplotlib.colors.Normalize(vmin=0, vmax=5.0)
        ax.scatter(xs, ys, zs, c=errors, cmap=cmap, norm=norm, s=10, alpha=1.0, label='3D points')
    else:
        ax.scatter(xs, ys, zs, c=default_color, s=10, alpha=0.8, label='3D points')

    if worst_point_idx is not None and 0 <= worst_point_idx < len(pts_viz):
        wx, wy, wz = pts_viz[worst_point_idx]
        ax.scatter(wx, wy, wz, c='red', marker='x', s=25, linewidth=2, zorder=100, label='Worst error')

    return ax


def draw_rays(
        points2d: ArrayLike,
        points3d: ArrayLike,
        T_c2w: ArrayLike,
        K: ArrayLike,
        D: ArrayLike,
        depths: ArrayLike,
        ax: Axes3D,
        view_transform: Optional[np.ndarray] = None,
        visibility_mask: Optional[ArrayLike] = None,
        colors: Optional[Sequence[str]] = None,
        worst_point_idx: Optional[int] = None
) -> Axes3D:
    """
    Draws the camera rays extending to the depth of the 3D point.
    Visualizes alignment errors by drawing a circle where the ray ends vs where the point is.
    """

    points3d = np.asarray(points3d)
    points2d = np.asarray(points2d)
    T_c2w = np.asarray(T_c2w)
    depths = np.asarray(depths)

    C, N = points2d.shape[:2]

    # Transform geometry to plot reference
    pts_3d_viz = transform_points(points3d, view_transform)
    T_c2w_viz = view_transform @ T_c2w if view_transform is not None else T_c2w

    # Observation points on frustum
    T_identity = np.eye(4)[None, ...].repeat(C, axis=0)
    # Note: we use the frustum depths here to get the 'start' of the ray (on the image plane)
    pts_obs_local = unproject(points2d, depths, K, T_identity, D, distortion_model='standard')

    if colors is None:
        colors = CUSTOM_COLORS

    for c in range(C):
        col = colors[c % len(colors)]

        R_viz = T_c2w_viz[c, :3, :3]
        t_viz = T_c2w_viz[c, :3, 3]

        # Observations on frustum
        pts_obs_viz = (R_viz @ pts_obs_local[c].T).T + t_viz

        segments = []
        ray_ends = []

        for n in range(N):
            if visibility_mask is not None and not visibility_mask[c, n]:
                continue

            p_cam = t_viz
            p_obs = pts_obs_viz[n]  # point on frustum
            p_tri = pts_3d_viz[n]   # triangulated result

            if not np.isfinite(p_obs).all() or not np.isfinite(p_tri).all():
                continue

            # Distance to triangulated point
            dist_to_tri = np.linalg.norm(p_tri - p_cam)

            # Extend the ray from camera through observed point to that distance
            # vector cam->obs
            vec_ray = p_obs - p_cam
            vec_ray_norm = vec_ray / (np.linalg.norm(vec_ray) + 1e-9)

            p_ray_end = p_cam + vec_ray_norm * dist_to_tri

            if worst_point_idx is not None and n == worst_point_idx:
                ax.plot(*zip(p_obs, p_ray_end), color='red', alpha=0.3, linestyle=(0, (5, 5)), linewidth=0.7)
                ax.scatter(*p_ray_end, marker='.', s=100, alpha=0.5, edgecolors='none', facecolors='red')
            else:
                segments.append([p_obs, p_ray_end])
                ray_ends.append(p_ray_end)

        if segments:
            lc = Line3DCollection(segments, colors=col, alpha=0.15, linestyle='-', linewidths=0.5)
            ax.add_collection3d(lc)

        if ray_ends:
            re = np.array(ray_ends)
            ax.scatter(re[:, 0], re[:, 1], re[:, 2], marker='.', s=100, alpha=0.5, edgecolors='none', facecolors=col)

    return ax


def draw_object(
        object_points: ArrayLike,
        object_pose: ArrayLike,
        ax: Axes3D,
        view_transform: Optional[np.ndarray] = None,
        color: str = '#0000ff',
        label: str = 'Ground truth board'
) -> Axes3D:
    """
    Draws a rigid object (e.g. calibration pattern) based on local points and a pose T.
    Also draws a small RGB coordinate frame at the object's origin.
    """
    pts_local = np.asarray(object_points)
    T_obj = np.asarray(object_pose)

    # Transform points: object local -> world
    pts_local_h = homogenize(pts_local)
    pts_world_h = pts_local_h @ T_obj.T
    pts_world = pts_world_h[:, :3]

    # Transform world -> visualisation
    pts_viz = transform_points(pts_world, view_transform)

    ax.scatter(pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2],
               c=color, s=20, marker='+', label=label, alpha=0.8)

    # Draw object gizmo (RGB)
    axis_len = np.max(np.ptp(pts_local, axis=0)) * 0.2  # 20% of object size
    if axis_len < 1e-3:
        axis_len = 0.1

    # Origin X, Y, Z in *object local* frame
    axes_local = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ])

    # Transform axes: object local -> world
    axes_h = homogenize(axes_local)
    axes_world_h = axes_h @ T_obj.T[:, :3]
    axes_world = axes_world_h[:, :3]

    # Transform world -> visualisation
    axes_viz = transform_points(axes_world, view_transform)

    origin = axes_viz[0]
    ax.plot(*zip(origin, axes_viz[1]), color='r', linewidth=2)  # X
    ax.plot(*zip(origin, axes_viz[2]), color='g', linewidth=2)  # Y
    ax.plot(*zip(origin, axes_viz[3]), color='b', linewidth=2)  # Z

    return ax


def visualise_calibration_scene(
        T_c2w: ArrayLike,
        K: ArrayLike,
        D: ArrayLike,
        points3d: Optional[ArrayLike] = None,
        points2d: Optional[ArrayLike] = None,
        visibility_mask: Optional[ArrayLike] = None,
        trust_volume: Optional[Dict] = None,
        camera_names: Optional[List[str]] = None,
        point_errors: Optional[ArrayLike] = None,
        worst_point_idx: Optional[int] = None,
        object_points: Optional[ArrayLike] = None,
        object_pose: Optional[ArrayLike] = None,
        orientation: Optional[str] = None,
        frustum_scale: float = 0.5,
        ax: Optional[Axes3D] = None
) -> Axes3D:

    ax = init_3d_plot(ax)

    # View transform
    T_view = np.eye(4)
    if orientation is not None and orientation == 'upright':
        T_view[1, 1] = -1
        T_view[2, 2] = -1

    # Scene geom
    focal_point = scene_focal_point(T_c2w)
    depths = calculate_depths(T_c2w, focal_point, scale=frustum_scale)

    if trust_volume is not None:
        draw_trust_volume(trust_volume, ax, center_point=focal_point, view_transform=T_view)

    draw_cameras(T_c2w, K, D, depths, ax,
                 view_transform=T_view,
                 cameras_names=camera_names)

    if object_points is not None and object_pose is not None:
        draw_object(object_points, object_pose, ax,
                    view_transform=T_view,
                    color='blue',
                    label='Calibration board')

    if points2d is not None:
        draw_observations(points2d, T_c2w, K, D, depths, ax,
                          view_transform=T_view,
                          visibility_mask=visibility_mask)

    if points3d is not None:
        draw_points(points3d, ax,
                    view_transform=T_view,
                    errors=point_errors,
                    worst_point_idx=worst_point_idx)

    if points3d is not None and points2d is not None:
        draw_rays(points2d, points3d, T_c2w, K, D, depths, ax,
                  view_transform=T_view,
                  visibility_mask=visibility_mask,
                  worst_point_idx=worst_point_idx)

    # Overlay stats
    if point_errors is not None and camera_names is not None:
        stats_text = "Per-camera errors (px):\n"
        for i, cam_name in enumerate(camera_names):
            mean_err = np.nanmean(point_errors[i])
            max_err = np.nanmax(point_errors[i])
            min_err = np.nanmin(point_errors[i])

            stats_text += (f"{cam_name:>10}: Mean={mean_err if np.isfinite(mean_err) else '-':.2f}, "
                           f"Max={max_err if np.isfinite(max_err) else '-':.2f}, "
                           f"Min={min_err if np.isfinite(min_err) else '-':.2f}\n")
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                  family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    _set_axes_equal(ax)

    ax.grid(False)
    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    return ax