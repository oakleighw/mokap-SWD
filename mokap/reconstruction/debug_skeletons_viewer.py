import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pickle
from pathlib import Path
from mokap.utils import fileio


def draw_skeletons(frame_data: dict, bones: list, ax: plt.Axes):

    skeletons = frame_data.get("skeletons", [])
    color_map = plt.get_cmap('tab20', 20)

    # we return the artists we create so they can be removed later
    artists = []

    for i, skel in enumerate(skeletons):
        keypoints = skel['keypoints']
        track_id = skel.get('track_id', -1)
        color = color_map(track_id % color_map.N) if track_id != -1 else 'gray'

        # bones
        for kp1_name, kp2_name in bones:
            if kp1_name in keypoints and kp2_name in keypoints:
                p1 = keypoints[kp1_name]
                p2 = keypoints[kp2_name]
                line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=0.8, lw=2)
                artists.append(line)

        # keypoints
        points_arr = np.array(list(keypoints.values()))
        scatter = ax.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], c=[color], s=10, alpha=0.8)
        artists.append(scatter)

        # track ID
        if 'thorax' in keypoints:
            thorax_pos = keypoints['thorax']
            text = ax.text(thorax_pos[0], thorax_pos[1], thorax_pos[2], f"ID:{track_id}", color=color,
                           fontsize='x-small')
            artists.append(text)

    return artists


def run_viewer(all_tracked_skeletons, bones, volume_bounds):

    try:
        plt.rcParams['keymap.forward'].remove('right')
        plt.rcParams['keymap.back'].remove('left')
    except ValueError:
        pass

    num_frames = len(all_tracked_skeletons)
    if num_frames == 0:
        print("No data to display.")
        return

    # One-time Setup of the Figure and Axes
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0, 0.1, 1, 0.9], projection='3d')
    slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(volume_bounds['x'])
    ax.set_ylim(volume_bounds['y'])
    ax.set_zlim(volume_bounds['z'])
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    z_range = abs(ax.get_zlim()[1] - ax.get_zlim()[0])
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.view_init(elev=20., azim=-70)

    # Slider and event handling
    frame_slider = Slider(ax=slider_ax, label='Frame', valmin=0, valmax=num_frames - 1, valinit=0, valstep=1)

    # Keep track of the artists (lines, points) from the last frame
    current_artists = []

    def update(val):
        nonlocal current_artists
        frame_idx = int(frame_slider.val)

        for artist in current_artists:
            artist.remove()
        current_artists.clear()

        frame_data = all_tracked_skeletons[frame_idx]
        current_artists = draw_skeletons(frame_data, bones, ax)

        ax.set_title(f"Frame {frame_data.get('frame_idx', '')}")

        fig.canvas.draw_idle()

    frame_slider.on_changed(update)

    def on_key(event):
        current_val = frame_slider.val
        if event.key == 'right':
            frame_slider.set_val(min(current_val + 1, num_frames - 1))
        elif event.key == 'left':
            frame_slider.set_val(max(current_val - 1, 0))
        elif event.key == 'pageup':
            frame_slider.set_val(min(current_val + 50, num_frames - 1))
        elif event.key == 'pagedown':
            frame_slider.set_val(max(current_val - 50, 0))

    fig.canvas.mpl_connect('key_press_event', on_key)
    update(0)
    plt.show()

##
folder = Path().home() / 'Desktop' / '3d_ant_data'
prefix = '240905-1616'

skeleton_input_path = folder / prefix / 'inputs' / 'tracking'
_, bones = fileio.load_skeleton_SLEAP(skeleton_input_path, indices=False)
volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

with open('final_tracked_skeletons.pkl', 'rb') as f:
    all_tracked_skeletons = pickle.load(f)

run_viewer(all_tracked_skeletons, bones, volume_bounds)