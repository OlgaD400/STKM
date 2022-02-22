import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
from typing import List


def animate_path_positions(
    swarms_data: List[List[float]],
    anim_frames: int,
    anim_interval: int,
    colors: List[str],
) -> HTML:
    """
    Create an animation of the swarms.

    Each frame displays the locations of the points, with each swarm having a different color.

    Args:
        swarms_data (List[List[float]]): Each element of the list contains the coordinates of all
            particles in a swarm.
        anim_frames (int): Number of frames in the animation.
        anim_interval (int): Delay between frames in ms.
        colors (List[str]): List containing color codes i.e. ['b', 'r', 'g'].  There needs to be at least as many colors as swarms.

    Returns:
        HTML(anim.to_jshtml()) (TextDisplayObject): Animation of swarms
    """
    assert len(colors) >= len(swarms_data), (
        "Colors list must contain at least as many colors as "
        "swarms in the swarms data list."
    )

    assert anim_frames <= max([len(swarm) for swarm in swarms_data]), (
        "Number of frames cannot be larger than the "
        "number of time steps in the data."
    )

    fig = plt.figure()

    swarm_concatenate = []
    for swarm in swarms_data:
        swarm_concatenate.append(np.concatenate(swarm))

    fig_min = np.floor(np.min(np.concatenate(swarm_concatenate)))
    fig_max = np.ceil(np.max(np.concatenate(swarm_concatenate)))

    ax = plt.axes(xlim=(fig_min, fig_max), ylim=(fig_min, fig_max))
    scats = [ax.scatter([], [], c=colors[j]) for j in range(len(swarms_data))]

    def init():
        for scat in scats:
            scat.set_offsets([])
        return tuple(scats)

    # animate function. This is called sequentially
    def animate(i):
        for idx, coords in enumerate(swarms_data):
            scats[idx].set_offsets(coords[i])

        return tuple(scats)

    anim = animation.FuncAnimation(
        fig,
        animate,
        # init_func=init,
        frames=anim_frames,
        interval=anim_interval,
        blit=True,
    )

    return HTML(anim.to_jshtml())

def animate_path(swarms_data, assignment, anim_frames: int, anim_interval: int):
    fig = plt.figure()

    fig_min = np.floor(np.min(swarms_data))
    fig_max = np.ceil(np.max(swarms_data))

    ax = plt.axes(xlim=(fig_min, fig_max), ylim=(fig_min, fig_max))
    scats = ax.scatter(swarms_data[0, 0, :], swarms_data[0, 1, :], c=assignment[:, 0])

    def init():
        for scat in scats:
            scat.set_offsets([])
        return tuple(scats)

    # animate function. This is called sequentially
    def animate(i):
        scats.set_offsets(swarms_data[i, :, :].T)
        scats.set_array(assignment[:, i])

        return scats,

    anim = animation.FuncAnimation(
        fig,
        animate,
        #         init_func=init,
        frames=anim_frames,
        interval=anim_interval,
        blit=True,
    )

    return HTML(anim.to_jshtml())
