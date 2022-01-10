import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML

def animate_path(swarms_data, anim_frames, anim_interval, colors):

    ## Animate Paths ##
    fig = plt.figure()
    
    fig_min = np.floor(np.min(swarms_data))
    fig_max = np.ceil(np.max(swarms_data))
        
    ax = plt.axes(xlim=(fig_min,fig_max), ylim=(fig_min, fig_max))

#     colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    scats = [ax.scatter([], [], c = colors[j]) for j in range(len(swarms_data))]
    
    def init():
        for scat in scats:
            scat.set_offsets([])
        return tuple(scats)
    
    # animate function. This is called sequentially
    def animate(i):             
        for k in range(len(swarms_data)):
            scats[k].set_offsets(swarms_data[k][i])
            
        return tuple(scats)

    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = anim_frames, 
                                   interval = anim_interval, blit= True)
    
    return HTML(anim.to_jshtml())
    