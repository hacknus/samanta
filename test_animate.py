import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

n=100
fig = plt.figure()
ax = plt.axes(xlim = (0,1), ylim = (0,1))
line, = ax.plot([], [], animated = True, lw=1)
points = np.random.rand(n,2)
ax.scatter(points[:,0], points[:,1], color = 'orange')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    """"""
    config = np.arange(n)
    np.random.shuffle(config)
    """"""

    xdata = points[config, 0]
    ydata = points[config, 1]
    line.set_data(xdata, ydata)
    return line,

#ax.scatter(xdata, ydata, s=200, color='black', zorder=1)
#ax.scatter(xdata, ydata, s=200, color='orange', zorder=1)
#ax.scatter(xdata, ydata, s=2, color='red', zorder=1)



ani = animation.FuncAnimation(fig, animate, np.arange(0,20), blit=True, interval=20,
                              repeat=False, init_func=init)
plt.show()