import h5py
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

with h5py.File("output.h5", "r") as f:
    fields = np.array(f["fields"][:100])  # Load first 100 fields

intensities = np.abs(fields) ** 2

fig, ax = plt.subplots()
im = ax.imshow(intensities[0], cmap="hot", animated=True)


def update(frame):
    im.set_array(intensities[frame])
    return [im]


ani = animation.FuncAnimation(
    fig, update, frames=intensities.shape[0], interval=100, blit=True
)

plt.show()
