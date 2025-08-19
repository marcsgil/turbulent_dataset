import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import jax.numpy as jnp
from in_out import read_first_batch

matplotlib.use("QtAgg")

fields = read_first_batch("output.h5", 128)

intensities = jnp.abs(fields) ** 2

fig, ax = plt.subplots()
im = ax.imshow(intensities[0], cmap="hot", animated=True)


def update(frame):
    im.set_array(intensities[frame])
    return [im]


ani = animation.FuncAnimation(
    fig, update, frames=intensities.shape[0], interval=100, blit=True
)

plt.show()
