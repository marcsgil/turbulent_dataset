from in_out import read_batches, load_parameters
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.use("QtAgg")

params = load_parameters("config.toml")
data = read_batches("output.h5", params.batchsize)

N = params.N
xs = params.xs

mean_intensity = jnp.zeros((N, N))
square_intensity = jnp.zeros_like(mean_intensity)

for batch in tqdm(data):
    intensities = jnp.abs(batch) ** 2
    mean_intensity = mean_intensity + jnp.mean(intensities, axis=0) / params.nbatches
    square_intensity = (
        square_intensity + jnp.mean(intensities**2, axis=0) / params.nbatches
    )

scintillation = square_intensity / mean_intensity**2 - 1
ymin = 0
ymax = 3 * params.rytov_variance()

line_scintillation = (
    scintillation[N // 2, N // 2 :]
    + scintillation[N // 2 :, N // 2]
    + jnp.flip(scintillation[N // 2, : N // 2] + scintillation[: N // 2, N // 2])
) / 4

plt.plot(xs[N // 2, N // 2 :] / params.final_waist(), line_scintillation)
plt.hlines(
    params.rytov_variance(),
    color="r",
    linestyle="--",
    label="Plane Wave Rytov Variance",
    xmin=xs[N // 2, N // 2] / params.final_waist(),
    xmax=xs[N // 2, -1] / params.final_waist(),
)

# plt.ylim(ymin, ymax)

plt.xlabel("Distance from the beam's center (waists)")
plt.ylabel("Scintillation Index")

plt.legend(loc="upper left")

""" plt.annotate(
    "l0 = 3e-10 m",  # The text to display
    xy=(0.25, 0.6),  # The point to annotate
) """

plt.show()
