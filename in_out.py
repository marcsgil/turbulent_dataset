import tomllib
import jax.numpy as jnp
from dataclasses import dataclass
from jax import Array
from turbulent_propagation import atmospheric_coherence_length, rytov_variance
from pathlib import Path
import h5py


def gaussian_beam(x, y, z, w0, wavelength):
    zr = jnp.pi * w0**2 / wavelength
    w = w0 * jnp.sqrt(1 + (z / zr) ** 2)
    if z == 0:
        R = jnp.inf
    else:
        R = z + zr**2 / z
    psi = jnp.arctan(z / zr)
    r2 = x**2 + y**2
    k = 2 * jnp.pi / wavelength
    return (w0 / w) * jnp.exp(-r2 / w**2 - 1j * (k * r2 / 2 / R - psi))


@dataclass
class SimulationParameters:
    N: int
    L: float
    d: float
    wavelength: float
    w0: float
    xs: Array
    ys: Array
    u0: Array
    batchsize: int
    nsteps: int
    nbatches: int
    z: float
    magnification: float
    Cn2: float
    L0: float
    l0: float

    def rayleigh_length(self):
        return jnp.pi * self.w0**2 / self.wavelength

    def Lambda0(self):
        return self.z / self.rayleigh_length()

    def final_waist(self):
        return self.w0 * jnp.sqrt(1 + (self.z / self.rayleigh_length()) ** 2)

    def Lambda(self):
        return self.wavelength * self.z / jnp.pi / self.final_waist() ** 2

    def rytov_variance(self):
        return rytov_variance(self.z, self.Cn2, self.wavelength)


def load_parameters(config_path: str | Path) -> SimulationParameters:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    N = config["grid"]["size"]  # Number of points in each dimension
    L = config["grid"]["length"]
    d = L / N
    wavelength = config["beam"]["wavelength"]  # Wavelength of the wave
    w0 = config["beam"]["waist"]  # Beam waist

    xs = jnp.arange(-L / 2, L / 2, d)
    ys = jnp.arange(-L / 2, L / 2, d)
    xs, ys = jnp.meshgrid(xs, ys, sparse=True)
    u0 = gaussian_beam(xs, ys, 0, w0, wavelength)
    batchsize = config["simulation"]["batchsize"]
    nsteps = config["simulation"]["nsteps"]
    nbatches = config["simulation"]["nbatches"]

    z = config["simulation"]["propagation_distance"]
    magnification = config["simulation"]["magnification"]  # Magnification factor

    Cn2 = config["turbulence"]["Cn2"]  # Structure constant of the refractive index
    L0 = config["turbulence"]["L0"]  # Outer scale of turbulence
    l0 = config["turbulence"]["l0"]  # Inner scale of turbulence

    return SimulationParameters(
        N=N,
        L=L,
        d=d,
        wavelength=wavelength,
        w0=w0,
        xs=xs,
        ys=ys,
        u0=u0,
        batchsize=batchsize,
        nsteps=nsteps,
        nbatches=nbatches,
        z=z,
        magnification=magnification,
        Cn2=Cn2,
        L0=L0,
        l0=l0,
    )


def read_all_data(filepath, dataset_name="fields"):
    with h5py.File(filepath, "r") as f:
        dataset = f[dataset_name]
        return jnp.array(dataset[:])


def read_batches(filepath, batch_size, dataset_name="fields"):
    with h5py.File(filepath, "r") as f:
        dataset = f[dataset_name]
        total_samples = dataset.shape[0]

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch = dataset[start_idx:end_idx]  # Only loads this slice
            yield jnp.array(batch)


def read_first_batch(filepath, batchsize, dataset_name="fields"):
    with h5py.File(filepath, "r") as f:
        dataset = f[dataset_name]
        return jnp.array(dataset[:batchsize])  # Load only the first batch
