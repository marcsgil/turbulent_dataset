import tomllib
import jax.numpy as jnp
from turbulent_propagation import (
    atmospheric_coherence_length,
    rytov_variance,
    turbulent_propagation,
    hill_andrews_spectrum,
)
from tqdm import trange
from jax import random
import h5py
import numpy as np


def gaussian_beam(x, y, z, w0, wavelength):
    zr = jnp.pi * w0**2 / wavelength
    w = w0 * jnp.sqrt(1 + (z / zr) ** 2)
    if z == 0:
        R = jnp.inf
    else:
        R = z * (1 + (zr / z) ** 2)
    psi = jnp.arctan(z / zr)
    r2 = x**2 + y**2
    k = 2 * jnp.pi / wavelength
    return (w0 / w) * jnp.exp(-r2 / w**2 - 1j * (k * r2 / 2 / R - psi))


def init(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    N = config["grid"]["size"]  # Number of points in each dimension
    L = config["grid"]["length"]
    d = L / N
    wavelength = config["beam"]["wavelength"]  # Wavelength of the wave
    w0 = config["beam"]["waist"]  # Beam waist

    xs = jnp.arange(-L / 2, L / 2, d)
    ys = jnp.arange(-L / 2, L / 2, d)
    xs, ys = jnp.meshgrid(xs, ys)
    u0 = gaussian_beam(xs, ys, 0, w0, wavelength)
    batchsize = config["simulation"]["batchsize"]
    nsteps = config["simulation"]["nsteps"]
    nbatches = config["simulation"]["nbatches"]

    z = config["simulation"]["propagation_distance"]

    Cn2 = config["turbulence"]["Cn2"]  # Structure constant of the refractive index
    r0 = atmospheric_coherence_length(z / nsteps, Cn2, wavelength)
    L0 = config["turbulence"]["L0"]  # Outer scale of turbulence
    l0 = config["turbulence"]["l0"]  # Inner scale of turbulence

    s2 = rytov_variance(z, Cn2, wavelength)
    rayleigh_length = jnp.pi * w0**2 / wavelength
    Lambda0 = z / rayleigh_length
    w = w0 * jnp.sqrt(1 + Lambda0**2)
    Lambda = wavelength * z / jnp.pi / w**2

    print("Rytov variance: {:.2e}".format(s2))
    print("Rayleigh length: {:.2e}".format(rayleigh_length))
    print("Final waist: {:.2e}".format(w))
    print("Lambda0: {:.2e}".format(Lambda0))
    print("Lambda: {:.2e}".format(Lambda))
    print("Second turbulence condition: {:.2e}".format(s2 * Lambda ** (5 / 6)))

    return N, u0, wavelength, d, z, batchsize, nsteps, nbatches, r0, L0, l0


def main(config_path="config.toml", output_path="output.h5"):
    N, u0, wavelength, d, z, batchsize, nsteps, nbatches, r0, L0, l0 = init(config_path)

    keys = random.split(random.key(42), nbatches)

    f = h5py.File(output_path, "a")
    dset = f.create_dataset("fields", (nbatches * batchsize, N, N), dtype=np.complex64)

    for n in trange(nbatches):
        u = turbulent_propagation(
            u0,
            d,
            d,
            z,
            wavelength,
            1,
            hill_andrews_spectrum,
            key=keys[n],
            nsamples=batchsize,
            nsteps=nsteps,
            r0=r0,
            L0=L0,
            l0=l0,
        )
        dset[n * batchsize : (n + 1) * batchsize] = np.array(u, dtype=np.complex64)

    f.close()


if __name__ == "__main__":
    main()
