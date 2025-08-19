from turbulent_propagation import (
    turbulent_propagation,
    hill_andrews_spectrum,
)
from tqdm import trange
from jax import random
import h5py
import numpy as np

from in_out import load_parameters


def main(config_path="config.toml", output_path="output.h5", dataset_name="fields"):
    param = load_parameters(config_path)

    print("Rytov variance: {:.2e}".format(param.rytov_variance()))
    print("Rayleigh length: {:.2e}".format(param.rayleigh_length()))
    print("Final waist: {:.2e}".format(param.final_waist()))
    print("Lambda0: {:.2e}".format(param.Lambda0()))
    print("Lambda: {:.2e}".format(param.Lambda()))
    print(
        "Second turbulence condition: {:.2e}".format(
            param.rytov_variance() * param.Lambda() ** (5 / 6)
        )
    )

    keys = random.split(random.key(42), param.nbatches)

    f = h5py.File(output_path, "a")
    dset = f.create_dataset(
        dataset_name,
        (param.nbatches * param.batchsize, param.N, param.N),
        dtype=np.complex64,
        chunks=(param.batchsize, param.N, param.N),
    )

    for n in trange(param.nbatches):
        u = turbulent_propagation(
            param.u0,
            param.d,
            param.d,
            param.z,
            param.wavelength,
            param.magnification,
            hill_andrews_spectrum,
            key=keys[n],
            nsamples=param.batchsize,
            nsteps=param.nsteps,
            r0=param.r0,
            L0=param.L0,
            l0=param.l0,
        )
        dset[n * param.batchsize : (n + 1) * param.batchsize] = u

    f.close()


if __name__ == "__main__":
    main()
