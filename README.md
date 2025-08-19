# Turbulent Dataset

This repository contains the code for the generation of a dataset of laser beams propagating through a turbulent medium. 

The dataset is generated using the `main.py` script, and is saved in HDF5 format. The parameters of the simulation (grid size, wavelength, etc.) are read from the `config.toml` file. 

There are two example files: `visualization.py` reads a single batch of data and generates an animation showing different realizations of the turbulent propagation. `scintillation.py` reads the data in batches and computes and visualizes the scintillation index of the beam.

It is recommended to use [uv](https://docs.astral.sh/uv/#installation) in order to manage the dependencies. Once installed, you can run `uv sync` to install the required packages, and `uv run main.py` to execute the main script.