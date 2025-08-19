# Turbulent Dataset

This repository contains the code for the generation of a dataset of laser beams propagating through a turbulent medium. 

The dataset is generated using the main.py script, and is saved in HDF5 format. The parameters of the simulation (grid size, wavelength, etc.) are read from the `config.toml` file. An example of how to read and visualize the data is provided in the read.py script.

It is recommended to use [uv](https://docs.astral.sh/uv/#installation) in order to manage the dependencies. Once installed, you can run `uv sync` to install the required packages, and `uv run main.py` to execute the main script.