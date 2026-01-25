import os
from os.path import dirname, join, realpath

import numpy as np
import pyvista as pv
from tqdm import tqdm

from shaft import sft


def main():
    shrec_surfaces = join(dirname(__file__), "../../data/surfaces/shrec")

    k_range = 20

    all_coefs = np.zeros((len(os.listdir(shrec_surfaces)), k_range**3, 2))
    for el in tqdm(os.listdir(shrec_surfaces)):
        surface_path = join(shrec_surfaces, el)
        surface = pv.read(surface_path)
        fcs = sft(surface, k_range)
        idx = eval(el.split(".")[0])
        all_coefs[idx] = fcs

    np.save(join(dirname(realpath(__file__)), "fourier_coefs.npy"), all_coefs)


if __name__ == "__main__":
    main()
