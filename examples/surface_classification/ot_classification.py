from os.path import dirname, join, realpath

import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def main():
    fcs = np.load(join(dirname(realpath(__file__)), "fourier_coefs.npy"))
    info = pd.read_csv(
        join(dirname(__file__), "../../data/surfaces", "shrec_info.csv"),
        delimiter=";",
    )

    X = torch.from_numpy(fcs[:, :, 0]).contiguous()  # only sin values
    X = X.reshape(-1, 20, 20, 20)[:, :10, :10, :10]
    X = X.reshape(600, -1)
    x = X / X.sum(dim=1, keepdim=True)
    y = np.array(info.labels)

    lattice = np.mgrid[0:9.01:1, 0:9.01:1, 0:9.01:1].reshape(3, -1).T
    pos = torch.from_numpy(lattice).contiguous()

    n = len(x)
    dmat = torch.zeros((n, n))
    for i in tqdm(range(n)):
        for j in tqdm(range(i + 1, n), leave=False):
            Loss = SamplesLoss("sinkhorn", p=2, blur=0.01, backend="multiscale")
            wass = Loss(x[i], pos, x[j], pos)
            dmat[i, j] = wass.item()
    precomputed_dist = dmat + dmat.T

    all_ns = np.arange(n)
    _, _, y_train, y_test, train_ix, test_ix = train_test_split(x.numpy(), y, all_ns)
    D = precomputed_dist[:, train_ix]
    D_train, D_test = D[train_ix], D[test_ix]
    knn = KNeighborsClassifier(metric="precomputed", n_neighbors=3)
    knn.fit(D_train, y_train)

    test_acc = knn.score(D_test, y_test)
    print(f"Quick knn run gives accuracy of: {test_acc} ")

    breakpoint()


if __name__ == "__main__":
    main()
