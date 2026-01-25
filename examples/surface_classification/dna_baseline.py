import os
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import robust_laplacian
import scipy
import scipy.sparse.linalg as sla
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def main():
    shrec_surfaces = join(dirname(__file__), "../../data/surfaces/shrec")
    shrec_info = join(dirname(__file__), "../../data/surfaces", "shrec_info.csv")

    print("....................... COMPUTING DNAs ....................... ")
    k = 245
    dna = np.zeros((len(os.listdir(shrec_surfaces)), k))
    for el in tqdm(os.listdir(shrec_surfaces)):
        surface_path = join(shrec_surfaces, el)
        surface = pv.read(surface_path)
        verts = np.array(surface.points)
        faces = surface.regular_faces
        L, M = robust_laplacian.mesh_laplacian(verts, faces)
        massvec = M.diagonal()
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * 1e-8).tocsc()
        Mmat = scipy.sparse.diags(massvec)
        evals, _ = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=1e-8)
        evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
        idx = eval(el.split(".")[0])
        dna[idx] = evals

    print("....................... FINDING BEST PARAMETERS ....................... ")

    info = pd.read_csv(shrec_info, delimiter=";")
    X = dna
    y = info.labels

    rng = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=rng
    )

    accuracies = []
    neighbors = []
    max_neigh = 10
    max_comp = 60
    for n_comp in tqdm(range(2, max_comp)):
        best_n = 0
        best_acc = 0
        for n_neighbors in range(2, max_neigh):
            nca = make_pipeline(
                StandardScaler(),
                NeighborhoodComponentsAnalysis(n_components=n_comp, random_state=rng),
            )
            nca.fit(X_train, y_train)

            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(nca.transform(X_train), y_train)
            acc_knn = knn.score(nca.transform(X_test), y_test)
            if acc_knn > best_acc:
                best_acc = acc_knn
                best_n = n_neighbors
        accuracies.append(best_acc)
        neighbors.append(best_n)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.arange(2, max_comp), accuracies)
    ax1.set_title(
        f"Test accuracies against n_components. \n Best is {np.max(accuracies)}"
    )
    ax2.plot(np.arange(2, max_comp), neighbors)
    ax2.set_title(f"Best number of neighbors for each n_components")
    plt.show()


if __name__ == "__main__":
    main()
