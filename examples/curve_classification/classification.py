from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from shaft import cft


def main():
    dataset = "leaves"  # shells, vases

    # Fourier PARAMETERS:
    k_range = 20  # Number of FC coefficients
    only_cos = False  # How to use them
    absolute_value = False
    with_PCA = True  # Classifier options

    # Model PARAMETERS
    clf = "svc"  # or svc
    max_comp = 80  # Number of components

    data = np.load(join(dirname(__file__), f"../../data/curves/{dataset}/curves.npy"))
    data_info = pd.read_csv(
        join(dirname(__file__), f"../../data/curves/{dataset}/info.csv")
    )

    if dataset == "shells":
        y = data_info.genusNumber
    else:
        y = data_info.speciesNumber

    print("....................... COMPUTING CFTs ....................... ")
    k_range = 20
    all_coefs = np.zeros((len(data), k_range**2, 2))
    for i in tqdm(range(len(data))):
        curve = data[i] / 1.5
        fcs = cft(curve, k_range)
        all_coefs[i] = fcs

    if only_cos:
        X = all_coefs[:, :, 1]
    elif absolute_value:
        X = np.sqrt(all_coefs[:, :, 0] ** 2 + all_coefs[:, :, 1] ** 2)
    else:
        X = all_coefs.reshape((-1, k_range**2 * 2))  # flatten

    print("....................... FINDING BEST PARAMETERS ....................... ")

    rng = 42
    models = {
        "svc": SVC(kernel="linear", C=0.025, random_state=rng),
        "knc": KNeighborsClassifier(n_neighbors=3),
    }
    f1 = []
    for n_comp in tqdm(range(2, max_comp)):
        clf = models[clf]
        if with_PCA:
            model = make_pipeline(
                StandardScaler(), PCA(n_components=n_comp, random_state=rng), clf
            )
        else:
            model = make_pipeline(StandardScaler(), clf)
        score = cross_val_score(model, X, y, cv=5, scoring="f1_weighted").mean()
        f1.append(score)

    plt.plot(np.arange(2, max_comp), f1)
    plt.title(f"Test accuracies against number of components.\n Best is {np.max(f1)}")
    plt.show()


if __name__ == "__main__":
    main()
