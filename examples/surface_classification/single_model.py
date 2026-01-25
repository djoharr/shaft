from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def main():
    fcs = np.load(join(dirname(realpath(__file__)), "../fourier_coefs.npy"))
    info = pd.read_csv(
        join(dirname(__file__), "../../../data/surfaces", "shrec_info.csv"),
        delimiter=";",
    )

    X = fcs[:, :, 1]  # only cos values
    y = info.labels

    rng = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=rng
    )

    accuracies = []
    max_comp = 60
    for n_comp in tqdm(range(2, max_comp)):
        nca = make_pipeline(
            StandardScaler(),
            NeighborhoodComponentsAnalysis(n_components=n_comp, random_state=rng),
        )
        nca.fit(X_train, y_train)
        model = SVC(kernel="linear", C=0.025, random_state=42)
        model.fit(nca.transform(X_train), y_train)
        acc = model.score(nca.transform(X_test), y_test)
        accuracies.append(acc)

    plt.plot(np.arange(2, max_comp), accuracies)
    plt.title(
        f"Test accuracies against number of components.\n Best is {np.max(accuracies)}"
    )
    plt.show()

    breakpoint()


if __name__ == "__main__":
    main()
