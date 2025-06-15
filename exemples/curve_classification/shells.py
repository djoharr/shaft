import os
from os.path import join, dirname, realpath
from tqdm import tqdm

import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt

from shaft import cft

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def main():
    shells = np.load(join(dirname(__file__), "../../data/curves/shells/curves.npy"))
    shells_info = pd.read_csv(join(dirname(__file__), "../../data/curves/shells/info.csv"))

    print('....................... COMPUTING CFTs ....................... ')
    k_range = 20
    all_coefs = np.zeros((len(shells), k_range**2, 2))
    for i in tqdm(range(len(shells))):
        curve = shells[i] / 1.5
        fcs = cft(curve, k_range)
        all_coefs[i]= fcs

    print('....................... FINDING BEST PARAMETERS ....................... ')

    # X = all_coefs[:, :, 1] # sin=0, cos = 1
    X = all_coefs.reshape((-1, k_range**2*2)) # flatten
    # X = np.sqrt(all_coefs[:,:,0]**2 + all_coefs[:,:,1]**2) # absolute
    y = shells_info.genusNumber

    rng = 42
    ###################################### SVM ######################################

    # f1 = []
    # all_comp = [2, 10, 20, 50, 100, 200, 400, 800]
    # for n_comp in tqdm(all_comp):
    #     clf = SVC(kernel="linear", C=0.025, random_state=rng)
    #     clf = KNeighborsClassifier(n_neighbors=3)
    #     model = make_pipeline(StandardScaler(), clf)
    #     score = cross_val_score(model, X[:, :n_comp], y, cv=5, scoring='f1_weighted').mean()
    #     f1.append(score)

    # plt.plot(all_comp, f1)
    # plt.title(f"Test accuracies against number of components.\n Best is {np.max(f1)}")
    # plt.show()

    ###################################### SVM + PCA ######################################

    f1 = []
    max_comp = 80
    for n_comp in tqdm(range(2, max_comp)):
        # clf = SVC(kernel="linear", C=0.025, random_state=rng)
        clf = KNeighborsClassifier(n_neighbors=3)
        model = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=rng), clf)
        score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean()
        f1.append(score)

    plt.plot(np.arange(2, max_comp), f1)
    plt.title(f"Test accuracies against number of components.\n Best is {np.max(f1)}")
    plt.show()




    breakpoint()


if __name__ == "__main__":
    main()
