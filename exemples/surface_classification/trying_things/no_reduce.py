from os.path import join, dirname, realpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC


def main():
    fcs = np.load(join(dirname(realpath(__file__)), '../fourier_coefs.npy'))
    info = pd.read_csv(join(dirname(__file__), "../../../data/surfaces", "shrec_info.csv"), delimiter=';')

    X = fcs[:, :, 1] # sin=0, cos = 1
    # X = fcs.reshape((-1, 2000)) # flatten
    # X = np.sqrt(fcs[:,:,0]**2 + fcs[:,:,1]**2) # norm
    # X = np.arctan2(fcs[:, :, 1], fcs[:,:,0]) # angle
    y = info.labels

    rng = 42
    X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=rng)
    accuracies = []
    all_comp = [2, 10, 50, 100, 200, 500, 1000, 2000]
    for n_comp in tqdm(all_comp):
        X_train, X_test = X_tr[:, :n_comp], X_te[:, :n_comp]
        model = KNeighborsClassifier(n_neighbors=3)
        # model = SVC(kernel="linear", C=0.025, random_state=rng)
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        accuracies.append(acc)

    plt.plot(all_comp, accuracies)
    plt.title(f"Test accuracies against number of components.\n Best is {np.max(accuracies)}")
    plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(np.arange(2, max_comp), accuracies)
    # ax1.set_title(f"Test accuracies against number of components.\n Best is {np.max(accuracies)}")
    # ax2.plot(np.arange(2, max_comp), neighbors)
    # ax2.set_title(f"Best number of neighbors for each n_components")
    # plt.show()
    # X_embedded = model.transform(X)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
    # plt.title(f"KNN (k={n_neighbors}) \n Test accuracy = {acc_knn}")
    # plt.show()
    breakpoint()

if __name__ == "__main__":
    main()
