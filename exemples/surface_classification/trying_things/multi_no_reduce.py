from os.path import join, dirname, realpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC


def main():
    fcs = np.load(join(dirname(realpath(__file__)), '../fourier_coefs.npy'))
    info = pd.read_csv(join(dirname(__file__), "../../../data/surfaces", "shrec_info.csv"), delimiter=';')

    # X = fcs[:, :, 1] # sin=0, cos = 1
    # X = fcs.reshape((-1, 16000)) # flatten
    X = np.sqrt(fcs[:,:,0]**2 + fcs[:,:,1]**2) # norm
    # X = np.arctan2(fcs[:, :, 1], fcs[:,:,0]) # angle

    y = info.labels
    rng = 42
    X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=rng)


    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM"]

    classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025, random_state=42), SVC(C=1, random_state=42)]
    accuracies = {k: [] for k in names}
    all_comp = [2, 8, 10, 50, 100, 200, 500, 1000, 2000, 4000]
    for n_comp in tqdm(all_comp):
        X_train, X_test = X_tr[:, :n_comp], X_te[:, :n_comp]
        for name, clf in zip(names, classifiers):
            model = make_pipeline(StandardScaler(), clf)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            accuracies[name].append(score)

    for name in accuracies.keys():
        plt.plot(all_comp, accuracies[name], label=name)
    plt.legend(loc='lower right')
    plt.title(f"Test accuracies against number of components for all methods.")
    plt.show()

    breakpoint()

if __name__ == "__main__":
    main()
