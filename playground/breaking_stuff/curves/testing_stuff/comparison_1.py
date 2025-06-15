import numpy as np


def fourier_features(triangles, k_range):
    ks = np.mgrid[0 : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T

    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    mtks = np.matvec(mt, ks[:, np.newaxis, :]).transpose(1, 0, 2)
    a = np.power(mtks[:, :, 1] * (mtks[:, :, 0] - mtks[:, :, 1]), -1)
    b = np.power(mtks[:, :, 0] * mtks[:, :, 1], -1)
    sin_k1, sin_k2 = np.sin(mtks[:, :, 0]), np.sin(mtks[:, :, 1])
    cos_k1, cos_k2 = np.cos(mtks[:, :, 0]), np.cos(mtks[:, :, 1])
    sTks = a * (sin_k2 - sin_k1) + b * sin_k1
    cTks = a * (cos_k2 - cos_k1) + b * (cos_k1 - 1)

    cos = np.cos(np.inner(ks, triangles[:, 2])).T
    sin = np.sin(np.inner(ks, triangles[:, 2])).T
    stk = det_m[:, np.newaxis] * (sTks * cos + cTks * sin)
    ctk = det_m[:, np.newaxis] * (cTks * cos - sTks * sin)

    triangles_features = np.stack((stk, ctk), axis=-1)
    shape_features = np.sum(triangles_features, axis=0)

    return shape_features


def loopy_fourier(triangles, k_range):
    ks = np.mgrid[0 : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T

    Sks = []
    Cks = []
    for triangle in triangles:
        M = np.array([triangle[0] - triangle[2], triangle[1] - triangle[2]])
        Mt = M.T
        detM = np.linalg.det(Mt)

        Mtk = np.matvec(Mt, ks)
        k1, k2 = Mtk.T
        coef_1 = [(k2[i] * (k1[i] - k2[i])) ** (-1) for i in range(len(k1))]
        coef_2 = [(k1[i] * k2[i]) ** (-1) for i in range(len(k1))]
        Stmtk = coef_1 * (np.sin(k2) - np.sin(k1)) + coef_2 * np.sin(k1)
        Ctmtk = coef_1 * (np.cos(k2) - np.cos(k1)) + coef_2 * (np.cos(k1) - 1)

        Sk_1 = detM * np.cos(np.dot(ks, triangle[2])) * Stmtk
        Sk_2 = detM * np.sin(np.dot(ks, triangle[2])) * Ctmtk
        Ck_1 = detM * np.cos(np.dot(ks, triangle[2])) * Ctmtk
        Ck_2 = detM * np.sin(np.dot(ks, triangle[2])) * Stmtk
        Sk = Sk_1 + Sk_2
        Ck = Ck_1 - Ck_2

        Sks.append(Sk)
        Cks.append(Ck)

    fourier_sin = np.sum(np.array(Sks), axis=0)
    fourier_cos = np.sum(np.array(Cks), axis=0)
    shape_features = np.stack((fourier_sin, fourier_cos), axis=-1)

    return shape_features


def main():
    k_range = 2
    dummy_triangles = np.array(
        [[[0, 0], [0, 1], [1, 1]], [[0, 0], [1, 1], [1, 0]], [[-1, 0], [0, 0], [0, 1]]]
    )

    # With this dummy shape let's compare
    loopy_feats = loopy_fourier(dummy_triangles, k_range)
    fourier_feats = fourier_features(dummy_triangles, k_range)

    # We're getting rid of the nans (just deleting them)
    loopy_feats = loopy_feats[~np.isnan(loopy_feats)].reshape((-1, 2))
    fourier_feats = fourier_feats[~np.isnan(fourier_feats)].reshape((-1, 2))

    if (loopy_feats == fourier_feats).all():
        print("It's the same result !")


if __name__ == "__main__":
    main()
