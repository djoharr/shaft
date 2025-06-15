import numpy as np

def get_masks(coords):
    nx, ny, nz = coords[:, :, 0] == 0, coords[:, :, 1] == 0, coords[:, :, 2] == 0
    exy, exz, eyz = coords[:,:,0]==coords[:,:,1], coords[:,:,0]==coords[:,:,2], coords[:,:,1]==coords[:,:,2]
    nxyz = nx & ny & nz
    nxy = nx & ny & ~nxyz
    nxz = nx & nz & ~nxyz
    nyz = ny & nz & ~nxyz
    exynz = exy & nz & ~nxyz
    exy = exy & ~nx & ~ny & ~nz & ~exynz
    exzny = exz & ny & ~nxyz
    exz = exz & ~nx & ~ny & ~nz & ~exzny
    eyznx = eyz & nx & ~nxyz
    eyz = eyz & ~nx & ~ny & ~nz & ~eyznx
    exyz = exy & eyz
    exy = exy & ~exyz
    exz = exz & ~exyz
    eyz = eyz & ~exyz
    nx = nx & ~nxy & ~nxz & ~eyznx & ~nxyz
    ny = ny & ~nxy & ~nyz & ~exzny & ~nxyz
    nz = nz & ~nxz & ~nyz & ~exynz & ~nxyz
    rest = ~nx & ~ny & ~nz & ~nxy & ~nxz & ~nyz & ~nxyz
    rest = rest & ~exy & ~exz & ~eyz & ~exyz & ~exynz & ~exzny & ~eyznx

    return rest, nx, ny, nz, nxy, nxz, nyz, nxyz, exy, exz, eyz, exyz, exynz, exzny, eyznx


def get_sc(ks, mt):
    mat_base = np.einsum('ijk,lj->lik', mt, ks)
    sins = np.zeros(mat_base.shape[:2])
    coss = np.zeros(mat_base.shape[:2])

    normies, nx, ny, nz, nxy, nxz, nyz, nxyz, exy, exz, eyz, exyz, exynz, exzny, eyznx = get_masks(mat_base)

    mnorm = mat_base[normies]
    x, y, z = mnorm[:, 0], mnorm[:, 1], mnorm[:, 2]
    sinx, siny, sinz = np.sin(x), np.sin(y), np.sin(z)
    cosx, cosy, cosz = np.cos(x), np.cos(y), np.cos(z)
    a, b = np.power(z * (y-z) * (x-y), -1), np.power(z * (y-z) * (x-z), -1)
    c, d = np.power(y * z * (x-y), -1), np.power(x * y * z, -1)
    sins[normies] = a * (cosx - cosy) - b * (cosx - cosz) - c * (cosx - cosy) + d * (cosx - 1)
    coss[normies] = - a * (sinx - siny) + b * (sinx - sinz) + c * (sinx - siny) - d * sinx

    mnx = mat_base[nx]
    y, z = mnx[:, 1], mnx[:, 2]
    siny, sinz, cosy, cosz = np.sin(y), np.sin(z), np.cos(y), np.cos(z)
    a, b = np.power(y*z*(y-z), -1), np.power(z**2*(y-z), -1)
    c, d = np.power(z*y**2, -1), np.power(y*z, -1)
    sins[nx] = (c - a) * (1 - cosy) + b * (1 - cosz)
    coss[nx] = (c - a) * siny + b * sinz - d

    mny = mat_base[ny]
    x, z = mny[:, 0], mny[:, 2]
    sinx, sinz, cosx, cosz = np.sin(x), np.sin(z), np.cos(x), np.cos(z)
    a, b = np.power(x*z**2, -1), np.power(z**2*(x-z), -1)
    c, d = np.power(z*x**2, -1), np.power(x*z, -1)
    sins[ny] = (a + c)*(1 - cosx) + b*(cosx - cosz)
    coss[ny] = (a + c) * sinx - b * (sinx - sinz) - d

    mnz = mat_base[nz]
    x, y = mnz[:, 0], mnz[:, 1]
    sinx, siny, cosx, cosy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    a, b = np.power(y*x**2, -1), np.power(y**2*(x-y), -1)
    c, d = np.power(x*y**2, -1), np.power(x*y, -1)
    sins[nz] = (a + c) * (1 - cosx) + b * (cosx - cosy)
    coss[nz] = (a + c) * sinx - b * (sinx - siny) - d

    mnxy = mat_base[nxy]
    z = mnxy[:, 2]
    sinz, cosz = np.sin(z), np.cos(z)
    a, b, c = np.power(z, -3), np.power(2*z, -1), np.power(z, -2)
    sins[nxy] = a*(cosz - 1) + b
    coss[nxy] = c - a*sinz

    mnxz = mat_base[nxz]
    y = mnxz[:, 1]
    siny, cosy = np.sin(y), np.cos(y)
    a, b, c = np.power(2*y, -1), np.power(y, -3), np.power(y, -2)
    sins[nxz] = a - b * (1 - cosy)
    coss[nxz] = c - b * siny

    mnyz = mat_base[nyz]
    x = mnyz[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b, c = np.power(2*x, -1), np.power(x, -3), np.power(x, -2)
    sins[nyz] = a + b * (cosx - 1)
    coss[nyz] = c - b * sinx

    mnxyz = mat_base[nxyz]
    sins[nxyz] = 0
    coss[nxyz] = 0

    mexy = mat_base[exy]
    x, z = mexy[:, 0], mexy[:, 2]
    sinx, sinz, cosx, cosz = np.sin(x), np.sin(z), np.cos(x), np.cos(z)
    a, b = np.power(z*(x-z), -1), np.power(z*(x-z)**2, -1)
    c, d = np.power(x*z, -1), np.power(z*x**2, -1)
    sins[exy] = -a*sinx - b*(cosx - cosz) + c*sinx + d*(cosx - 1)
    coss[exy] = -a*cosx + b*(sinx - sinz) + c*cosx - d*sinx

    mexz = mat_base[exz]
    x, y = mexz[:, 0], mexz[:, 1]
    sinx, siny, cosx, cosy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    a, b = np.power(x*(y-x), -1), np.power(x*(y-x)**2, -1)
    c, d, e = np.power(y*x*(x-y), -1), np.power(y*x**2, -1), np.power(x*(x-y)**2, -1)
    sins[exz] = a*sinx - (b + c)*(cosx - cosy) + d*(cosx - 1)
    coss[exz] = e*(sinx - siny) - a*cosx + c*(sinx - siny) - d*sinx

    meyz = mat_base[eyz]
    x, y = meyz[:, 0], meyz[:, 1]
    sinx, siny, cosx, cosy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    a, b = np.power(y*(x-y), -1), np.power(y*(x-y)**2, -1)
    c, d = np.power((x-y)*y**2, -1), np.power(x*y**2, -1)
    sins[eyz] = a*siny + b*(cosx - cosy) - c*(cosx - cosy) + d*(cosx-1)
    coss[eyz] = -a*cosy - b*(sinx - siny) + c*(sinx - siny) - d*sinx

    mexyz = mat_base[exyz]
    x = mexyz[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b, c = np.power(2*x, -1), np.power(x, -2), np.power(x, -3)
    sins[exyz] = -a*cosx + b*sinx + c*(cosx - 1)
    coss[exyz] = a*sinx + b*cosx - c*sinx

    mexynz = mat_base[exynz]
    x = mexynz[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b = np.power(x, -2), np.power(x, -3)
    sins[exynz] = -a * sinx - b * (cosx - 1)
    coss[exynz] = a * (cosx - 1) + 2 * b * sinx - 2 * a * cosx

    mexzny = mat_base[exzny]
    x = mexzny[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b = np.power(x, -3), np.power(x, -2)
    sins[exzny] = -2 * a * (cosx - 1) - b * sinx
    coss[exzny] = 2 * a * sinx - b * (cosx - 1)

    meyznx = mat_base[eyznx]
    y = meyznx[:, 1]
    siny, cosy = np.sin(y), np.cos(y)
    a, b = np.power(y, -2), np.power(y, -3)
    sins[eyznx] = -a * siny + 2 * b * (1 - cosy)
    coss[eyznx] = -a * (1 + cosy) + 2 * b * siny

    return sins.T, coss.T


def get_sc_no_taylor(ks, mt):
    mat_base = np.einsum('ijk,lj->lik', mt, ks).transpose(1, 0, 2)
    x, y, z = mat_base[:, :, 0], mat_base[:, :, 1], mat_base[:, :, 2]
    a = np.power(z * (y-z) * (x-y), -1)
    b = np.power(z * (y-z) * (x-z), -1)
    c = np.power(y * z * (x-y), -1)
    d = np.power(x * y * z, -1)
    sinx, siny, sinz = np.sin(x), np.sin(y), np.sin(z)
    cosx, cosy, cosz = np.cos(x), np.cos(y), np.cos(z)
    sins = - a * (sinx - siny) + b * (sinx - sinz) + c * (sinx - siny) - d * sinx
    coss = a * (cosx - cosy) - b * (cosx - cosz) - c * (cosx - cosy) + d * (cosx - 1)
    return sins, coss
