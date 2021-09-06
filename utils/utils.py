import numpy as np


def rad2deg(rad):
    return rad * (180.0 / np.pi)


def deg2rad(deg):
    return deg * (np.pi / 180.0)


def quat_to_mat(q):     # single element
    sqw = q[3] * q[3]
    sqx = q[0] * q[0]
    sqy = q[1] * q[1]
    sqz = q[2] * q[2]

    # invs(inverse square length) is only required if quaternion is not already normalised
    m = np.zeros([3, 3])
    invs = 1 / (sqx + sqy + sqz + sqw)
    m[0, 0] = (sqx - sqy - sqz + sqw) * invs    # since sqw + sqx + sqy + sqz = 1 / invs * invs
    m[1, 1] = (-sqx + sqy - sqz + sqw) * invs
    m[2, 2] = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = q[0] * q[1]
    tmp2 = q[2] * q[3]
    m[1, 0] = 2.0 * (tmp1 + tmp2) * invs
    m[0, 1] = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = q[0] * q[2]
    tmp2 = q[1] * q[3]
    m[2, 0] = 2.0 * (tmp1 - tmp2) * invs
    m[0, 2] = 2.0 * (tmp1 + tmp2) * invs
    tmp1 = q[1] * q[2]
    tmp2 = q[0] * q[3]
    m[2, 1] = 2.0 * (tmp1 + tmp2) * invs
    m[1, 2] = 2.0 * (tmp1 - tmp2) * invs
    return np.array(m)
