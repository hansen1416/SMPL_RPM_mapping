import math
import numpy as np


def apply_quaternion_to_vector(q, v):
    # Convert the quaternion to a rotation matrix
    r = np.array([[1-2*q[2]**2-2*q[3]**2, 2*q[1]*q[2]-2*q[3]*q[0], 2*q[1]*q[3]+2*q[2]*q[0]],
                  [2*q[1]*q[2]+2*q[3]*q[0], 1-2*q[1]**2 -
                      2*q[3]**2, 2*q[2]*q[3]-2*q[1]*q[0]],
                  [2*q[1]*q[3]-2*q[2]*q[0], 2*q[2]*q[3]+2*q[1]*q[0], 1-2*q[1]**2-2*q[2]**2]])
    # Apply the rotation matrix to the vector
    return np.dot(r, v)


def axis_angle_to_quaternion(axis_angle):

    # print(axis_angle)

    axis = axis_angle / np.linalg.norm(axis_angle)
    angle = np.linalg.norm(axis_angle)
    half_angle = angle / 2

    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)

    return np.array([x, y, z, w])


def multiply_quaternions(q1, q2):
    """
    This function takes two quaternions as input (each represented as a tuple of 4 numbers) 
    and returns the product of the quaternions as a new tuple. 
    Note that the order of the input quaternions matters, 
    first apply rotation a then rotation b, the combined c = b * a
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])


def clamp(n, minn, maxn):
    """
    Constrains a number `n` between a minimum value `minn` and a maximum value `maxn`.
    Returns `n` if it is within the range, otherwise returns the nearest boundary value.
    """
    return max(min(maxn, n), minn)


def combine_quaternions(a, b):
    """Combine two quaternions to represent their relative rotation."""
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    c = multiply_quaternions(a, b)
    return c / np.linalg.norm(c)


def quaternion2matrix4(x, y, z, w):

    te = [

        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1

    ]

    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    te[0] = (1 - (yy + zz))
    te[1] = (xy + wz)
    te[2] = (xz - wy)
    te[3] = 0

    te[4] = (xy - wz)
    te[5] = (1 - (xx + zz))
    te[6] = (yz + wx)
    te[7] = 0

    te[8] = (xz + wy)
    te[9] = (yz - wx)
    te[10] = (1 - (xx + yy))
    te[11] = 0

    te[12] = 0
    te[13] = 0
    te[14] = 0
    te[15] = 1

    return te


def euler2quaternion(x, y, z, order='XYZ'):
    """
    order is xyz 
    """

    # // http://www.mathworks.com/matlabcentral/fileexchange/
    # // 	20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors/
    # //	content/SpinCalc.m

    c1 = math.cos(x / 2)
    c2 = math.cos(y / 2)
    c3 = math.cos(z / 2)

    s1 = math.sin(x / 2)
    s2 = math.sin(y / 2)
    s3 = math.sin(z / 2)

    if order == 'XYZ':
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3

    elif order == 'YXZ':
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3

    elif order == 'ZXY':
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3

    elif order == 'ZYX':
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3

    elif order == 'YZX':
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3

    elif order == 'XZY':
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3

    else:
        print('`euler2quaternion` encountered an unknown order: ' + order)

    return [_x, _y, _z, _w]


def matrix42euler(te, order='XYZ'):
    """

    """

    # // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
    m11 = te[0]
    m12 = te[4]
    m13 = te[8]
    m21 = te[1]
    m22 = te[5]
    m23 = te[9]
    m31 = te[2]
    m32 = te[6]
    m33 = te[10]

    if order == 'XYZ':

        _y = math.asin(clamp(m13, - 1, 1))

        if (abs(m13) < 0.9999999):

            _x = math.atan2(- m23, m33)
            _z = math.atan2(- m12, m11)

        else:

            _x = math.atan2(m32, m22)
            _z = 0

    elif order == 'YXZ':

        _x = math.asin(- clamp(m23, - 1, 1))

        if (abs(m23) < 0.9999999):

            _y = math.atan2(m13, m33)
            _z = math.atan2(m21, m22)

        else:

            _y = math.atan2(- m31, m11)
            _z = 0

    elif order == 'ZXY':

        _x = math.asin(clamp(m32, - 1, 1))

        if (abs(m32) < 0.9999999):

            _y = math.atan2(- m31, m33)
            _z = math.atan2(- m12, m22)

        else:

            _y = 0
            _z = math.atan2(m21, m11)

    elif order == 'ZYX':

        _y = math.asin(- clamp(m31, - 1, 1))

        if (abs(m31) < 0.9999999):

            _x = math.atan2(m32, m33)
            _z = math.atan2(m21, m11)

        else:

            _x = 0
            _z = math.atan2(- m12, m22)

    elif order == 'YZX':

        _z = math.asin(clamp(m21, - 1, 1))

        if (abs(m21) < 0.9999999):

            _x = math.atan2(- m23, m22)
            _y = math.atan2(- m31, m11)

        else:

            _x = 0
            _y = math.atan2(m13, m33)

    elif order == 'XZY':

        _z = math.asin(- clamp(m12, - 1, 1))

        if (abs(m12) < 0.9999999):

            _x = math.atan2(m32, m22)
            _y = math.atan2(m13, m11)

        else:

            _x = math.atan2(- m23, m33)
            _y = 0

    else:

        print(
            'THREE.Euler: .setFromRotationMatrix() encountered an unknown order: ' + order)

    return _x, _y, _z, order


def quaternion2euler(x, y, z, w, order='XYZ'):

    return matrix42euler(quaternion2matrix4(x, y, z, w), order)


if __name__ == '__main__':

    order = 'ZYX'

    res = euler2quaternion(-0.6, 0.6, -1.7, order=order)

    print(res)

    res = quaternion2euler(*res, order=order)

    print(res)

    res = euler2quaternion(*res)

    print(res)

    res = quaternion2euler(*res, order=order)

    print(res)
