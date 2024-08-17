import joblib
import os
import math

import numpy as np


def clamp(n, minn, maxn):
    """
    Constrains a number `n` between a minimum value `minn` and a maximum value `maxn`.
    Returns `n` if it is within the range, otherwise returns the nearest boundary value.
    """
    return max(min(maxn, n), minn)


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


def combine_quaternions(a, b):
    """Combine two quaternions to represent their relative rotation."""
    a = np.array(a) / np.linalg.norm(a)
    b = np.array(b) / np.linalg.norm(b)
    c = multiply_quaternions(a, b)
    return c / np.linalg.norm(c)


def axis_angle_to_quaternion(axis_angle):

    # print(axis_angle)

    axis = axis_angle / np.linalg.norm(axis_angle)
    angle = np.linalg.norm(axis_angle)
    half_angle = angle / 2

    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)

    return np.array([x, y, z, w])


def euler2quaternion(x, y, z, order="XYZ"):
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

    if order == "XYZ":
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3

    elif order == "YXZ":
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3

    elif order == "ZXY":
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3

    elif order == "ZYX":
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3

    elif order == "YZX":
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3

    elif order == "XZY":
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3

    else:
        print("`euler2quaternion` encountered an unknown order: " + order)

    return [_x, _y, _z, _w]


def matrix42euler(te, order="XYZ"):
    """ """

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

    if order == "XYZ":

        _y = math.asin(clamp(m13, -1, 1))

        if abs(m13) < 0.9999999:

            _x = math.atan2(-m23, m33)
            _z = math.atan2(-m12, m11)

        else:

            _x = math.atan2(m32, m22)
            _z = 0

    elif order == "YXZ":

        _x = math.asin(-clamp(m23, -1, 1))

        if abs(m23) < 0.9999999:

            _y = math.atan2(m13, m33)
            _z = math.atan2(m21, m22)

        else:

            _y = math.atan2(-m31, m11)
            _z = 0

    elif order == "ZXY":

        _x = math.asin(clamp(m32, -1, 1))

        if abs(m32) < 0.9999999:

            _y = math.atan2(-m31, m33)
            _z = math.atan2(-m12, m22)

        else:

            _y = 0
            _z = math.atan2(m21, m11)

    elif order == "ZYX":

        _y = math.asin(-clamp(m31, -1, 1))

        if abs(m31) < 0.9999999:

            _x = math.atan2(m32, m33)
            _z = math.atan2(m21, m11)

        else:

            _x = 0
            _z = math.atan2(-m12, m22)

    elif order == "YZX":

        _z = math.asin(clamp(m21, -1, 1))

        if abs(m21) < 0.9999999:

            _x = math.atan2(-m23, m22)
            _y = math.atan2(-m31, m11)

        else:

            _x = 0
            _y = math.atan2(m13, m33)

    elif order == "XZY":

        _z = math.asin(-clamp(m12, -1, 1))

        if abs(m12) < 0.9999999:

            _x = math.atan2(m32, m22)
            _y = math.atan2(m13, m11)

        else:

            _x = math.atan2(-m23, m33)
            _y = 0

    else:

        print(
            "THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "
            + order
        )

    return _x, _y, _z, order


def quaternion2matrix4(x, y, z, w):

    te = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

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

    te[0] = 1 - (yy + zz)
    te[1] = xy + wz
    te[2] = xz - wy
    te[3] = 0

    te[4] = xy - wz
    te[5] = 1 - (xx + zz)
    te[6] = yz + wx
    te[7] = 0

    te[8] = xz + wy
    te[9] = yz - wx
    te[10] = 1 - (xx + yy)
    te[11] = 0

    te[12] = 0
    te[13] = 0
    te[14] = 0
    te[15] = 1

    return te


def quaternion2euler(x, y, z, w, order="XYZ"):

    return matrix42euler(quaternion2matrix4(x, y, z, w), order)


def LeftArmAxisMapping(x, y, z, w):
    """
    * for left arm, left forearm
    * global +x is local +y
    * global +y is local -z
    * global +z is local -x
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order="XYZ")

    return euler2quaternion(-zr, xr, -yr, order="YZX")


def RightArmAxisMapping(x, y, z, w):
    """
    * for right arm, right forearm
    * global +x is local -y
    * global +y is local -z
    * global +z is local +x
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order="XYZ")

    return euler2quaternion(zr, -xr, -yr, order="YZX")


def LeftLegAxisMapping(x, y, z, w):
    """
    * for left thigh, calf
    * global +x is local -x
    * global +y is local -y
    * global +z is local +z
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order="XYZ")

    return euler2quaternion(-xr, -yr, zr, order="XYZ")


def RightLegAxisMapping(x, y, z, w):
    """
    * for right thigh, calf
    * global +x is local -x
    * global +y is local -y
    * global +z is local +z
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order="XYZ")

    return euler2quaternion(-xr, -yr, zr, order="XYZ")


output_pth = os.path.join(".", "output")

# Assuming output_pth is defined
output_file = os.path.join(output_pth, "madfit1", "wham_output.pkl")
loaded_results = joblib.load(output_file)

pose = loaded_results[0]["pose"]
trans = loaded_results[0]["trans"]
pose_world = loaded_results[0]["pose_world"]
trans_world = loaded_results[0]["trans_world"]
betas = loaded_results[0]["betas"]
verts = loaded_results[0]["verts"]
frame_ids = loaded_results[0]["frame_ids"]


rpm_bones = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]


smpl_skeleton = {
    0: "pelvis",
    1: "left_hip",
    2: "right_hip",
    3: "spine1",
    4: "left_knee",
    5: "right_knee",
    6: "spine2",
    7: "left_ankle",
    8: "right_ankle",
    9: "spine3",
    10: "left_foot",
    11: "right_foot",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
    22: "left_hand",
    23: "right_hand",
}

smpl_skeleton_idx = {value: key for key, value in smpl_skeleton.items()}

rpm_smpl_mapping = {
    "Hips": "pelvis",
    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",
    # 'Neck': 'neck',
    # 'Head': 'head',
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    # 'LeftFoot': 'left_ankle',
    # 'RightFoot': 'right_ankle',
    # 'LeftToeBase': 'left_foot',
    # 'RightToeBase': 'right_foot',
    "LeftShoulder": "left_collar",
    "RightShoulder": "right_collar",
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
    # 'LeftHand': 'right_wrist',
    # 'RightHand': 'left_wrist',
}

init_rotation = {
    "LeftShoulder": euler2quaternion(math.pi / 2, 0, -math.pi / 2),
    "RightShoulder": euler2quaternion(math.pi / 2, 0, math.pi / 2),
    "LeftUpLeg": euler2quaternion(0, 0, -math.pi),
    "RightUpLeg": euler2quaternion(0, 0, math.pi),
    "LeftFoot": euler2quaternion(0.9, 0, 0),
    "RightFoot": euler2quaternion(0.9, 0, 0),
}

axes_mapping = {
    "LeftArm": LeftArmAxisMapping,
    "RightArm": RightArmAxisMapping,
    "LeftForeArm": LeftArmAxisMapping,
    "RightForeArm": RightArmAxisMapping,
    "LeftLeg": LeftLegAxisMapping,
    "RightLeg": RightLegAxisMapping,
}


tracks = {}

for bone in rpm_bones:

    tracks[bone + ".quaternion"] = {
        "name": bone + ".quaternion",
        "type": "quaternion",
        "times": [],
        "values": [],
        "quaternions": [],
    }


for pose_frame in pose:
    # print(pose.shape)

    axis_angles = pose_frame.reshape((24, 3))

    # print(axis_angles.shape)

    quaternions = np.apply_along_axis(axis_angle_to_quaternion, axis=1, arr=axis_angles)

    # print(quaternions)

    for bone in rpm_bones:

        if bone in rpm_smpl_mapping:

            quaternion = quaternions[smpl_skeleton_idx[rpm_smpl_mapping[bone]]]

            # print(bone, quaternion)

            if bone in init_rotation:
                quaternion = combine_quaternions(quaternion, init_rotation[bone])

            if bone in axes_mapping:
                quaternion = np.array(axes_mapping[bone](*quaternion))

            # the order of quaternion must be x, y, z, w
            tracks[bone + ".quaternion"]["quaternions"].append(quaternion.tolist())

            for num in quaternion:
                tracks[bone + ".quaternion"]["values"].append(num)

        else:

            tracks[bone + ".quaternion"]["values"].append(0)
            tracks[bone + ".quaternion"]["values"].append(0)
            tracks[bone + ".quaternion"]["values"].append(0)
            tracks[bone + ".quaternion"]["values"].append(1)


print(tracks)
