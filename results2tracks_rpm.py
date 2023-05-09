import os
import joblib
import numpy as np
import json
import uuid
import math
import subprocess
import platform

from utils import *
# from scipy.spatial.transform import Rotation

rpm_bones = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
]

smpl_skeleton = {
    0: 'pelvis',
    1: 'left_hip',
    2: 'right_hip',
    3: 'spine1',
    4: 'left_knee',
    5: 'right_knee',
    6: 'spine2',
    7: 'left_ankle',
    8: 'right_ankle',
    9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}

smpl_skeleton_idx = {value: key for key, value in smpl_skeleton.items()}


# axis_angle = np.array([1.0, 2.0, 3.0])
# quaternion = axis_angle_to_quaternion(axis_angle)
# print(quaternion)


def get_limb_tracks(pose_frame, limb_name, limb_upvector):
    axis_angles = pose_frame.reshape((24, 3))
    quaternions = np.apply_along_axis(
        axis_angle_to_quaternion, axis=1, arr=axis_angles)

    limb_quaternion = quaternions[smpl_skeleton_idx[limb_name]]

    target_vector = apply_quaternion_to_vector(limb_quaternion, limb_upvector)

    return target_vector / np.linalg.norm(target_vector)


def LeftArmAxisMapping(x, y, z, w):
    """
    * for left arm, left forearm
    * global +x is local +y
    * global +y is local -z
    * global +z is local -x
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order='XYZ')

    return euler2quaternion(-zr, xr, -yr, order='YZX')


def RightArmAxisMapping(x, y, z, w):
    """
    * for right arm, right forearm
    * global +x is local -y
    * global +y is local -z
    * global +z is local +x
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order='XYZ')

    return euler2quaternion(zr, -xr, -yr, order='YZX')


def LeftLegAxisMapping(x, y, z, w):
    """
    * for left thigh, calf
    * global +x is local -x
    * global +y is local -y
    * global +z is local +z
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order='XYZ')

    return euler2quaternion(-xr, -yr, zr, order='XYZ')


def RightLegAxisMapping(x, y, z, w):
    """
    * for right thigh, calf
    * global +x is local -x
    * global +y is local -y
    * global +z is local +z
    """

    xr, yr, zr, _ = quaternion2euler(x, y, z, w, order='XYZ')

    return euler2quaternion(-xr, -yr, zr, order='XYZ')


def save_rpm_tracks(output_pkl):
    """
    save VIBE predicted pose to Rready Player Me bones and quaternions
    """

    output = joblib.load(output_pkl)

    output = output[list(output.keys())[0]]

    tracks = {}

    for bone in rpm_bones:

        tracks[bone + '.quaternion'] = {
            "name": bone + '.quaternion',
            "type": "quaternion",
            "times": [],
            "values": [],
            "quaternions": [],
        }

    # print(tracks)

    millisec = 0
    interval = 1000 / 30

    rpm_smpl_mapping = {
        'Hips': 'pelvis',
        'Spine': 'spine1',
        'Spine1': 'spine2',
        'Spine2': 'spine3',
        # 'Neck': 'neck',
        # 'Head': 'head',

        'LeftUpLeg': 'left_hip',
        'RightUpLeg': 'right_hip',
        'LeftLeg': 'left_knee',
        'RightLeg': 'right_knee',
        # 'LeftFoot': 'left_ankle',
        # 'RightFoot': 'right_ankle',
        # 'LeftToeBase': 'left_foot',
        # 'RightToeBase': 'right_foot',

        'LeftShoulder': 'left_collar',
        'RightShoulder': 'right_collar',
        'LeftArm': 'left_shoulder',
        'RightArm': 'right_shoulder',
        'LeftForeArm': 'left_elbow',
        'RightForeArm': 'right_elbow',
        # 'LeftHand': 'right_wrist',
        # 'RightHand': 'left_wrist',
    }

    """
    // after init rotation, the new basis of leftshoulder is x: (0,0,-1), y: (1,0,0), z:(0,-1,0)
    LeftShoulder: new THREE.Euler(Math.PI / 2, 0, -Math.PI / 2),
    // after init rotation, the new basis of RightShoulder is x: (0,0,1), y: (-1,0,0), z:(0,-1,0)
    RightShoulder: new THREE.Euler(Math.PI / 2, 0, Math.PI / 2),
    // after init rotation, the new basis of LeftUpLeg/RightUpLeg is x: (-1,0,0), y: (-1,0,0), z:(0,0,1)
    LeftUpLeg: new THREE.Euler(0, 0, -Math.PI),
    RightUpLeg: new THREE.Euler(0, 0, Math.PI),
    LeftFoot: new THREE.Euler(0.9, 0, 0),
    RightFoot: new THREE.Euler(0.9, 0, 0),
    """
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

    for pose_frame in output['pose']:
        # print(pose.shape)

        axis_angles = pose_frame.reshape((24, 3))

        quaternions = np.apply_along_axis(
            axis_angle_to_quaternion, axis=1, arr=axis_angles)

        for bone in rpm_bones:

            tracks[bone + '.quaternion']['times'].append(millisec)

            if bone in rpm_smpl_mapping:

                quaternion = quaternions[smpl_skeleton_idx[rpm_smpl_mapping[bone]]]

                if bone in init_rotation:
                    quaternion = combine_quaternions(
                        quaternion, init_rotation[bone])

                if bone in axes_mapping:
                    quaternion = np.array(axes_mapping[bone](*quaternion))

                # the order of quaternion must be x, y, z, w
                tracks[bone +
                       '.quaternion']['quaternions'].append(quaternion.tolist())

                for num in quaternion:
                    tracks[bone + '.quaternion']['values'].append(num)

            else:

                tracks[bone + '.quaternion']['values'].append(0)
                tracks[bone + '.quaternion']['values'].append(0)
                tracks[bone + '.quaternion']['values'].append(0)
                tracks[bone + '.quaternion']['values'].append(1)

        millisec += interval

    return tracks


if __name__ == '__main__':

    vibe_results_dir = os.path.join('.', 'vibe_results')

    for item in os.listdir(vibe_results_dir):
        # for item in ['2_29-40_29-44_vibe.pkl']:

        fpath = os.path.join(vibe_results_dir, item)

        print(f"reading tracks from {fpath}")

        tracks = save_rpm_tracks(fpath)

        # print(tracks)
        # exit()

        animation_name = os.path.basename(fpath).replace('_vibe.pkl', '')

        animation = {
            "name": animation_name,
            "duration": 10,
            "tracks": list(tracks.values()),
            "uuid": str(uuid.uuid4()),
            "blendMode": 2500,
        }

        filename = os.path.join('tracks_json', animation_name + '_rpm.json')

        with open(filename, 'w') as f:

            json.dump(animation, f)

            print('animation data saved to ' + filename)

        if platform.system() == "Linux":

            command = 'cp /home/hlz/VIBE/results_interpreter/tracks_json/* ~/my-app/web/public/animations/'
            # command = ' '.join(command)
            subprocess.run(command, shell=True)

            # subprocess.run(
            # ['ls', "/home/hlz/VIBE/results_interpreter/tracks_json/"])
            print('files copied to ' + '~/my-app/web/public/animations/')
