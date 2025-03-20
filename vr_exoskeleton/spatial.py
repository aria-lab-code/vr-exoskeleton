import numpy as np
from scipy.spatial.transform import Rotation


def to_pitch(a_v1, a_v2):
    if isinstance(a_v1, list):
        a_v1 = np.array(a_v1)
    if isinstance(a_v2, list):
        a_v2 = np.array(a_v2)
    norm1 = np.sqrt(a_v1[:, 1] ** 2 + a_v1[:, 2] ** 2)
    norm2 = np.sqrt(a_v2[:, 1] ** 2 + a_v2[:, 2] ** 2)
    return np.arcsin(a_v2[:, 1] / norm2) - np.arcsin(a_v1[:, 1] / norm1)
    # return np.arccos(a_v2[:, 2] / norm2) - np.arccos(a_v1[:, 2] / norm1)


def to_yaw(a_v1, a_v2):
    if isinstance(a_v1, list):
        a_v1 = np.array(a_v1)
    if isinstance(a_v2, list):
        a_v2 = np.array(a_v2)
    norm1 = np.sqrt(a_v1[:, 0] ** 2 + a_v1[:, 2] ** 2)
    norm2 = np.sqrt(a_v2[:, 0] ** 2 + a_v2[:, 2] ** 2)
    return np.arccos(a_v2[:, 0] / norm2) - np.arccos(a_v1[:, 0] / norm1)
    # return np.arcsin(a_v2[:, 2] / norm2) - np.arcsin(a_v1[:, 2] / norm1)


def to_angle_difference(a_v1, a_v2):
    if isinstance(a_v1, list):
        a_v1 = np.array(a_v1)
    if isinstance(a_v2, list):
        a_v2 = np.array(a_v2)
    norm1 = np.linalg.norm(a_v1, axis=1)
    norm2 = np.linalg.norm(a_v2, axis=1)
    dot_norm = np.sum(a_v1 * a_v2, axis=1) / norm1 / norm2
    return np.arccos(np.clip(dot_norm, -1., 1.))  # Floating point math could yield values outside [-1, 1].


def to_rotation_difference(a_quat1, a_quat2):
    a_diff = list()
    for quat1, quat2 in zip(a_quat1, a_quat2):
        # https://stackoverflow.com/a/22167097/1559071
        # diff * q1 = q2 --> diff = q2 * inverse(q1)
        q1 = Rotation.from_quat(quat1)
        q2 = Rotation.from_quat(quat2)
        q_diff = q2 * q1.inv()
        a_diff.append(q_diff.as_quat())
    return np.stack(a_diff)


def main():
    assert np.isclose(to_pitch([[0., 1., 1.]], [[0., 0., 1.]])[0], -np.pi / 4)
    assert np.isclose(to_pitch([[0., -1., 1.]], [[0., 0., 1.]])[0], np.pi / 4)

    assert np.isclose(to_yaw([[0., 0., 1.]], [[1., 0., np.sqrt(3)]])[0], -np.pi / 6)
    assert np.isclose(to_yaw([[0., 0., 1.]], [[-1., 0., np.sqrt(3)]])[0], np.pi / 6)
    assert np.isclose(to_yaw([[1., 0., np.sqrt(3)]], [[-1., 0., np.sqrt(3)]])[0], np.pi / 3)
    assert np.isclose(to_yaw([[-np.sqrt(3), 0, 1.]], [[1., 0., np.sqrt(3)]])[0], -np.pi / 2)

    assert np.isclose(to_angle_difference([[0., 0., 1.]], [[0., 0., 1.]])[0], 0.)
    assert np.isclose(to_angle_difference([[0., 0., 1.]], [[1., 0., 0.]])[0], np.pi / 2)

    quat1 = np.array([0.134555, 0.024022, 0.012331, -0.990538])
    quat2 = np.array([0.134438, 0.023766, 0.012284, -0.990561])
    quat_diff = to_rotation_difference([quat1], [quat2])[0]
    q2_hat = Rotation.from_quat(quat_diff) * Rotation.from_quat(quat1)  # Apply rotational difference to first rotation.
    assert np.allclose(q2_hat.as_quat(), quat2)


if __name__ == '__main__':
    main()
