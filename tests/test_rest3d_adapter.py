import numpy as np

from tools.prepare_rest3d_object import rigid_transform


def test_rigid_transform_recovers_pose():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.array([2, -1, 0.5])
    recovered_r, recovered_t = rigid_transform(source, source @ rotation.T + translation)
    assert np.allclose(recovered_r, rotation)
    assert np.allclose(recovered_t, translation)
