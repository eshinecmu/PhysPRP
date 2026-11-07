import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('first_frame', Path(__file__).parents[1] / 'tools/validate_first_frame.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_camera():
    module.self_check()
    import numpy as np
    zz, xx = np.mgrid[1:2:20j, -1:1:20j]
    points = np.stack((xx, -.4 + .2 * zz, zz), -1)
    config = dict(ground_from_pointmap=True, gravity_from_ground=True,
                  scene_scale=.5, mpm_origin=[1, 1, 0], g=[0, -9.81, 0],
                  boundary_conditions=[dict(type='bounding_box'), dict(type='surface_collider', point=[0,0,0])])
    resolved, report = module.resolve_ground(points, np.ones((20,20), bool), config)
    normal = np.array([0, 1, -.2]); normal /= np.linalg.norm(normal)
    assert np.allclose(report['normal'], normal)
    assert report['metric_rms'] < 1e-12
    assert np.allclose(resolved['g'], -9.81 * normal)
    assert np.allclose(resolved['boundary_conditions'][-1]['point'], points.mean((0,1)) * .5 + [1,1,0])
    assert len(resolved['boundary_conditions']) == 2
    assert config['g'] == [0, -9.81, 0]  # input retained
    config['gravity_from_ground'] = False
    assert module.resolve_ground(points, np.ones((20,20), bool), config)[0]['g'] == config['g']
    try:
        module.resolve_ground(points, np.zeros((20,20), bool), config)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty floor must fail')

if __name__ == '__main__':
    test_camera()
    print('single-frame pinhole self-check passed')
