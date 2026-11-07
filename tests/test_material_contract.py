"""Run directly: python tests/test_material_contract.py (stdlib only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PhysGaussian'))
from gaussian_property_adapter import MATERIAL_VOCAB, material_parameters, parse_predictions


def check():
    manifest = [{'object_id': 9, 'name': 'bird'}, {'object_id': 2, 'name': 'floor'}]
    records = parse_predictions('9.png,white, fluffy bird,foam,10-20,Shore A\n2.png,floor,stone,80,Shore D', manifest)
    floor, bird = records
    assert bird['caption'] == 'white, fluffy bird'
    assert bird['material_id'] == MATERIAL_VOCAB.index('foam')
    assert not bird['hardness_used_for_parameters']
    assert floor['material'] == 'unknown' and floor['material_id'] == 0
    assert floor['fallback_material'] == 'plastic' and floor['hardness_range'] == [80., 80.]
    assert floor['material_parameters']['requested']['E'] == 8e7
    assert floor['material_parameters']['actual']['E'] == 1e7
    assert floor['material_parameters']['clamped'] == ['E']
    for field, value in [('E', float('nan')), ('density', -1), ('nu', .5)]:
        bad = dict(E=1e5, nu=.3, density=1000); bad[field] = value
        try:
            material_parameters(bad)
        except ValueError:
            pass
        else:
            raise AssertionError((field, value))
    for text in ['9.png,error,-1', '9.png,bird,foam,20-10,Shore A',
                 '9.png,bird,foam,20,Shore A\n9.png,bird,foam,20,Shore A',
                 '8.png,bird,foam,20,Shore A']:
        try:
            parse_predictions(text, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError(text)
    print('material contract checks passed')


def test_material_contract():
    check()


if __name__ == '__main__':
    check()
