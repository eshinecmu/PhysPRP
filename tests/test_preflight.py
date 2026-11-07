"""CPU-only preflight check using temporary files and stdlib import probes."""
import importlib.util
from pathlib import Path
import sys
import tempfile

spec = importlib.util.spec_from_file_location('preflight', Path(__file__).parents[1] / 'tools/preflight_inference.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_preflight():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'source').mkdir()
        (root / 'source' / 'marker').write_text('source')
        (root / 'model').write_bytes(b'model')
        runtime = {'python': dict.fromkeys(('rest', 'gp', 'phys'), sys.executable),
                   'paths': dict.fromkeys(module.DIRECTORIES, 'source'),
                   'weights': [{'path': 'model', 'source': 'test fixture', 'sha256': module.digest(root / 'model')}],
                   'probes': dict.fromkeys(('rest', 'gp', 'phys'), ['json']),
                   'env': {'TORCH_HOME': 'cache'}, 'pythonpath': ['source']}
        runtime['paths'].update(dict.fromkeys(module.FILES, 'model'))
        report = module.inspect(runtime, root, True)
        assert report['status'] == 'assets_ready', report
        assert report['simulation_blockers'] and report['interpreters']['rest']['modules']['json']
        original = module.NATIVE_ASSET_PROBES
        module.NATIVE_ASSET_PROBES = ['math']
        cached = module.inspect(runtime, root, True, cached_rest_assets=True)
        module.NATIVE_ASSET_PROBES = original
        assert cached['status'] == 'assets_ready' and cached['rest_probe_scope'] == 'native_asset_adapter_only'
        assert cached['interpreters']['rest']['requested_modules'] == ['math']
        assert cached['interpreters']['phys']['requested_modules'] == ['json']
        assert module.inspect(runtime, root)['status'] == 'failed'
        runtime['relation_checkpoint'] = 'model'
        assert module.inspect(runtime, root)['status'] == 'preflight_passed'
        runtime['weights'][0]['sha256'] = 'wrong'
        assert module.inspect(runtime, root)['status'] == 'failed'
        runtime['weights'][0].pop('sha256')
        (root / 'model').write_bytes(b'')
        assert module.inspect(runtime, root, True)['status'] == 'failed'
        runtime['paths']['sam3_source'] = 'nonexistent'
        assert not module.inspect(runtime, root, True)['paths']['sam3_source']['valid']
    print('preflight checks passed')


if __name__ == '__main__':
    test_preflight()
