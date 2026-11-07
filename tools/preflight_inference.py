"""Read-only CPU environment/weight inventory; no model construction or downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_PROBES = {'rest': ['torch', 'sam3', 'sam3d_objects'],
                  'gp': ['torch'], 'phys': ['torch', 'warp', 'diff_gaussian_rasterization']}
NATIVE_ASSET_PROBES = ['numpy', 'scipy', 'plyfile']
SAFE_ENV = {'CONDA_PREFIX', 'SAM3_CHECKPOINT', 'SAM3D_CONFIG', 'TORCH_HOME', 'XDG_CACHE_HOME', 'PHYS_GS_ROOT'}
DIRECTORIES = ('gaussian_splatting', 'sam3_source', 'sam3d_source')
FILES = ('sam3_checkpoint', 'sam3d_config')
PROBE = '''import importlib,json,sys
result={"python":sys.version.split()[0],"modules":{},"errors":[]}
for name in json.loads(sys.argv[1]):
 try:
  module=importlib.import_module(name)
  result["modules"][name]={"version":str(getattr(module,"__version__","unreported")),"file":getattr(module,"__file__",None)}
 except Exception as exc:
  result["errors"].append(name+": "+type(exc).__name__)
print("PHYS_PRP_PROBE="+json.dumps(result))
'''


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def inspect(runtime, base, assets_only=False, cached_rest_assets=False):
    report = {'status': 'failed', 'assets_only': assets_only, 'paths': {}, 'weights': [],
              'interpreters': {}, 'repositories': {}, 'errors': [], 'simulation_blockers': [],
              'rest_probe_scope': 'native_asset_adapter_only' if cached_rest_assets else 'reconstruction'}

    def path(value):
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else base / candidate

    def validate(label, value, directory=False):
        if not value:
            report['errors'].append(label + ': missing declaration')
            return None
        candidate = path(value)
        valid = candidate.is_dir() if directory else candidate.is_file()
        if valid:
            valid = next(candidate.iterdir(), None) is not None if directory else candidate.stat().st_size > 0
        report['paths'][label] = {'path': str(candidate), 'valid': valid}
        if not valid:
            report['errors'].append(label + ': absent, empty, wrong type, or broken link')
            return None
        return candidate

    paths = runtime.get('paths', {})
    checked = {name: validate(name, paths.get(name), name in DIRECTORIES) for name in (*DIRECTORIES, *FILES)}
    if paths.get('rest_root'):
        checked['rest_root'] = validate('rest_root', paths['rest_root'], True)
    relation = runtime.get('relation_checkpoint')
    if not relation:
        report['simulation_blockers'].append('Missing compatible relation_checkpoint; assets may run but full inference cannot pass')
    elif validate('relation_checkpoint', relation) is None:
        report['simulation_blockers'].append('Invalid relation_checkpoint')
    unknown_env = set(runtime.get('env', {})) - SAFE_ENV
    if unknown_env:
        report['errors'].append('Unsupported environment keys: ' + ','.join(sorted(unknown_env)))
    weights = list(runtime.get('weights', []))
    declared = {str(path(item['path']).resolve()) for item in weights}
    for name, value in [('sam3_checkpoint', paths.get('sam3_checkpoint')), ('relation_checkpoint', relation)]:
        if value and str(path(value).resolve()) not in declared:
            weights.append({'path': value, 'source': None})
    if not weights:
        report['errors'].append('No weights declared')
    hashes = {}
    for item in weights:
        candidate = validate('weight:' + item['path'], item['path'])
        record = {'path': str(path(item['path'])), 'source': item.get('source')}
        if not item.get('source'):
            report['errors'].append('Weight source missing: ' + item['path'])
        if candidate:
            key = str(candidate.resolve())
            if key not in hashes:
                hashes[key] = digest(candidate)
            record.update(size_bytes=candidate.stat().st_size, sha256=hashes[key])
            if item.get('sha256') and record['sha256'] != item['sha256'].lower():
                report['errors'].append('Weight SHA256 mismatch: ' + item['path'])
        report['weights'].append(record)
    repositories = {'physprp': str(Path(__file__).resolve().parents[1])}
    repositories.update({name: str(value) for name, value in checked.items() if name in DIRECTORIES or name == 'rest_root' if value})
    repositories.update(runtime.get('repositories', {}))
    for name, value in repositories.items():
        try:
            revision = subprocess.run(['git', '-C', str(path(value)), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=15)
            dirty = subprocess.run(['git', '-C', str(path(value)), 'status', '--porcelain'], capture_output=True, text=True, timeout=15)
            report['repositories'][name] = {'path': str(path(value)), 'revision': revision.stdout.strip() if revision.returncode == 0 else None,
                                            'dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}
        except (OSError, subprocess.TimeoutExpired):
            report['repositories'][name] = {'path': str(path(value)), 'revision': None, 'dirty': None}
    for stage in DEFAULT_PROBES:
        interpreter = validate('python.' + stage, runtime.get('python', {}).get(stage))
        if interpreter is None:
            continue
        modules = NATIVE_ASSET_PROBES if cached_rest_assets and stage == 'rest' else runtime.get('probes', {}).get(stage, DEFAULT_PROBES[stage])
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        env.update({key: str(path(value)) for key, value in runtime.get('env', {}).items() if key in SAFE_ENV})
        roots = [str(p) for name, p in checked.items() if p and (name in DIRECTORIES or name == 'rest_root')]
        roots.extend(str(path(value)) for value in runtime.get('pythonpath', []))
        env['PYTHONPATH'] = os.pathsep.join(roots + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        try:
            process = subprocess.run([str(interpreter), '-B', '-c', PROBE, json.dumps(modules)], env=env,
                                     capture_output=True, text=True, timeout=600)
            lines = [line for line in process.stdout.splitlines() if line.startswith('PHYS_PRP_PROBE=')]
            result = json.loads(lines[-1].split('=', 1)[1]) if lines else {'errors': ['Probe returned no structured result']}
            result['exit_code'] = process.returncode
            result['requested_modules'] = modules
            report['interpreters'][stage] = result
            if process.returncode or result['errors']:
                report['errors'].append('Import probe failed: ' + stage)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['interpreters'][stage] = {'errors': [type(exc).__name__]}
            report['errors'].append('Interpreter probe failed: ' + stage)
    if not report['errors'] and (assets_only or not report['simulation_blockers']):
        report['status'] = 'assets_ready' if assets_only else 'preflight_passed'
    report['limitations'] = ['Import success does not validate GPU kernels or checkpoint architecture compatibility',
                             'Model construction and full inference are not performed']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-only', action='store_true')
    parser.add_argument('--cached-rest-assets', action='store_true', help='Probe native adapter only; runner separately verifies explicit segmentation/reconstruction caches')
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text(encoding='utf-8'))
    report = inspect(runtime, args.runtime.resolve().parent, args.assets_only, args.cached_rest_assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['status'] + ': ' + str(args.output))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
