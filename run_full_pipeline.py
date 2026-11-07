"""Single image -> real reconstruction/materials/native Gaussians -> Warp inference.

Each stage records commands and artifact hashes. Resume requires identical inputs,
configuration, source files, and stage outputs; it never borrows unrelated results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for base in ('run_full_pipeline.py', 'tools', 'REST3D/scripts', 'REST3D/rest3d', 'GaussianProperty', 'PhysGaussian'):
        root = ROOT / base
        files = [root] if root.is_file() else sorted(root.rglob('*.py'))
        for path in files:
            if any(part in ('envs', '.venv', 'vendor_zhuzhihan', 'checkpoints', 'outputs', 'gaussian-splatting') for part in path.relative_to(ROOT).parts):
                continue
            digest.update(str(path.relative_to(ROOT)).replace('\\', '/').encode())
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def stage(name, command, outputs, work, env, cwd=ROOT, resume=False):
    receipt = work / 'validation' / (name + '.json')
    expected = dict(command=list(map(str, command)), cwd=str(cwd))
    if resume and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('status') == 'passed' and all(previous.get(k) == v for k, v in expected.items()):
            if all(Path(p).is_file() and sha256(p) == h for p, h in previous['outputs'].items()):
                return
        raise ValueError(f'Cannot resume changed/incomplete stage {name}; use a clean work directory')
    report = dict(expected, status='running', started=time.time())
    receipt.write_text(json.dumps(report, indent=2))
    print('+', name, *command, flush=True)
    try:
        with (work / 'validation' / (name + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        artifacts = []
        for output in outputs:
            if output.is_dir():
                artifacts.extend(p for p in output.rglob('*') if p.is_file())
            elif output.is_file():
                artifacts.append(output)
            else:
                raise FileNotFoundError(f'Stage {name} did not produce {output}')
        if not artifacts:
            raise ValueError(f'Stage {name} produced no artifacts')
        report.update(status='passed', outputs={str(p): sha256(p) for p in sorted(artifacts)})
    except Exception as exc:
        report.update(status='failed', error=str(exc), exit_code=getattr(exc, 'returncode', None))
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started']
        receipt.write_text(json.dumps(report, indent=2))


def copy_reconstruction_cache(cache, scene, work, identity):
    producer = json.loads((cache/'run.json').read_text())
    receipt_path = cache/'validation/reconstruction.json'
    receipt = json.loads(receipt_path.read_text())
    if any(producer['identity'][key] != identity[key] for key in ('image_sha256', 'runtime_sha256')):
        raise ValueError('Reconstruction cache image/runtime mismatch')
    source_scene = cache/'reconstruction'/scene.name
    def contents(root):
        return {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}
    if contents(source_scene/'stage1') != contents(scene/'stage1'):
        raise ValueError('Reconstruction cache segmentation mismatch')
    raw = source_scene/'stage2/scene_raw'
    actual = {str(p):sha256(p) for p in raw.rglob('*') if p.is_file()}
    if receipt.get('status') != 'passed' or not actual or actual != receipt.get('outputs'):
        raise ValueError('Reconstruction cache is incomplete or modified')
    target = scene/'stage2/scene_raw'
    shutil.copytree(raw, target)
    (work/'validation/reconstruction.json').write_text(json.dumps(dict(status='verified_cache', producer=str(cache), producer_identity=producer['identity'], producer_receipt_sha256=sha256(receipt_path), outputs={str(p):sha256(p) for p in target.rglob('*') if p.is_file()}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Explicit single-image scene and simulation config')
    parser.add_argument('--runtime', type=Path, required=True, help='External interpreters, sources, weights and env')
    parser.add_argument('--work_dir', type=Path, required=True)
    parser.add_argument('--assets_only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--segmentation_cache', type=Path, help='Explicit prior run with matching image/prompt and verified stage hashes')
    parser.add_argument('--reconstruction_cache', type=Path, help='Explicit successful native reconstruction with matching image/runtime/segmentation and verified hashes')
    parser.add_argument('--without_relation', action='store_true', help='Explicit unlearned MPM baseline; never counts as full inference')
    parser.add_argument('--model_scope', choices=('interaction', 'full'), help='REST3D objects to reconstruct and simulate')
    parser.add_argument('--subject', help='Exact REST3D object mask name required for interaction scope')
    args = parser.parse_args()
    image, config_path, runtime_path = (p.resolve(strict=True) for p in (args.image, args.config, args.runtime))
    work = args.work_dir.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    model_scope = args.model_scope or config.get('model_scope', 'full')
    subject = args.subject or config.get('subject')
    if model_scope == 'interaction' and not subject:
        parser.error('--subject is required when --model_scope interaction')
    def external(value):
        path = Path(value)
        return str((runtime_path.parent / path).resolve()) if not path.is_absolute() else str(path)
    interpreters = {k: external(v) for k,v in runtime['python'].items()}
    env = os.environ.copy()
    env.update({k:str(v) for k,v in runtime.get('env', {}).items()})
    env['PYTHONUTF8'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PHYS_GS_ROOT'] = external(runtime['paths']['gaussian_splatting'])
    env['SAM3_CHECKPOINT'] = external(runtime['paths']['sam3_checkpoint'])
    env['SAM3D_OBJECTS_ROOT'] = external(runtime['paths']['sam3d_source'])
    env['PYTHONUNBUFFERED'] = '1'
    env['SAM3D_CONFIG'] = external(runtime['paths']['sam3d_config'])
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'REST3D'), external(runtime['paths']['sam3_source']), external(runtime['paths']['sam3d_source']), *[external(p) for p in runtime.get('pythonpath', [])]])
    identity = dict(image_sha256=sha256(image), config_sha256=sha256(config_path), runtime_sha256=sha256(runtime_path), source_sha256=source_hash(), mode='single_image', assets_only=args.assets_only, without_relation=args.without_relation, model_scope=model_scope, subject=subject)
    identity['segmentation_cache'] = str(args.segmentation_cache.resolve()) if args.segmentation_cache else None
    identity['reconstruction_cache'] = str(args.reconstruction_cache.resolve()) if args.reconstruction_cache else None
    run_file = work / 'run.json'
    if work.exists() and any(work.iterdir()):
        if not args.resume or not run_file.exists() or json.loads(run_file.read_text())['identity'] != identity:
            raise ValueError('Run directory is not empty or belongs to different inputs/config/source; use a new directory')
    (work / 'validation').mkdir(parents=True, exist_ok=True)
    (work / 'input').mkdir(exist_ok=True)
    local_image = work / 'input' / image.name
    shutil.copy2(image, local_image)
    shutil.copy2(config_path, work / 'input' / 'scene.json')
    report = dict(identity=identity, status='running', training_verified=False)
    run_file.write_text(json.dumps(report, indent=2))
    def run(name, command, outputs, cwd=ROOT):
        stage(name, command, outputs, work, env, cwd, args.resume)
    try:
        command = [interpreters['gp'], ROOT/'tools/preflight_inference.py', '--runtime', runtime_path, '--output', work/'validation/preflight_report.json']
        if args.assets_only or args.without_relation:
            command.append('--assets-only')
        if args.segmentation_cache and args.reconstruction_cache:
            command.append('--cached-rest-assets')
        run('preflight', command, [work/'validation/preflight_report.json'])
        reconstruction = work / 'reconstruction'
        scene = reconstruction / image.stem
        command = [interpreters['rest'], ROOT/'REST3D/scripts/infer_scenetree.py', '--image_folder', local_image, '--output_folder', reconstruction, '--vlm_backend', config.get('rest_vlm', 'gpt4o')]
        if config.get('object_prompt'):
            prompt = work/'input/objects.txt'
            prompt.write_text(config['object_prompt'], encoding='utf-8')
            command += ['--vlm_prompt_file', prompt]
        if args.segmentation_cache:
            cache = args.segmentation_cache.resolve(strict=True)
            producer = json.loads((cache/'run.json').read_text())
            old_config = json.loads((cache/'input/scene.json').read_text())
            receipt = json.loads((cache/'validation/segmentation.json').read_text())
            if producer['identity']['image_sha256'] != identity['image_sha256'] or any(old_config.get(k) != config.get(k) for k in ('object_prompt','rest_vlm')):
                raise ValueError('Segmentation cache image/prompt/backend mismatch')
            if receipt.get('status') != 'passed' or not all(Path(p).is_file() and sha256(p)==h for p,h in receipt['outputs'].items()):
                raise ValueError('Segmentation cache is incomplete or modified')
            source_stage = cache/'reconstruction'/image.stem/'stage1'
            shutil.copytree(source_stage, scene/'stage1', dirs_exist_ok=args.resume)
            (work/'validation/segmentation.json').write_text(json.dumps(dict(status='verified_cache',producer=str(cache),producer_identity=producer['identity'],producer_receipt_sha256=sha256(cache/'validation/segmentation.json'),outputs={str(p):sha256(p) for p in (scene/'stage1').rglob('*') if p.is_file()}),indent=2))
        else:
            run('segmentation', command, [scene/'stage1'])
        run('segmentation_validation', [interpreters['gp'], ROOT/'tools/validate_segmentation.py', '--scene-root', scene, '--scene-config', config_path, '--output', scene/'segmentation_validation.json'], [scene/'segmentation_validation.json'])
        selection = work/'object_selection.json'
        run('model_scope', [interpreters['gp'], ROOT/'tools/select_model_scope.py', '--tree', scene/'stage1/scene_tree.json', '--mask-dir', scene/'stage1/segemented_obj', '--scope', model_scope, '--output', selection, *(['--subject', subject] if subject else [])], [selection])
        if args.reconstruction_cache:
            copy_reconstruction_cache(args.reconstruction_cache.resolve(strict=True), scene, work, identity)
        else:
            run('reconstruction', [interpreters['rest'], ROOT/'REST3D/scripts/infer_scene3d.py', '--image_folder', local_image, '--output_folder', reconstruction, '--object_names_file', selection], [scene/'stage2/scene_raw'])
        manifest, relations = work/'object_manifest.json', work/'object_relations.json'
        model = work/'scene'
        run('native_scene', [interpreters['rest'], ROOT/'tools/prepare_rest3d_scene.py', '--scene-root', scene, '--scene-config', config_path, '--model-path', model, '--manifest-out', manifest, '--relations-out', relations, '--selection-file', selection], [manifest, relations, model])
        shutil.copy2(local_image, model/'input.png')
        run('first_frame', [interpreters['phys'], ROOT/'tools/validate_first_frame.py', '--scene-root', scene, '--model-path', model, '--image', local_image, '--output', work/'render', '--scene-config', config_path, '--resolved-config', work/'resolved_scene.json'], [model/'cameras.json', work/'render', work/'resolved_scene.json'])
        properties = work/'materials.json'
        gp = work/'materials'
        run('material_inputs', [interpreters['gp'], ROOT/'prepare_rest3d_gaussian_property.py', '--image', local_image, '--mask_dir', scene/'stage1/segemented_obj', '--manifest', manifest, '--output_root', gp, '--case_name', image.stem], [gp])
        run('material_inference', [interpreters['gp'], ROOT/'GaussianProperty/vlm_predict.py', '--dataset_path', gp, '--vlm', config.get('material_vlm','gpt')], [gp/image.stem/(image.stem+'.txt')], ROOT/'GaussianProperty')
        run('material_contract', [interpreters['gp'], ROOT/'PhysGaussian/gaussian_property_adapter.py', gp/image.stem/(image.stem+'.txt'), properties, '--manifest', manifest], [properties])
        scope_record=json.loads(selection.read_text(encoding='utf-8'))
        report.update(model_scope=model_scope, subject=subject, selected_objects=scope_record['objects'])
        if args.assets_only:
            report.update(status='assets_complete', complete_inference=False, remaining=['compatible_relation_checkpoint','Warp rollout','first_frame_projection','physical_contact_validation','clean_checkout_full_reproduction'])
        else:
            command = [interpreters['phys'], ROOT/'PhysGaussian/gs_simulation.py', '--model_path', model, '--output_path', work/'simulation', '--config', work/'resolved_scene.json', '--object_id_path', model/'point_cloud/iteration_60000/point_cloud.object_id.npy', '--property_path', properties, '--scene_relation_path', relations, '--object_manifest', manifest, '--render_img', '--compile_video', '--output_h5']
            if not args.without_relation:
                command += ['--enable_relation', '--relation_full_checkpoint', external(runtime['relation_checkpoint'])]
            run('simulation', command, [work/'simulation/output.mp4', work/'simulation/validation.json'])
            report.update(status='baseline_forward_complete' if args.without_relation else 'forward_completed_requires_review', complete_inference=False, learned_relation_verified=False)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        run_file.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
