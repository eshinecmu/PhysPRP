"""Fit source pinhole camera and render native Gaussians without asset optimization."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np


def fit_camera(points, width, height):
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError('Expected pointmap [H,W,3]')
    h, w, _ = points.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack(((xx + .5) * width / w - .5, (yy + .5) * height / h - .5), -1)
    valid = np.isfinite(points).all(-1) & (points[..., 2] > 1e-6)
    xyz, uv = points[valid].astype(float), pixels[valid]
    if len(xyz) < 100:
        raise ValueError('Insufficient finite positive-depth source points')
    coefficients, residuals = [], []
    for axis in range(2):
        design = np.column_stack((xyz[:, axis] / xyz[:, 2], np.ones(len(xyz))))
        fitted, _, rank, _ = np.linalg.lstsq(design, uv[:, axis], rcond=None)
        if rank != 2 or abs(fitted[0]) < 1e-6:
            raise ValueError('Degenerate camera fit')
        coefficients.append(fitted.tolist())
        residuals.append(design @ fitted - uv[:, axis])
    error = np.linalg.norm(np.stack(residuals, axis=1), axis=1)
    return dict(signed_fx_cx=coefficients[0], signed_fy_cy=coefficients[1],
                point_count=len(xyz), rms_pixels=float(np.sqrt(np.mean(error ** 2))),
                p95_pixels=float(np.percentile(error, 95)))


def resolve_ground(points, mask, config):
    """Resolve an explicitly requested ground plane in MPM coordinates."""
    config = json.loads(json.dumps(config))
    if not config.get('ground_from_pointmap', False):
        return config, None
    if mask.shape != points.shape[:2]:
        raise ValueError('Floor mask must match pointmap resolution')
    valid = mask & np.isfinite(points).all(-1) & (points[..., 2] > 1e-6)
    samples = points[valid].astype(float)
    if len(samples) < 100:
        raise ValueError('Insufficient finite positive-depth floor points')
    center = samples.mean(0)
    _, singular, vh = np.linalg.svd(samples - center, full_matrices=False)
    if singular[1] <= max(singular[0], 1.) * 1e-8 or singular[2] >= singular[1] * .5:
        raise ValueError('Floor points are degenerate or do not support a plane')
    normal = vh[-1]
    if normal[1] < 0:
        normal = -normal
    if normal[1] <= 1e-6:
        raise ValueError('Floor normal cannot be oriented toward camera up')
    scale = float(config['scene_scale'])
    origin = np.asarray(config['mpm_origin'], dtype=float)
    if not np.isfinite(scale) or scale <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Require finite positive scene_scale and finite mpm_origin[3]')
    residuals = (samples - center) @ normal
    plane = dict(type='surface_collider', point=(center * scale + origin).tolist(),
                 normal=normal.tolist(), surface='separate', friction=.3, start_time=0., end_time=1000.)
    boundaries = config.get('boundary_conditions', [])
    previous = [bc for bc in boundaries if bc.get('type') == 'surface_collider']
    if len(previous) > 1:
        raise ValueError('Ambiguous ground replacement: more than one surface_collider configured')
    config['boundary_conditions'] = [bc for bc in boundaries if bc.get('type') != 'surface_collider'] + [plane]
    provenance = dict(source='single_image_pointmap_plane_estimate', is_measured=False,
                      point_count=len(samples), source_center=center.tolist(), normal=normal.tolist(),
                      source_rms=float(np.sqrt(np.mean(residuals ** 2))),
                      source_p95_abs=float(np.percentile(abs(residuals), 95)),
                      metric_rms=float(np.sqrt(np.mean(residuals ** 2))) * scale,
                      metric_p95_abs=float(np.percentile(abs(residuals), 95)) * scale,
                      scene_scale=scale, mpm_origin=origin.tolist(), collider=plane,
                      replaced_colliders=previous, friction_source='explicit assumption 0.3')
    if config.get('gravity_from_ground', False):
        provenance['previous_gravity'] = config.get('g')
        config['g'] = (-9.81 * normal).tolist()
        provenance['gravity_source'] = 'explicit gravity_from_ground: standard magnitude 9.81 opposite estimated floor normal'
    config['ground_provenance'] = provenance
    return config, provenance


def self_check():
    yy, xx = np.mgrid[:32, :48]
    z = 2 + xx / 100
    points = np.stack((-(xx - 23) * z / 40, -(yy - 15) * z / 41, z), -1)
    fit = fit_camera(points, 48, 32)
    assert np.allclose(fit['signed_fx_cx'], [-40, 23])
    assert np.allclose(fit['signed_fy_cy'], [-41, 15])
    assert fit['rms_pixels'] < 1e-10
    assert np.allclose(fit_camera(points * .4, 48, 32)['signed_fx_cx'], [-40, 23])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path)
    parser.add_argument('--resolved-config', type=Path)
    parser.add_argument('--camera-only', action='store_true', help='CPU diagnostic; does not pass rendered-asset validation')
    args = parser.parse_args()
    if bool(args.scene_config) != bool(args.resolved_config):
        parser.error('--scene-config and --resolved-config must be provided together')
    self_check()
    from PIL import Image
    reference = np.asarray(Image.open(args.image).convert('RGB'))
    height, width = reference.shape[:2]
    poses = sorted(args.scene_root.rglob('*_source_pose.npz'))
    pose_path = None
    for candidate in poses:
        with np.load(candidate, allow_pickle=False) as pose:
            if 'pointmap' in pose:
                points = pose['pointmap']
                pose_path = candidate
                break
    if pose_path is None:
        raise ValueError('No source pose containing pointmap; reconstruct with pose export enabled')
    fitted = fit_camera(points, width, height)
    fx, cx = fitted['signed_fx_cx']; fy, cy = fitted['signed_fy_cy']
    rotation = np.diag([np.sign(fx), np.sign(fy), 1.]).astype(np.float32)
    if np.linalg.det(rotation) <= 0:
        raise ValueError('Pinhole fit implies reflection; source coordinate convention needs explicit conversion')
    camera_record = dict(id=0, img_name=args.image.stem, width=width, height=height,
                         position=[0., 0., 0.], rotation=rotation.tolist(),
                         fx=abs(fx), fy=abs(fy), cx=cx, cy=cy)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(source_image=str(args.image), source_pose=str(pose_path), camera=fitted,
                  status='camera_only', fitted_asset_parameters=False,
                  note='Single-image pointmap fit; no video frames, scale fit, translation fit or material updates.')
    report_path = args.output / 'camera_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    if fitted['p95_pixels'] > 2:
        raise ValueError('Source pinhole projection residual exceeds 2 pixels; see camera_report.json')
    args.model_path.mkdir(parents=True, exist_ok=True)
    (args.model_path / 'cameras.json').write_text(json.dumps([camera_record], indent=2))
    if args.scene_config:
        config = json.loads(args.scene_config.read_text(encoding='utf-8'))
        ground_report = None
        if config.get('ground_from_pointmap', False):
            masks = sorted(args.scene_root.rglob('the_floor*.png'))
            masks = [candidate for candidate in masks if candidate.parent.name == 'segemented_obj']
            if len(masks) != 1:
                raise ValueError('Expected exactly one segmented the_floor RGBA mask')
            floor_image = Image.open(masks[0])
            if 'A' not in floor_image.getbands():
                raise ValueError('Floor segmentation must contain an alpha mask')
            mask = np.asarray(floor_image.getchannel('A').resize((points.shape[1], points.shape[0]), Image.Resampling.NEAREST)) > 127
            config, ground_report = resolve_ground(points, mask, config)
            ground_report.update(source_pose=str(pose_path), source_mask=str(masks[0]))
        args.resolved_config.parent.mkdir(parents=True, exist_ok=True)
        args.resolved_config.write_text(json.dumps(config, indent=2))
        report['ground'] = ground_report
        report['resolved_config'] = str(args.resolved_config)
        report_path.write_text(json.dumps(report, indent=2))
    if args.camera_only:
        return
    gs_path = Path(os.environ['PHYS_GS_ROOT'])
    if not (gs_path / 'scene/gaussian_model.py').is_file():
        raise ValueError('PHYS_GS_ROOT must name the declared Gaussian-splatting source')
    sys.path.insert(0, str(gs_path))
    gs_utils = ModuleType('utils'); gs_utils.__path__ = [str(gs_path / 'utils')]
    sys.modules['utils'] = gs_utils
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    camera = Camera(colmap_id=0, R=rotation, T=np.zeros(3),
                    FoVx=2 * math.atan(width / (2 * abs(fx))),
                    FoVy=2 * math.atan(height / (2 * abs(fy))),
                    image=torch.zeros((3, height, width)), gt_alpha_mask=None,
                    image_name=args.image.stem, uid=0)
    camera.projection_matrix[2, 0] = (2 * cx + 1) / width - 1
    camera.projection_matrix[2, 1] = (2 * cy + 1) / height - 1
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    model = GaussianModel(3)
    model.load_ply(str(args.model_path / 'point_cloud/iteration_60000/point_cloud.ply'))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device='cuda')
    with torch.no_grad():
        rgb = render(camera, model, pipe, background)['render']
        alpha = render(camera, model, pipe, background, override_color=torch.ones_like(model.get_xyz))['render'][0]
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise ValueError('Nonfinite first-frame render')
    rgb = rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    alpha = alpha.clamp(0, 1).cpu().numpy()
    image = np.round(rgb * 255).astype(np.uint8)
    overlay = np.round(.5 * reference + .5 * image).astype(np.uint8)
    Image.fromarray(image).save(args.output / 'first_frame.png')
    Image.fromarray(overlay).save(args.output / 'overlay.png')
    Image.fromarray(np.concatenate([reference, image, overlay], axis=1)).save(args.output / 'comparison.jpg')
    Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(args.output / 'alpha.png')
    report.update(status='rendered_requires_visual_review', gaussians=len(model.get_xyz),
                  foreground_pixels=int((alpha > .5).sum()), projection_alignment_accepted=False,
                  note=report['note'] + ' No segmentation overlap metric asserted; inspect overlay for actual asset alignment.')
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
