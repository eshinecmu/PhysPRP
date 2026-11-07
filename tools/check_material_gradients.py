"""Real-scene MPM gradient/checkpoint probes; run each probe in a fresh process.

Short probes compare direct/checkpoint terminal states and raw material gradients.
Window probes differentiate the actual two-frame motion loss and check central
finite differences on the same frozen CFL-safe time grid. Capacity probes retain
the complete original WorldParticle graph and differentiate motion loss alone.
"""
import argparse
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_material_video import (STATE_KEYS, episode_forward,
    load_observed_episode, material_rollout_block)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MemoryPeak:
    """Process-wide device memory includes Warp allocations outside Torch."""
    def __init__(self):
        self.stop = threading.Event()
        self.samples, self.process_peak, self.error = 0, 0, None
        self.selector = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not self.selector or ',' in self.selector:
            raise ValueError('Set CUDA_VISIBLE_DEVICES to one physical GPU index or UUID')
        self.pids = {os.getpid()}
        status = Path('/proc/self/status')
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith('NSpid:'):
                    self.pids.update(map(int, line.split()[1:]))
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = (pynvml.nvmlDeviceGetHandleByIndex(int(self.selector))
                           if self.selector.isdigit() else
                           pynvml.nvmlDeviceGetHandleByUUID(self.selector.encode()))
            self.backend = 'pynvml'
        except Exception:
            self.nvml = None
            self.backend = 'nvidia-smi'
        # Capture before wp.init or any CUDA allocation. Container-local PIDs
        # need not match the host PIDs exposed by NVML/nvidia-smi.
        self.before = self.processes()
        self.result = dict(selected_physical_gpu=self.selector,
            before_cuda_host_pids=sorted(self.before), namespace_pids=sorted(self.pids))

    def processes(self):
        if self.nvml:
            try:
                return {int(p.pid): (int(p.usedGpuMemory)
                        if 0 <= p.usedGpuMemory < 2**60 else None)
                        for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)}
            except Exception:
                self.nvml, self.backend = None, 'nvidia-smi'
        output = subprocess.check_output(['nvidia-smi', '--id='+self.selector,
            '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=3)
        return {int(pid.strip()): (int(memory.strip())*1024**2
                if memory.strip().isdigit() else None)
                for row in output.splitlines() if ',' in row
                for pid, memory in [row.split(',', 1)]}

    def sample(self):
        try:
            memory = self.processes().get(self.host_pid)
            if memory is not None:
                self.samples += 1
                self.process_peak = max(self.process_peak, memory)
        except Exception as exc:
            self.error = repr(exc)

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        current = self.processes()
        added = set(current)-set(self.before)
        self.result.update(at_probe_start_host_pids=sorted(current),
                           newly_seen_host_pids=sorted(added))
        if len(added) != 1:
            raise RuntimeError(f'Cannot uniquely map own host GPU PID: added={sorted(added)}')
        self.host_pid = added.pop()
        self.result.update(host_pid=self.host_pid, pid_mapping=(
            'new selected-GPU PID also matches os.getpid/NSpid'
            if self.host_pid in self.pids else
            'unique new host PID on selected GPU since pre-CUDA snapshot'))
        self.sample()
        self.baseline = self.process_peak
        def poll():
            while not self.stop.wait(.5):
                self.sample()
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        self.started = time.monotonic()
        return self

    def __exit__(self, *_):
        torch.cuda.synchronize()
        self.sample()
        self.stop.set()
        self.thread.join(4)
        self.result.update(backend=self.backend, interval_seconds=.5,
            process_baseline_bytes=self.baseline, process_peak_bytes=self.process_peak,
            process_peak_increase_bytes=self.process_peak-self.baseline,
            process_samples=self.samples, sampling_error=self.error,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-self.started)


def summarize(tensor):
    if tensor is None:
        return dict(present=False, finite=False, norm=0.)
    tensor = tensor.detach()
    if not torch.isfinite(tensor).all():
        return dict(present=True, finite=False, norm=None, sum=None, max_abs=None)
    return dict(present=True, finite=bool(torch.isfinite(tensor).all()),
                norm=float(tensor.norm()), sum=float(tensor.sum()),
                max_abs=float(tensor.abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', choices=('short', 'window', 'capacity'), required=True)
    parser.add_argument('--case')
    parser.add_argument('--checkpoint', choices=('off', 'reentrant'), default='reentrant')
    parser.add_argument('--grad-mode', choices=('full', 'tail'), default='full')
    parser.add_argument('--checkpoint-steps', type=int, default=64)
    parser.add_argument('--short-steps', type=int, default=128)
    parser.add_argument('--fd-h', type=float, nargs='+', default=[.001, .003, .01])
    parser.add_argument('--no-fd', action='store_true')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.pt').exists():
        parser.error('Use a new output name; existing probe artifacts are preserved')
    if args.checkpoint_steps <= 0 or args.short_steps <= 0 or any(h <= 0 for h in args.fd_h):
        parser.error('Probe step counts and finite-difference shifts must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory = MemoryPeak()
    import warp as wp
    wp.config.kernel_cache_dir = str(args.output.parent / 'cache' / args.output.stem / 'warp')
    wp.init()
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)
    from omegaconf import OmegaConf
    from PhysGaussian.worldparticle_grid_correction import MaterialTransformer
    from train_relation_full import make_physcore_corrector, make_physcore_engine
    from physcore.particle_flow.episode_runtime import _piecewise_catmull_rom_window
    settings = argparse.Namespace(prefix_frames=4, target_frames=2, dt=5e-5,
        bptt=40, w_track=1., w_depth=1., w_prior=0., grad_mode=args.grad_mode,
        checkpoint=args.checkpoint, checkpoint_steps=args.checkpoint_steps)
    defaults = dict(short='physcore__double_stretch_plastic',
                    window='phystwin__single_push_rope_1',
                    capacity='physcore__double_stretch_bear_1')
    name = args.case or defaults[args.probe]
    record = next(r for r in json.loads(args.manifest.read_text()) if r['case'] == name)
    case = load_observed_episode(record, settings, torch.device('cpu'))
    cfg = OmegaConf.create(dict(model=dict(kind='worldparticle',
        source_root=str(ROOT/'third_party/WorldParticle'), width=32,
        merge_iterations=3, decoder_layers=2), correction=dict(max_delta_v=.002)))
    wrapper = make_physcore_corrector(cfg, device)
    boundary, normals = wrapper._boundaries(case['physics'], case['state']['x'][0].to(device))
    case['boundary'], case['normals'] = boundary.cpu(), normals.cpu()
    model = MaterialTransformer(wrapper.backbone, case['motion'].shape[-1]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location='cpu',
                         weights_only=False)['model_state_dict'], strict=True)
    provenance = wrapper.provenance()
    provenance.update(
        pipeline='Original CConv -> k-partite merging -> RoPE -> SuperParticleDecoder -> adapted 2-channel material head',
        physical_features=f'x0/v0 geometry plus {case["motion"].shape[-1]} observed-prefix motion/controller features and log(E/2000), nu, log(rho/1000), projected to 32 channels; no F/C labels or material embedding',
        correction='Per-particle log(E) and nu refinement around GaussianProperty priors; density fixed; no dynamics or P2G residual',
        predictor='MaterialTransformer predicts material once; original PhysCoRe MPM propagates x/v/F/C through the observed window')
    del wrapper
    model.train(args.probe == 'capacity')
    model.backbone.apply_radius_override(radius_scale=1.5,
        particle_radius=case['physics']['volume']**(1/3)/2)
    prior = {k: v.to(device) for k, v in case['prior'].items()}
    with torch.no_grad():
        material = model(case['model_x'].to(device), case['model_v'].to(device),
            case['motion'].to(device), prior, boundary, normals)
    # One time grid also covers the largest positive finite-difference shift.
    hmax = max(args.fd_h) if args.probe == 'window' and not args.no_fd else .01
    E = (material['log_E'] + hmax).exp()
    nu = material['nu']
    speed = (E*(1-nu)/((1+nu)*(1-2*nu)*prior['density'])).sqrt().max()
    frame_steps = math.ceil(case['frame_dt']/min(settings.dt, .35/(32*float(speed))))
    controls = _piecewise_catmull_rom_window(case['controller'].to(device),
                                            0, case['end']-1, frame_steps)
    result = dict(status='running', probe=args.probe, case=name,
        particles=case['metadata']['particles'], seed=args.seed,
        checkpoint=args.checkpoint, grad_mode=args.grad_mode,
        checkpoint_steps=args.checkpoint_steps, fixed_frame_steps=frame_steps,
        dt=case['frame_dt']/frame_steps, full_window_steps=(case['end']-1)*frame_steps,
        simulated_seconds=(case['end']-1)*case['frame_dt'],
        weights_sha256=sha256(args.weights), manifest_sha256=sha256(args.manifest),
        runner_sha256=sha256(ROOT/'tools/train_material_video.py'),
        diagnostic_sha256=sha256(__file__), case_metadata=case['metadata'],
        controls_sha256=hashlib.sha256(controls.cpu().numpy().tobytes()).hexdigest(),
        provenance=provenance, runtime=dict(python=sys.executable, torch=torch.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name()))
    # ponytail: independent CLI workers isolate CUDA/Warp caches and OOM recovery;
    # a sampled process peak has 0.5 s resolution, so keep Torch peaks as well.
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        with memory:
            if args.probe == 'short':
                if args.short_steps > frame_steps:
                    raise ValueError('Short terminal probe must fit the first observed frame')
                physics = dict(case['physics'], dt=case['frame_dt']/frame_steps)
                engine = make_physcore_engine(dict(physcore_adapter=physics),
                    OmegaConf.create(case['rollout']), device, material_only=True)
                engine.enable_corrector(False)
                advance = partial(material_rollout_block, engine)
                leaves = {k: v.to(device).detach().clone().requires_grad_()
                          for k, v in case['state'].items()}
                leaves.update({k: v.detach().clone().requires_grad_() for k, v in material.items()})
                state = tuple(leaves[k] for k in STATE_KEYS)
                for start in range(0, args.short_steps, args.checkpoint_steps):
                    end = min(start+args.checkpoint_steps, args.short_steps)
                    tensors = (*state, leaves['log_E'], leaves['nu'],
                               controls[start:end+1][None], controls.new_ones(1))
                    state = (checkpoint(advance, *tensors, use_reentrant=True)
                             if args.checkpoint == 'reentrant' else advance(*tensors))
                scalar = sum(scale*value.square().mean() for scale, value in
                             zip((1., 1e-4, 1e-5, 1e-6), state))
                scalar.backward()
                vectors = dict(states={k: v.detach().cpu() for k, v in zip(STATE_KEYS, state)},
                               gradients={k: v.grad.detach().cpu() for k, v in leaves.items()})
                torch.save(vectors, args.output.with_suffix('.pt'))
                result.update(terminal_scalar=float(scalar.detach()), short_steps=args.short_steps,
                    simulated_seconds=args.short_steps*physics['dt'],
                    objective='terminal state scalar; not video motion supervision',
                    gradients={k: summarize(v.grad) for k, v in leaves.items()})
            else:
                leaves = {k: v.detach().clone().requires_grad_() for k, v in material.items()}
                loss, metrics = episode_forward(model, case, settings, training=True,
                    material_override=None if args.probe == 'capacity' else leaves,
                    fixed_frame_steps=frame_steps)
                loss.backward()
                result['metrics'] = metrics
                if args.probe == 'capacity':
                    head_ids = {id(p) for p in model.backbone.output_network[-1].parameters()}
                    groups = dict(head=[p.grad for p in model.parameters() if id(p) in head_ids],
                        original_core=[p.grad for p in model.backbone.parameters() if id(p) not in head_ids])
                    result['gradients'] = {k: summarize(torch.cat([g.detach().flatten() for g in values if g is not None]))
                                           for k, values in groups.items()}
                else:
                    result['gradients'] = {k: summarize(v.grad) for k, v in leaves.items()}
                del loss
        result['memory'] = memory.result
        if not memory.result['process_samples']:
            raise RuntimeError('No process GPU-memory samples; Warp memory is unverified')
        if any(not g['finite'] or not g['norm'] for g in result['gradients'].values()):
            raise RuntimeError('Missing, nonfinite or zero motion/terminal gradient')
        if args.probe == 'window' and not args.no_fd:
            result['finite_differences'] = []
            for h in args.fd_h:
                losses = []
                for sign in (-1., 1.):
                    shifted = {k: v.detach() + (sign*h if k == 'log_E' else 0.)
                               for k, v in material.items()}
                    loss, _ = episode_forward(model, case, settings, training=False,
                        material_override=shifted, fixed_frame_steps=frame_steps)
                    losses.append(float(loss))
                fd = (losses[1]-losses[0])/(2*h)
                ad = result['gradients']['log_E']['sum']
                result['finite_differences'].append(dict(h=h, minus=losses[0], plus=losses[1],
                    derivative=fd, autodiff=ad, relative_error=abs(fd-ad)/max(abs(fd), abs(ad), 1e-12)))
        result['status'] = 'ok'
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='oom', error=str(exc), memory=getattr(memory, 'result', None))
    except Exception as exc:
        result.update(status='error', error=repr(exc), memory=getattr(memory, 'result', None))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(dict(status=result['status'], output=str(args.output),
                              memory=result.get('memory'))), flush=True)


if __name__ == '__main__':
    main()
