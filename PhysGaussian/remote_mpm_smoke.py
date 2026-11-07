"""Real Warp integration checks using a controlled teacher, not a real GT dataset."""

import argparse
import io
import os
from pathlib import Path
import sys

import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_relation_full import (MATERIAL_VOCAB, SAMPLE_METADATA, build_trajectory_samples,
                                 make_baseline_solver, load_gt)
from PhysGaussian.relation_mpm import PostStepRelationMPM, RELATION_TYPES, load_relation_checkpoint
from PhysGaussian.synthetic_two_object import make_scene


def controlled_trajectory(device, dt):
    """Keep full elastic states from two substeps with a known post-step impulse."""
    scene = make_scene()
    n = len(scene["x"])
    data = {key: torch.from_numpy(scene[key]).clone() for key in
            ("volume", "mass", "E", "nu", "density", "object_id", "material_id")}
    # These semantic names and all teacher values are confined to this integration check.
    data["material_id"] = torch.where(data["material_id"] == 0,
                                      MATERIAL_VOCAB.index("foam"), MATERIAL_VOCAB.index("plastic"))
    data.update(particle_id=torch.arange(n).expand(3, -1).clone(),
                selection=torch.zeros(n, dtype=torch.int32), material_vocab=list(MATERIAL_VOCAB),
                object_relations={(0, 1): RELATION_TYPES["attach"]},
                mpm_config={"material": "jelly", "n_grid": 32, "grid_lim": 2.,
                            "g": [0., -1., 0.], "rpic_damping": 0., "grid_v_damping_scale": 1.,
                            "substep_dt": dt, "boundary_conditions": [{"type": "surface_collider",
                                "point": [0., .5, 0.], "normal": [0., 1., 0.], "surface": "separate",
                                "friction": 0., "start_time": 0., "end_time": 10.}]})
    data["selection"][0] = 1
    initial_v = torch.from_numpy(scene["v"]).clone()
    initial_v[0] = 0
    states = {"x": [torch.from_numpy(scene["x"]).clone()], "v": [initial_v],
              "F_trial": [torch.from_numpy(scene["F"]).clone()], "C": [torch.zeros(n, 3, 3)]}
    known_delta = torch.tensor([.03, -.02, .01], device=device).expand(n, -1).clone()
    known_delta[data["selection"] != 0] = 0

    def teacher(solver, step_dt):
        x, v = solver.export_particle_x_to_torch(), solver.export_particle_v_to_torch()
        solver.import_particle_v_from_torch((v + known_delta).contiguous(), device=str(device))
        solver.import_particle_x_from_torch((x + step_dt * known_delta).contiguous(), device=str(device))

    for index in range(2):
        data.update({key: torch.stack(value) for key, value in states.items()})
        data["time"] = torch.arange(index + 1, dtype=torch.float64) * dt
        solver = make_baseline_solver(data, index, device)
        # Assert the adapter imported the full true state before the first kernel runs.
        torch.testing.assert_close(solver.export_particle_v_to_torch(), data["v"][index].to(device), rtol=0, atol=0)
        torch.testing.assert_close(wp.to_torch(solver.mpm_state.particle_F_trial), data["F_trial"][index].to(device), rtol=0, atol=0)
        torch.testing.assert_close(wp.to_torch(solver.mpm_state.particle_C), data["C"][index].to(device), rtol=0, atol=0)
        torch.testing.assert_close(wp.to_torch(solver.mpm_state.particle_mass), data["mass"].to(device), rtol=1e-6, atol=0)
        assert solver.relation_step is None and getattr(solver, "grid_correction_step", None) is None
        solver.relation_step = teacher
        solver.p2g2p(index, dt, device=str(device))
        states["x"].append(solver.export_particle_x_to_torch().detach().cpu().clone())
        states["v"].append(solver.export_particle_v_to_torch().detach().cpu().clone())
        states["F_trial"].append(wp.to_torch(solver.mpm_state.particle_F_trial).detach().cpu().clone())
        states["C"].append(wp.to_torch(solver.mpm_state.particle_C).detach().cpu().clone())
    data.update({key: torch.stack(value) for key, value in states.items()})
    data["time"] = torch.arange(3, dtype=torch.float64) * dt
    return data, known_delta


def physcore_checks(trajectory, device):
    """Check the actual upstream grid corrector and its differentiable MPM path."""
    import copy
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from prepare_uniphy_jelly import to_physcore_episode
    from train_relation_full import make_physcore_engine, physcore_state, physcore_advance

    if device.type != "cuda":
        raise ValueError("The upstream spconv regression check requires a CUDA device")
    data = torch.load(trajectory, weights_only=True, map_location="cpu")
    original = {key: data[key].clone() for key in ("x", "v", "F_trial", "C")}
    original_config = copy.deepcopy(data["mpm_config"])
    episode = to_physcore_episode(data)
    physics = episode["physcore_adapter"]
    rotation = data["x"].new_tensor(physics["rotation"])
    translation = data["x"].new_tensor(physics["translation"])
    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.))
    torch.testing.assert_close((episode["particle_coords"] - translation) @ rotation, data["x"], rtol=0, atol=1e-7)
    torch.testing.assert_close(episode["particle_velocities"] @ rotation, data["v"], rtol=0, atol=0)
    for source, target in (("F_trial", "particle_deformation_gradients"), ("C", "particle_affine_velocity_matrices")):
        torch.testing.assert_close(rotation.T @ episode[target] @ rotation, data[source], rtol=0, atol=0)
    torch.testing.assert_close(rotation.T @ rotation.new_tensor(physics["gravity"]), rotation.new_tensor(original_config["g"]))
    assert len(physics["boundary_conditions"]) == len(original_config["boundary_conditions"]) == 6
    for before, after in zip(original_config["boundary_conditions"], physics["boundary_conditions"]):
        torch.testing.assert_close((rotation.new_tensor(after["point"]) - translation) @ rotation,
                                   rotation.new_tensor(before["point"]), rtol=0, atol=1e-7)
        torch.testing.assert_close(rotation.new_tensor(after["normal"]) @ rotation,
                                   rotation.new_tensor(before["normal"]), rtol=0, atol=0)
    for key in original:
        assert torch.equal(original[key], data[key]), f"Adapter mutated {key}"
    assert data["mpm_config"] == original_config
    assert "observation_data" not in episode and "camera" not in episode
    assert torch.equal(episode["tracked_visible_particle_indices"][100], torch.arange(len(data["mass"])))
    torch.testing.assert_close(episode["particle_material_params"]["log_E"], data["E"].log())

    vendor = Path(__file__).resolve().parents[1] / "third_party/PhysCoRe"
    sys.path.insert(0, str(vendor))
    import physcore.model_RfD as upstream
    from physcore.particle_flow.rfd_runtime import tracked_l2
    assert Path(upstream.__file__).resolve().is_relative_to(vendor.resolve()), "Wrong PhysCoRe source imported"
    torch.manual_seed(7)
    corrector = upstream.GridVelocityCorrector(in_channels=19, cond_dim=16,
        base_channels=8, levels=1, max_delta_v=.05).to(device).eval()
    cfg = OmegaConf.create(dict(clip_bound=.5, elasticity_model="CorotatedElasticity",
                               plasticity_model="IdentityPlasticity", use_explicit_rigid_contact=False))
    engine = make_physcore_engine(episode, cfg, device, corrector, cadence=1)
    state = physcore_state(episode, 100, device)
    material = {key: value.unsqueeze(0).to(device) for key, value in episode["particle_material_params"].items()}
    solver, _ = engine._make_batched_solver(state["x"])
    torch.testing.assert_close(solver.model.vol, state["x"].new_tensor([physics["volume"]]), rtol=0, atol=0)
    torch.testing.assert_close(solver.model.p_mass, state["x"].new_tensor([physics["mass"]]), rtol=1e-6, atol=0)
    assert len(solver.model.post_grid_process) == 7, "Expected all six surfaces followed by the RfD hook"
    with torch.no_grad():
        engine.enable_corrector(False)
        _, baseline = physcore_advance(engine, state, material, 2)
        engine.enable_corrector(True)
        _, zero = physcore_advance(engine, state, material, 2)
    for key in state:
        # C amplifies float32 scatter roundoff by 4/dx; use a two-substep absolute budget near zero.
        torch.testing.assert_close(zero[key], baseline[key], rtol=1e-5, atol=5e-5 if key == "C" else 1e-6)

    # A constant nonzero head gives a clean gradient probe independent of FiLM's segment-phase channels.
    with torch.no_grad():
        corrector.head.bias.copy_(state["x"].new_tensor([.15, -.10, .20]))
    indices = torch.arange(state["x"].shape[1], device=device)
    gt = episode["particle_coords"][102].to(device)

    def observed_loss():
        out, _ = physcore_advance(engine, state, material, 2)
        return tracked_l2(gt, out["predicted_positions"][-1, 0], indices)

    corrector.zero_grad(set_to_none=True)
    loss = observed_loss()
    loss.backward()
    grad = corrector.head.bias.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0, "MPM did not backpropagate into the official RfD head"
    weight_grad = corrector.head.weight.grad
    assert weight_grad is not None and torch.isfinite(weight_grad).all() and weight_grad.norm() > 0
    analytic = float(grad[2])
    with torch.no_grad():
        center = corrector.head.bias[2].item()
        epsilon = .02
        corrector.head.bias[2] = center + epsilon
        plus = observed_loss().item()
        corrector.head.bias[2] = center - epsilon
        minus = observed_loss().item()
        corrector.head.bias[2] = center
    finite_difference = (plus - minus) / (2 * epsilon)
    assert abs(analytic - finite_difference) <= .1 * max(abs(analytic), abs(finite_difference)) + 1e-7, (
        f"RfD/MPM derivative mismatch: autograd={analytic}, finite_difference={finite_difference}")

    # Spatial grid residuals must influence strain as well as translation through the same G2P.
    with torch.no_grad():
        corrector.head.weight.normal_(std=.01)
        _, spatial = physcore_advance(engine, state, material, 2)
    changes = {key: (spatial[key] - baseline[key]).abs().max().item() for key in state}
    assert changes["x"] > 1e-8 and changes["v"] > 1e-5
    assert changes["C"] > 1e-4 and changes["F"] > 1e-7, f"Grid residual failed to update the full state: {changes}"
    with torch.no_grad():
        corrector.head.weight.zero_()
        _, continuous = physcore_advance(engine, state, material, 4)
        _, middle = physcore_advance(engine, state, material, 2)
        carried = {key: value.detach() for key, value in middle.items()}
        _, segmented = physcore_advance(engine, carried, material, 2)
    for key in state:
        torch.testing.assert_close(segmented[key], continuous[key], rtol=2e-5, atol=1e-4 if key == "C" else 2e-6)
    print(f"official PhysCoRe checks passed: adapter=invertible, volume=GT, six_surfaces=restored, "
          f"zero_head=baseline, head_grad={grad.norm().item():.7g}, autograd={analytic:.7g}, fd={finite_difference:.7g}, "
          f"full_state_delta={changes}, TBPTT_state_carry=consistent; numerical integration check, not MfM training")


def worldparticle_split_checks(cfg, cases, device):
    """Prove actual upstream Transformer modules run and receive gradients through the MPM hook."""
    import json
    from prepare_uniphy_jelly import to_physcore_episode
    from train_relation_full import make_physcore_corrector, make_physcore_engine, physcore_state, physcore_advance, physcore_gt_step
    corrector = make_physcore_corrector(cfg, device).eval()
    core = corrector.backbone
    modules = dict(CConv=core.local_feature_extractor.conv0_fluid,
                   boundary_CConv=core.local_feature_extractor.conv0_obstacle,
                   merging=core.super_particle_cross_attn_feat, RoPE=core.particle_rope,
                   decoder=core.super_particle_decoder)
    calls = {key: 0 for key in modules}
    token_counts, gradients, derivatives = [], {}, []
    def record(key):
        def hook(module, inputs, output):
            calls[key] += 1
            if key == "merging":
                token_counts.append(int(output[0].shape[1]))
        return hook
    handles = [module.register_forward_hook(record(key)) for key, module in modules.items()]
    for case in cases['train'][:2]:
        episode = to_physcore_episode(torch.load(case['path'], map_location='cpu', weights_only=True))
        engine = make_physcore_engine(episode, cfg.rollout, device, corrector, cadence=1)
        material = {key: value.unsqueeze(0).to(device) for key, value in episode['particle_material_params'].items()}
        first = min(100, case['stop']-2)
        state = physcore_state(episode, first, device)
        with torch.no_grad():
            corrector.head.weight.zero_(); corrector.head.bias.zero_()
            engine.enable_corrector(False)
            _, baseline = physcore_advance(engine, state, material, 2)
            engine.enable_corrector(True)
            _, zero = physcore_advance(engine, state, material, 2)
            for key in state:
                torch.testing.assert_close(zero[key], baseline[key], rtol=1e-5, atol=5e-5 if key == 'C' else 1e-6)
            scale = case['physics']['dt'] * corrector.max_delta_v / core.output_scale
            corrector.head.weight.normal_(0., scale*.01)
        corrector.zero_grad(set_to_none=True)
        fields = ('particle_coords','particle_velocities','particle_deformation_gradients','particle_affine_velocity_matrices')
        snapshot = {key: episode[key].clone() for key in fields}
        out, result = physcore_gt_step(engine, episode, first, material, device)
        weights = torch.linspace(.5, 1.5, state['x'].shape[1], device=device)[:, None]
        objective = (result['v'][0] * weights).mean()
        objective.backward()
        for prefix in ('local_feature_extractor.conv0_fluid', 'local_feature_extractor.conv0_obstacle',
                       'super_particle_cross_attn_feat', 'super_particle_decoder'):
            values = [p.grad for name, p in core.named_parameters() if name.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(g).all() for g in values), prefix
            gradients[prefix] = gradients.get(prefix, 0.) + sum(float(g.norm()) for g in values)
        analytic = float(corrector.head.bias.grad[0])
        with torch.no_grad():
            center, epsilon = float(corrector.head.bias[0]), scale*.02
            corrector.head.bias[0] = center + epsilon
            _, plus = physcore_gt_step(engine, episode, first, material, device)
            corrector.head.bias[0] = center - epsilon
            _, minus = physcore_gt_step(engine, episode, first, material, device)
            corrector.head.bias[0] = center
            fd = float(((plus['v'][0]*weights).mean()-(minus['v'][0]*weights).mean()) / (2*epsilon))
        assert abs(analytic-fd) <= .15*max(abs(analytic),abs(fd))+1e-6, (case['episode_id'], analytic, fd)
        derivatives.append(dict(episode_id=case['episode_id'], autograd=analytic, finite_difference=fd))
        for key in fields:
            assert torch.equal(episode[key], snapshot[key]), key
    for handle in handles:
        handle.remove()
    assert all(calls.values()) and all(value > 0 for value in gradients.values()), (calls, gradients)
    print('WORLD PARTICLE CHECKS PASSED', json.dumps(dict(calls=calls, token_counts=token_counts,
          gradient_norms=gradients, derivatives=derivatives, provenance=corrector.provenance(), heldout_rollouts=0)), flush=True)


def physcore_split_checks(config_path, device):
    """Exercise the same official corrector on two independent training episodes."""
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from prepare_uniphy_jelly import to_physcore_episode
    from train_relation_full import physcore_trajectory_cases, make_physcore_engine, physcore_state, physcore_advance, physcore_gt_step

    if device.type != "cuda":
        raise ValueError("The upstream spconv split check requires a CUDA device")
    cfg = OmegaConf.load(config_path)
    cases, multiple = physcore_trajectory_cases(OmegaConf.to_container(cfg.dataset, resolve=True))
    if not multiple or len(cases["train"]) < 2:
        raise ValueError("The split integration check requires at least two independent training trajectories")
    vendor = Path(__file__).resolve().parents[1] / "third_party/PhysCoRe"
    sys.path.insert(0, str(vendor))
    if cfg.get("model", {}).get("kind") == "worldparticle":
        return worldparticle_split_checks(cfg, cases, device)
    from physcore.model_RfD import GridVelocityCorrector
    from physcore.particle_flow.rfd_runtime import tracked_l2

    corrector = GridVelocityCorrector(in_channels=19, cond_dim=16, base_channels=8,
                                      levels=1, max_delta_v=.01).to(device).eval()
    corrector.zero_grad(set_to_none=True)
    contributions = []
    # Validation/test are only checked for split leakage; no held-out rollout is used here.
    for case in cases["train"][:2]:
        if case["stop"] < 2:
            raise ValueError("The two-step integration check requires at least three saved states")
        episode = to_physcore_episode(torch.load(case["path"], map_location="cpu", weights_only=True))
        engine = make_physcore_engine(episode, cfg.rollout, device, corrector, cadence=1)
        assert engine._gvc_corrector is corrector
        if episode["physcore_adapter"].get("elasticity_model") == "NeoHookeanElasticity":
            assert type(engine.elasticity_model).__name__ == "NeoHookeanElasticity"
            E = episode["particle_material_params"]["log_E"].exp().to(device)
            nu = episode["particle_material_params"]["nu"].to(device)
            eye = torch.eye(3, device=device).expand(len(E), 3, 3)
            stress = engine.elasticity_model(1.1 * eye, E.log(), nu)
            expected = E / (2 * (1 + nu)) * (1.1 ** 2 - 1) + E * nu / ((1 + nu) * (1 - 2 * nu)) * (3 * torch.log(E.new_tensor(1.1)))
            torch.testing.assert_close(stress, expected[:, None, None] * eye, rtol=1e-5, atol=1e-4)
        first = min(100, case["stop"] - 2)
        state = physcore_state(episode, first, device)
        material = {key: value.unsqueeze(0).to(device) for key, value in episode["particle_material_params"].items()}
        with torch.no_grad():
            corrector.head.weight.zero_(); corrector.head.bias.zero_()
            engine.enable_corrector(False)
            _, baseline = physcore_advance(engine, state, material, 2)
            engine.enable_corrector(True)
            _, zero = physcore_advance(engine, state, material, 2)
            for key in state:
                torch.testing.assert_close(zero[key], baseline[key], rtol=1e-5, atol=5e-5 if key == "C" else 1e-6)
            corrector.head.bias.copy_(state["x"].new_tensor([.15, -.1, .2]))
        previous = torch.zeros_like(corrector.head.bias) if corrector.head.bias.grad is None else corrector.head.bias.grad.clone()
        out, _ = physcore_advance(engine, state, material, 2)
        loss = tracked_l2(episode["particle_coords"][first + 2].to(device), out["predicted_positions"][-1, 0],
                          torch.arange(state["x"].shape[1], device=device))
        loss.backward()
        contribution = corrector.head.bias.grad - previous
        assert torch.isfinite(contribution).all() and contribution.norm() > 0, case["episode_id"]
        with torch.no_grad():
            center, epsilon = float(corrector.head.bias[2]), .02
            corrector.head.bias[2] = center + epsilon
            plus, _ = physcore_advance(engine, state, material, 2)
            plus_loss = tracked_l2(episode["particle_coords"][first + 2].to(device), plus["predicted_positions"][-1, 0],
                                   torch.arange(state["x"].shape[1], device=device))
            corrector.head.bias[2] = center - epsilon
            minus, _ = physcore_advance(engine, state, material, 2)
            minus_loss = tracked_l2(episode["particle_coords"][first + 2].to(device), minus["predicted_positions"][-1, 0],
                                    torch.arange(state["x"].shape[1], device=device))
            corrector.head.bias[2] = center
            fd = float((plus_loss - minus_loss) / (2 * epsilon))
        analytic = float(contribution[2])
        assert abs(analytic - fd) <= .15 * max(abs(analytic), abs(fd)) + 1e-7, (case["episode_id"], analytic, fd)
        contributions.append(float(contribution.norm()))
        if cfg.train.get("mode") == "single-step":
            fields = ("particle_coords", "particle_velocities", "particle_deformation_gradients", "particle_affine_velocity_matrices")
            snapshot = {key: episode[key].clone() for key in fields}
            corrector.zero_grad(set_to_none=True)
            for frame in (first + 1, 0, first):
                out, _ = physcore_gt_step(engine, episode, frame, material, device)
                tracked_l2(episode["particle_coords"][frame + 1].to(device), out["predicted_positions"][0, 0],
                           torch.arange(state["x"].shape[1], device=device)).backward()
            assert corrector.head.bias.grad.norm() > 0
            for key in fields:
                assert torch.equal(episode[key], snapshot[key]), f"GT mutated: {key}"
    print(f"official PhysCoRe split checks passed: train_episodes={[case['episode_id'] for case in cases['train'][:2]]}, "
          f"zero_head=per_episode_baseline, shared_head_gradient_contributions={contributions}, heldout_rollouts=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:0 in an existing Warp environment")
    parser.add_argument("--trajectory", help="Also check original UniPhy jelly against explicitly compatible MLS transfers")
    parser.add_argument("--physcore", action="store_true", help="Check official PhysCoRe with --trajectory instead of the legacy post-step path")
    parser.add_argument("--physcore-splits", help="Check explicit episode splits and two train-episode gradients using a training YAML")
    args = parser.parse_args()
    device = torch.device(args.device)
    wp.config.kernel_cache_dir = os.environ.get("PHYSPRP_WARP_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "warp"))
    wp.init()
    if args.physcore_splits:
        if args.physcore or args.trajectory:
            parser.error("--physcore-splits is a separate check; do not combine with --physcore/--trajectory")
        physcore_split_checks(args.physcore_splits, device)
        return
    if args.physcore:
        if not args.trajectory:
            parser.error("--physcore requires --trajectory with complete solid states")
        physcore_checks(args.trajectory, device)
        return
    if args.trajectory:
        data = torch.load(args.trajectory, weights_only=True, map_location="cpu")
        for index in (100, 700, 900):
            legacy = make_baseline_solver(data, index, device)
            explicit = dict(data, mpm_config=dict(data["mpm_config"], transfer_scheme="gradient", grid_mass_epsilon=0.))
            compatible = dict(data, mpm_config=dict(data["mpm_config"], transfer_scheme="mls", grid_mass_epsilon=1e-7))
            old = build_trajectory_samples(explicit, device, [index])[0]
            legacy.p2g2p(index, data["mpm_config"]["substep_dt"], device=str(device))
            torch.testing.assert_close(legacy.export_particle_v_to_torch(), old["v_base"], rtol=1e-5, atol=1e-6)
            new = build_trajectory_samples(compatible, device, [index])[0]
            error = new["target"].square().mean().item()
            print(f"reference frame={index} gradient_v_mse={old['target'].square().mean().item():.8g} mls_v_mse={error:.8g}")
            assert error < 1e-10, "MLS reference mismatch: inspect transfer, epsilon, material and state phase"
            torch.testing.assert_close(new["F_trial"], data["F_trial"][index + 1].to(device), rtol=1e-6, atol=1e-6)
            # C amplifies float32 grid roundoff by inverse dx; near-zero entries need an absolute norm check.
            c_error = new["C"] - data["C"][index + 1].to(device)
            assert c_error.square().mean().sqrt() < 5e-5 and c_error.abs().max() < 1e-3, "MLS affine-state mismatch"
        for invalid in ({"transfer_scheme": "invalid"}, {"grid_mass_epsilon": -1.}, {"grid_mass_epsilon": float("nan")}):
            try:
                legacy.set_parameters_dict(invalid, device=str(device))
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid MPM settings accepted: {invalid}")
        print("original UniPhy MLS reference checks passed; numerical compatibility, not learned accuracy")
    dt = 1e-4
    data, known_delta = controlled_trajectory(device, dt)
    reordered_ids = data["particle_id"].clone()
    reordered_ids[1, :2] = reordered_ids[1, :2].flip(0)
    incomplete = dict(data)
    del incomplete["F_trial"]
    for invalid in (dict(data, time=2 * data["time"]), dict(data, particle_id=reordered_ids), incomplete):
        try:
            build_trajectory_samples(invalid, device)
        except ValueError:
            pass
        else:
            raise AssertionError("Sparse intervals, changed particle order or missing state accepted")
    samples = build_trajectory_samples(data, device)
    reverse = build_trajectory_samples(data, device, frame_indices=[1, 0])
    assert len(samples) == 2 and not torch.equal(data["v"][0], data["v"][1])
    for normal, reordered in zip(samples, reversed(reverse)):
        assert normal["frame_index"] == reordered["frame_index"]
        torch.testing.assert_close(normal["v_base"], reordered["v_base"], rtol=1e-5, atol=5e-7)
        torch.testing.assert_close(normal["target"], reordered["target"], rtol=1e-5, atol=5e-7)
        torch.testing.assert_close(normal["target"], known_delta, rtol=1e-5, atol=5e-7)
        assert torch.count_nonzero(normal["target"]) > 0
    # Precomputed GT loading uses the same schema and never writes an asset to disk.
    sample_buffer = io.BytesIO()
    torch.save(samples[0], sample_buffer)
    sample_buffer.seek(0)
    gt = load_gt(sample_buffer, device)
    assert gt["object_id"].dtype == gt["material_id"].dtype == torch.int64

    model = PostStepRelationMPM(n_materials=len(MATERIAL_VOCAB), local_radius=.095,
                                contact_radius=.09, max_tokens=4096).to(device)
    labels = (data["object_id"], data["material_id"])
    zero = make_baseline_solver(data, 0, device)
    zero.relation_step = lambda solver, step_dt: model.step_solver(solver, *labels, step_dt, data["object_relations"])
    zero.p2g2p(0, dt, device=str(device))
    x_base, v_base = samples[0]["x_base"], samples[0]["v_base"]
    torch.testing.assert_close(zero.export_particle_x_to_torch(), x_base, rtol=1e-5, atol=5e-7)
    torch.testing.assert_close(zero.export_particle_v_to_torch(), v_base, rtol=1e-5, atol=5e-7)
    if device.type == "cpu":
        assert torch.equal(zero.export_particle_x_to_torch(), x_base)
        assert torch.equal(zero.export_particle_v_to_torch(), v_base)
    prediction = model(x_base, v_base, gt["object_id"], gt["material_id"], gt["mass"], dt,
                       gt["E"], gt["nu"], gt["density"], gt["object_relations"], selection=gt["selection"])
    loss = (prediction - gt["target"]).square().mean()
    loss.backward()
    assert model.velocity_output.weight.grad is not None and torch.isfinite(model.velocity_output.weight.grad).all()
    assert torch.count_nonzero(model.velocity_output.weight.grad)
    with torch.no_grad():
        model.velocity_output.bias.copy_(known_delta[1])
    corrected = make_baseline_solver(data, 0, device)
    corrected.relation_step = lambda solver, step_dt: model.step_solver(solver, *labels, step_dt, data["object_relations"])
    corrected.p2g2p(0, dt, device=str(device))
    torch.testing.assert_close(corrected.export_particle_v_to_torch(), v_base + known_delta, rtol=1e-5, atol=5e-7)
    torch.testing.assert_close(corrected.export_particle_x_to_torch(), x_base + dt * known_delta, rtol=1e-5, atol=5e-7)
    assert torch.equal(corrected.export_particle_x_to_torch()[0], data["x"][0, 0].to(device))
    # This general residual has nonzero total momentum; no antisymmetric force restriction.
    assert (gt["mass"].unsqueeze(1) * known_delta).sum(0).norm() > 0
    metadata = dict(SAMPLE_METADATA, material_vocab=list(MATERIAL_VOCAB), relation_types=RELATION_TYPES,
                    n_materials=model.n_materials, hidden_dim=model.hidden_dim, local_radius=model.local_radius,
                    contact_radius=model.contact_radius, max_tokens=model.max_tokens, max_delta_v=model.max_delta_v)
    checkpoint = io.BytesIO()
    torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = PostStepRelationMPM(n_materials=len(MATERIAL_VOCAB), local_radius=.095,
                                   contact_radius=.09, max_tokens=4096).to(device)
    load_relation_checkpoint(restored, checkpoint, MATERIAL_VOCAB)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    print(f"real Warp post-MPM checks passed: device={device}, independent_samples=2, "
          f"controlled_target_max={float(known_delta.abs().max()):.6g}, zero_head=baseline, "
          "position=x_base+dt*delta_v, reverse_order=consistent, checkpoint=roundtrip; no real GT claim")


if __name__ == "__main__":
    main()
