"""CPU contract checks, not evidence of learned physical accuracy."""
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PhysGaussian"))
from relation_mpm import PostStepRelationMPM, RelationMPM, RELATION_TYPES, RELATION_SCHEMA, load_relation_checkpoint


def post_step_checks():
    torch.manual_seed(3)
    model = PostStepRelationMPM(n_materials=2, max_tokens=2)
    x = torch.tensor([[.2, .2, .2], [.22, .2, .2], [.6, .2, .2]])
    v = torch.zeros_like(x)
    oid = torch.tensor([0, 0, 1]); mid = oid.clone(); mass = torch.tensor([1., 2., 3.])
    attrs = (torch.full((3,), 1e5), torch.full((3,), .3), torch.full((3,), 500.))
    args = (x, v, oid, mid, mass, .01, *attrs)
    assert torch.equal(model(*args), v), "Zero residual head must reproduce the MPM baseline"
    with torch.no_grad():
        model.velocity_output.bias[:] = torch.tensor([.2, -.1, .3])
    single = model(x[:1], v[:1], oid[:1], mid[:1], mass[:1], .01, *(value[:1] for value in attrs))
    assert single.abs().sum() > 0, "An isolated particle must support a signed residual"
    predicted = model(*args)
    assert (predicted * mass[:, None]).sum(0).abs().sum() > 0, "Residuals must allow nonzero net momentum"
    assert model._last_token_count == 2
    assert torch.equal(predicted, model(x, v, oid, mid, mass, .02, *attrs)), "A residual must not be implicitly multiplied by dt"
    capped = PostStepRelationMPM(n_materials=2, max_tokens=2, max_delta_v=.05)
    with torch.no_grad(): capped.velocity_output.bias.fill_(1.)
    assert torch.allclose(capped(*args).norm(dim=1), torch.full((3,), .05))
    scaled = PostStepRelationMPM(n_materials=2, max_tokens=2, residual_scale=.002)
    with torch.no_grad(): scaled.velocity_output.bias.fill_(1.)
    assert torch.allclose(scaled(*args), torch.full_like(v, .002)), "Head output must use the physical residual scale"
    scaled.max_delta_v = .001
    assert torch.allclose(scaled(*args).norm(dim=1), torch.full((3,), .001)), "Clip after applying the residual scale"
    for invalid in (0., -1., float("nan"), float("inf")):
        try: PostStepRelationMPM(residual_scale=invalid)
        except ValueError: pass
        else: raise AssertionError("Invalid residual scale accepted")
    solid = PostStepRelationMPM(n_materials=2, max_tokens=2, use_state_features=True, residual_scale=.002)
    assert solid.input_schema == "post-mpm-solid-state-v2"
    assert model.input_schema == "post-mpm-velocity-residual-v1"
    assert not any("state_projection" in key or "residual_scale" in key for key in model.state_dict()), "Default parameter keys must remain legacy compatible"
    deformation = torch.eye(3).expand(3, 3, 3).clone()
    deformation[:, 0, 0] += .02
    deformation.requires_grad_()
    velocity_gradient = torch.zeros(3, 3, 3, requires_grad=True)
    with torch.no_grad():
        solid.state_projection.weight.zero_(); solid.state_projection.bias.zero_()
        solid.state_projection.weight[:, 0] = 1.; solid.state_projection.weight[:, 9] = 1.
        solid.velocity_output.weight[:, :solid.hidden_dim] = .02
    solid_args = dict(F_trial=deformation, C=velocity_gradient)
    solid_prediction = solid(*args, **solid_args)
    changed_f = deformation.detach().clone(); changed_f[:, 0, 0] += .1
    changed_c = velocity_gradient.detach().clone(); changed_c[:, 0, 0] = 10.
    assert not torch.allclose(solid_prediction, solid(*args, F_trial=changed_f, C=velocity_gradient))
    assert not torch.allclose(solid_prediction, solid(*args, F_trial=deformation, C=changed_c))
    solid_prediction.square().mean().backward()
    assert solid.state_projection.weight.grad.abs().sum() > 0
    assert deformation.grad.abs().sum() > 0 and velocity_gradient.grad.abs().sum() > 0
    for invalid in ({}, {"F_trial": deformation}, {"F_trial": deformation[:, 0], "C": velocity_gradient},
                    {"F_trial": deformation.double(), "C": velocity_gradient},
                    {"F_trial": deformation.to("meta"), "C": velocity_gradient},
                    {"F_trial": deformation, "C": torch.full_like(velocity_gradient, float("nan"))}):
        try: solid(*args, **invalid)
        except ValueError: pass
        else: raise AssertionError("Missing or invalid solid state accepted")
    # A local decoder skip preserves differences between particles assigned to one token.
    with torch.no_grad():
        model.velocity_output.weight[0, :model.hidden_dim] = 1.
    predicted = model(*args)
    assert not torch.equal(predicted[0], predicted[1])
    predicted.square().mean().backward()
    assert model.velocity_output.weight.grad is not None and model.position_projection.weight.grad.abs().sum() > 0
    assert model.force_output.weight.grad is None, "Legacy force decoder must not participate"
    selection = torch.tensor([1, 0, 0])
    assert torch.equal(model(*args, selection=selection)[0], torch.zeros(3))
    full = PostStepRelationMPM(n_materials=2, max_tokens=len(x))
    full.load_state_dict(model.state_dict())
    permutation = torch.tensor([2, 0, 1])
    shuffled = full(*(value[permutation] if isinstance(value, torch.Tensor) else value for value in args))
    assert torch.allclose(full(*args), shuffled[torch.argsort(permutation)], atol=1e-7, rtol=1e-5), "The full-particle branch must be permutation equivariant"

    saved_warp = sys.modules.get("warp")
    sys.modules["warp"] = SimpleNamespace(to_torch=lambda value: value)
    stored = {"x": x.clone(), "v": v.clone()}
    solver = SimpleNamespace(time=.5,
                             mpm_state=SimpleNamespace(particle_mass=mass, particle_density=attrs[2],
                                                       particle_selection=selection),
                             mpm_model=SimpleNamespace(E=attrs[0], nu=attrs[1]),
                             particle_velocity_modifier_params=[SimpleNamespace(start_time=0., end_time=1.,
                                                                                mask=torch.tensor([0, 1, 0]))],
                             export_particle_x_to_torch=lambda: stored["x"],
                             export_particle_v_to_torch=lambda: stored["v"],
                             import_particle_x_from_torch=lambda value, **kwargs: stored.__setitem__("x", value),
                             import_particle_v_from_torch=lambda value, **kwargs: stored.__setitem__("v", value))
    expected = model(*args, selection=torch.tensor([1, 1, 0])).detach()
    try:
        model.step_solver(solver, oid, mid, .01)
        assert torch.allclose(stored["v"] - v, expected), "Training and solver must use the same predictor"
        assert torch.allclose(stored["x"] - x, .01 * expected, atol=2e-8), "Position must use dt times the applied residual"
        assert torch.equal(stored["x"][:2], x[:2]), "Static and prescribed particles must remain unchanged"
        stored.update(x=x.clone(), v=v.clone()); solver.time = 1.
        model.step_solver(solver, oid, mid, .01)
        assert stored["v"][1].abs().sum() > 0, "Expired modifier masks must not freeze particles"
        stored.update(x=x.clone(), v=v.clone())
        solver.mpm_state.particle_F_trial = deformation.detach().clone()
        solver.mpm_state.particle_C = velocity_gradient.detach().clone()
        before_f, before_c = solver.mpm_state.particle_F_trial.clone(), solver.mpm_state.particle_C.clone()
        expected_solid = solid(*args, selection=selection, F_trial=before_f, C=before_c).detach()
        solid.step_solver(solver, oid, mid, .01)
        assert torch.allclose(stored["v"] - v, expected_solid), "Solver must read the same post-G2P F/C as training"
        assert torch.allclose(stored["x"] - x, .01 * expected_solid, atol=2e-8)
        assert torch.equal(solver.mpm_state.particle_F_trial, before_f) and torch.equal(solver.mpm_state.particle_C, before_c), "Translation split must retain baseline strain state"
        solid.baseline_discretization = {"transfer_scheme": "mls", "grid_mass_epsilon": 1e-7}
        for scheme, epsilon in ((0, 1e-7), (1, 0.)):
            solver.mpm_model.transfer_scheme, solver.mpm_model.grid_mass_epsilon = scheme, epsilon
            try: solid.step_solver(solver, oid, mid, .01)
            except ValueError: pass
            else: raise AssertionError("Checkpoint/solver baseline mismatch accepted")
        solver.mpm_model.transfer_scheme, solver.mpm_model.grid_mass_epsilon = 1, 1e-7
        solid.step_solver(solver, oid, mid, .01)
    finally:
        if saved_warp is None: del sys.modules["warp"]
        else: sys.modules["warp"] = saved_warp

    metadata = {"input_schema": model.input_schema, "material_vocab": ["a", "b"],
                "relation_types": RELATION_TYPES, "hidden_dim": model.hidden_dim, "n_materials": model.n_materials}
    for key in ("output_schema", "position_update", "max_tokens", "max_delta_v", "local_radius", "contact_radius"):
        metadata[key] = getattr(model, key)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "unit.pt"
        torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, path)
        load_relation_checkpoint(model, path, ["a", "b"])
        baseline = {"transfer_scheme": "mls", "grid_mass_epsilon": 1e-7}
        torch.save({"metadata": dict(metadata, baseline_discretization=baseline), "model_state_dict": model.state_dict()}, path)
        load_relation_checkpoint(model, path, ["a", "b"])
        assert model.baseline_discretization == baseline
        for invalid in ({"transfer_scheme": "other", "grid_mass_epsilon": 0.},
                        {"transfer_scheme": "mls", "grid_mass_epsilon": -1.},
                        {"transfer_scheme": "mls", "grid_mass_epsilon": float("nan")}, {}):
            torch.save({"metadata": dict(metadata, baseline_discretization=invalid), "model_state_dict": model.state_dict()}, path)
            try: load_relation_checkpoint(model, path, ["a", "b"])
            except ValueError: pass
            else: raise AssertionError("Invalid baseline discretization metadata accepted")
        torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, path)
        load_relation_checkpoint(model, path, ["a", "b"])
        assert model.baseline_discretization is None, "Legacy checkpoints must not retain a previously loaded baseline guard"
        for key, invalid in (("use_state_features", True), ("residual_scale", .002)):
            bad = dict(metadata); bad[key] = invalid
            torch.save({"metadata": bad, "model_state_dict": model.state_dict()}, path)
            try: load_relation_checkpoint(model, path, ["a", "b"])
            except ValueError: pass
            else: raise AssertionError(f"Incompatible opt-in metadata accepted: {key}")
        for key, invalid in (("input_schema", RELATION_SCHEMA), ("output_schema", "pair-force"),
                             ("position_update", "velocity-only"), ("max_tokens", 1), ("max_delta_v", .1)):
            bad = dict(metadata); bad[key] = invalid
            torch.save({"metadata": bad, "model_state_dict": model.state_dict()}, path)
            try: load_relation_checkpoint(model, path, ["a", "b"])
            except ValueError: pass
            else: raise AssertionError(f"Incompatible post-step metadata accepted: {key}")
        empty = PostStepRelationMPM(n_materials=2, max_tokens=2)
        torch.save({"metadata": metadata, "model_state_dict": empty.state_dict()}, path)
        try: load_relation_checkpoint(empty, path, ["a", "b"])
        except ValueError: pass
        else: raise AssertionError("Zero residual output head accepted")
        zero_metadata = dict(metadata, zero_correction=True)
        torch.save({"metadata": zero_metadata, "model_state_dict": empty.state_dict()}, path)
        load_relation_checkpoint(empty, path, ["a", "b"])
        assert torch.equal(empty(*args), v), "Explicit baseline checkpoint must apply no correction"
        torch.save({"metadata": zero_metadata, "model_state_dict": model.state_dict()}, path)
        try: load_relation_checkpoint(model, path, ["a", "b"])
        except ValueError: pass
        else: raise AssertionError("Nonzero weights accepted as an explicit zero correction")
        solid_metadata = dict(metadata, input_schema=solid.input_schema, use_state_features=True, residual_scale=.002)
        torch.save({"metadata": solid_metadata, "model_state_dict": solid.state_dict()}, path)
        load_relation_checkpoint(solid, path, ["a", "b"])
        for key in ("use_state_features", "residual_scale"):
            bad = dict(solid_metadata); del bad[key]
            torch.save({"metadata": bad, "model_state_dict": solid.state_dict()}, path)
            try: load_relation_checkpoint(solid, path, ["a", "b"])
            except ValueError: pass
            else: raise AssertionError(f"Nondefault metadata omitted: {key}")


def physcore_split_checks():
    """CPU-only split contracts using synthetic tensors, not simulated accuracy data."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_relation_full import MATERIAL_VOCAB, physcore_trajectory_cases, physcore_gt_step
    episode = {field: torch.arange(4., dtype=torch.float32).reshape(4, *((1,) * (len(shape) + 1))).expand(4, 2, *shape).clone()
               for field, shape in (("particle_coords", (3,)), ("particle_velocities", (3,)),
                                    ("particle_deformation_gradients", (3, 3)), ("particle_affine_velocity_matrices", (3, 3)))}
    observed = []
    def mock_engine(x, v, F, deltas, log_E, nu, C):
        observed.append(tuple(value.clone() for value in (x, v, F, C)))
        assert deltas.shape[0] == 1
        return dict(predicted_positions=(x + 100).unsqueeze(0), final_velocity=v + 100,
                    final_deformation_gradient=F + 100, final_C=C + 100)
    for index in (2, 0, 1):
        physcore_gt_step(mock_engine, episode, index, dict(log_E=torch.ones(1), nu=torch.ones(1)), "cpu")
    for index, inputs in zip((2, 0, 1), observed):
        assert all(torch.equal(value, torch.full_like(value, float(index))) for value in inputs)
    assert episode["particle_coords"][0].count_nonzero() == 0

    def trajectory(episode_id, stiffness, frames=3):
        n = 2
        data = dict(x=torch.tensor([[.3, .4, .5], [.4, .4, .5]]).expand(frames, n, 3).clone(),
                    v=torch.zeros(frames, n, 3), F_trial=torch.eye(3).expand(frames, n, 3, 3).clone(),
                    C=torch.zeros(frames, n, 3, 3), time=torch.arange(frames, dtype=torch.float64) * .001,
                    particle_id=torch.arange(n), selection=torch.zeros(n, dtype=torch.int32),
                    volume=torch.full((n,), .001), mass=torch.ones(n), density=torch.full((n,), 1000.),
                    E=torch.full((n,), float(stiffness)), nu=torch.full((n,), .3),
                    object_id=torch.zeros(n, dtype=torch.long), material_id=torch.zeros(n, dtype=torch.long),
                    material_vocab=list(MATERIAL_VOCAB), source=dict(episode_id=episode_id, dataset="synthetic split self-check"),
                    mpm_config=dict(material="jelly", n_grid=20, grid_lim=1., substep_dt=.001,
                        rpic_damping=0., grid_v_damping_scale=1., g=[0., -9.8, 0.], boundary_conditions=[dict(
                            type="surface_collider", point=[0., .1, 0.], normal=[0., 1., 0.],
                            surface="separate", friction=0., start_time=0., end_time=10.)]))
        return data

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        data = [trajectory(str(i), 1000. * (i + 1)) for i in range(4)]
        paths = [root / f"episode_{i}.pt" for i in range(4)]
        for path, item in zip(paths, data):
            torch.save(item, path)
        config = dict(train_trajectories=[str(path) for path in paths[:2]],
                      validation_trajectories=[str(paths[2])], test_trajectories=[str(paths[3])])
        cases, multiple = physcore_trajectory_cases(config)
        assert multiple and [len(cases[name]) for name in ("train", "validation", "test")] == [2, 1, 1]
        assert all(case["first"] == 0 and case["stop"] == 2 for items in cases.values() for case in items)
        assert len({case["content_sha256"] for items in cases.values() for case in items}) == 4

        def rejects(bad, override=None):
            try:
                physcore_trajectory_cases(bad, override)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError("Leaked/ambiguous episode split accepted")

        rejects(dict(config, validation_trajectories=[str(paths[0])]))
        rejects(dict(config, test_trajectories=[]))
        rejects(dict(config, train_trajectories=str(paths[0])))
        rejects(dict(config, trajectory=str(paths[0])))
        rejects(config, str(paths[0]))
        copied = root / "copied.pt"
        copied.write_bytes(paths[0].read_bytes())
        rejects(dict(config, validation_trajectories=[str(copied)]))
        # A different serialization/name/ID must not disguise duplicated numerical data.
        torch.save(dict(data[0], source=dict(episode_id="renamed")), copied)
        rejects(dict(config, validation_trajectories=[str(copied)]))
        torch.save(dict(data[2], source=dict(episode_id="0")), copied)
        rejects(dict(config, validation_trajectories=[str(copied)]))
        torch.save(dict(data[2], source={}), copied)
        rejects(dict(config, validation_trajectories=[str(copied)]))
        torch.save(dict(data[2], episode_id="conflict"), copied)
        rejects(dict(config, validation_trajectories=[str(copied)]))
        legacy = root / "legacy.pt"
        torch.save(dict(trajectory("legacy", 1000., frames=51), source={}), legacy)
        single, multiple = physcore_trajectory_cases(dict(trajectory=str(legacy)))
        assert not multiple
        assert {name: (items[0]["first"], items[0]["stop"]) for name, items in single.items()} == {
            "train": (0, 30), "validation": (31, 39), "test": (40, 50)}


def main():
    model = RelationMPM(n_materials=2, local_radius=.1, contact_radius=.1, max_refresh_steps=3)
    # Synthetic values exist only in this unit check; never saved as trained assets.
    with torch.no_grad():
        model.force_output.bias[:] = torch.tensor([.2, .2, .1, .2])
    x = torch.tensor([[0., 0., 0.], [.02, 0., 0.], [.3, 0., 0.]])
    v = torch.zeros_like(x)
    oid = torch.tensor([0, 0, 1]); mid = oid.clone(); mass = torch.ones(3)
    attrs = (torch.ones(3) * 1e5, torch.ones(3) * .3, torch.ones(3) * 500)
    def refresh():
        return model.refresh_graph(x, v, oid, mid, mass, *attrs)
    g = refresh(); rest = g["rest_distance"][0].clone()
    x[2, 0] = .08
    g = refresh()
    assert model.refresh_records[-1]["contact_started"]
    assert torch.equal(g["rest_distance"][0], rest)
    dv = model.apply_relation_graph(x, v, mass, .01, g)
    assert dv.abs().sum() > 0
    assert torch.allclose(dv.sum(0), torch.zeros(3), atol=1e-6)
    count = len(model.refresh_records)
    x[2, 0] = .11
    refresh()
    assert len(model.refresh_records) == count  # hysteresis, no spurious exit
    x[2, 0] = .3; x[1, 0] = .025
    g = refresh()
    assert model.refresh_records[-1]["contact_ended"]
    assert len(g["pairs"]) == 1 and torch.equal(g["rest_distance"][0], rest)
    for _ in range(3): refresh()
    assert model.refresh_records[-1]["reason"] == "maximum_interval"
    # Actual step_solver application against a minimal tensor-backed Warp interface.
    saved_warp = sys.modules.get("warp")
    sys.modules["warp"] = SimpleNamespace(to_torch=lambda value: value)
    solver = SimpleNamespace(mpm_state=SimpleNamespace(particle_mass=mass, particle_density=attrs[2],
                              particle_selection=torch.zeros(3)),
                             mpm_model=SimpleNamespace(E=attrs[0], nu=attrs[1], dx=.01),
                             export_particle_x_to_torch=lambda: x,
                             export_particle_v_to_torch=lambda: v)
    applied = []
    solver.import_particle_v_from_torch = lambda value, **kwargs: applied.append(value)
    model.reset_cache()
    x[2, 0] = .08
    try:
        model.step_solver(solver, oid, mid, .01)
        assert applied[-1].abs().sum() > 0
        solver.contact_event_revision = 1
        model.step_solver(solver, oid, mid, .01)
        assert model.refresh_records[-1]["reason"] == "surface_contact_event"
        solver.mpm_state.particle_selection[:] = 1
        model.step_solver(solver, oid, mid, .01)
        assert torch.equal(applied[-1], v)
    finally:
        if saved_warp is None: del sys.modules["warp"]
        else: sys.modules["warp"] = saved_warp
    metadata = {"input_schema": RELATION_SCHEMA, "material_vocab": ["a", "b"],
                "relation_types": RELATION_TYPES, "hidden_dim": model.hidden_dim, "n_materials": 2}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "unit.pt"
        torch.save({"metadata": metadata, "model_state_dict": model.state_dict()}, path)
        load_relation_checkpoint(model, path, ["a", "b"])
        try: load_relation_checkpoint(model, path, ["b", "a"])
        except ValueError: pass
        else: raise AssertionError("Reordered material vocabulary accepted")
        state = dict(model.state_dict()); del state["query.weight"]
        torch.save({"metadata": metadata, "model_state_dict": state}, path)
        try: load_relation_checkpoint(model, path, ["a", "b"])
        except RuntimeError: pass
        else: raise AssertionError("Partial weights accepted")
        empty = RelationMPM(n_materials=2)
        torch.save({"metadata": metadata, "model_state_dict": empty.state_dict()}, path)
        try: load_relation_checkpoint(model, path, ["a", "b"])
        except ValueError: pass
        else: raise AssertionError("Zero head accepted")
    post_step_checks()
    physcore_split_checks()
    print("relation events, post-step residuals, independent episode splits and strict checkpoint checks passed")

if __name__ == "__main__": main()
