"""Run with an existing Warp environment; exercises the real CPU collider kernel."""
import sys
from pathlib import Path
import numpy as np
import warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PhysGaussian" / "mpm_solver_warp"))
from mpm_solver_warp import MPM_Simulator_WARP


def main():
    wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[1] / ".cache" / "warp")
    wp.init()
    for surface, velocity, expected, constrained in [
        ("separate", [1., -2., 0.], [1., 0., 0.], 1),
        ("separate", [1., 2., 0.], [1., 2., 0.], 0),
        ("slip", [1., 2., 0.], [1., 0., 0.], 1),
        ("sticky", [1., 2., 0.], [0., 0., 0.], 1),
    ]:
        solver = MPM_Simulator_WARP(1, n_grid=4, device="cpu")
        solver.add_surface_collider([0., .5, 0.], [0., 1., 0.], surface=surface)
        mass = np.zeros((4, 4, 4), dtype=np.float32); mass[0, 0, 0] = 1
        solver.mpm_state.grid_m = wp.array(mass, dtype=float, device="cpu")
        velocities = np.zeros((4, 4, 4, 3), dtype=np.float32); velocities[0, 0, 0] = velocity
        for enabled in (False, True):
            solver.mpm_state.grid_v_out = wp.array(velocities, dtype=wp.vec3, device="cpu")
            solver._contact_counters.zero_()
            wp.launch(solver.grid_postprocess[0], dim=(4, 4, 4),
                      inputs=[0., .001, solver.mpm_state, solver.mpm_model, solver.collider_params[0],
                              int(enabled), solver._contact_counters], device="cpu")
            np.testing.assert_allclose(solver.mpm_state.grid_v_out.numpy()[0, 0, 0], expected)
        occupied, inward, corrected, active = solver._contact_counters.numpy()
        assert occupied == 1 and active == constrained  # empty grid nodes excluded
        assert corrected == int(velocity != expected)
        solver._collect_surface_contacts(0, .001)
        result = solver.contact_diagnostics_snapshot()
        assert len(result["events"]) == constrained
        solver._contact_counters.zero_(); solver.time = .001
        solver._collect_surface_contacts(0, .001)
        result = solver.contact_diagnostics_snapshot()
        assert len(result["events"]) == constrained
        if constrained: assert result["events"][0]["event"] == "contact_end"
    solver = MPM_Simulator_WARP(1, n_grid=4, device="cpu")
    solver.add_surface_collider([0., .5, 0.], [0., 1., 0.], surface="separate", radius=.3)
    velocities = np.zeros((4, 4, 4, 3), dtype=np.float32)
    velocities[0, 0, 0] = [1., -2., 0.]
    velocities[2, 0, 0] = [1., -2., 0.]
    solver.mpm_state.grid_v_out = wp.array(velocities, dtype=wp.vec3, device="cpu")
    wp.launch(solver.grid_postprocess[0], dim=(4, 4, 4),
              inputs=[0., .001, solver.mpm_state, solver.mpm_model, solver.collider_params[0],
                      0, solver._contact_counters], device="cpu")
    np.testing.assert_allclose(solver.mpm_state.grid_v_out.numpy()[0, 0, 0], [1., 0., 0.])
    np.testing.assert_allclose(solver.mpm_state.grid_v_out.numpy()[2, 0, 0], [1., -2., 0.])
    for radius in (-1., float("nan"), float("inf")):
        try: solver.add_surface_collider([0., .5, 0.], [0., 1., 0.], radius=radius)
        except ValueError: pass
        else: raise AssertionError("invalid collider radius accepted")
    print("actual Warp surface kernel checks passed")

if __name__ == "__main__": main()
