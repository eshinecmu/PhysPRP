import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import torch

module_path = Path(__file__).parents[1] / "model" / "dynamics" / "pac_nerf" / "lib" / "material_field.py"
spec = importlib.util.spec_from_file_location("material_field", module_path)
material_field = importlib.util.module_from_spec(spec)
spec.loader.exec_module(material_field)
particle_materials = material_field.particle_materials
particles_from_material = material_field.particles_from_material


class MaterialFieldTest(TestCase):
    def test_maps_pixie_values_to_pac_particles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "material.npz"
            np.savez(
                path,
                density=np.array([[[10, 11]], [[19, 20]]]),
                E=np.array([[[100, 110]], [[190, 200]]]),
                nu=np.array([[[0.2, 0.2]], [[0.25, 0.25]]]),
            )
            rho, mu, lam = particle_materials(
                path, torch.tensor([[0., 0., 0.], [2., 2., 2.]]), torch.ones(2)
            )
            self.assertEqual(rho.tolist(), [10, 20])
            self.assertAlmostEqual(mu[0].item(), 100 / 2.4, places=4)
            self.assertAlmostEqual(lam[1].item(), 80, places=4)

    def test_uses_pixie_occupancy_as_particle_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "material.npz"
            mask = np.zeros((2, 2, 2), dtype=bool)
            mask[0, 0, 0] = mask[1, 1, 1] = True
            np.savez(path, mask=mask)
            particles = particles_from_material(path, torch.tensor([-1., -2., -3.]), torch.tensor([1., 2., 3.]))
            self.assertTrue(torch.equal(particles, torch.tensor([[-1., -2., -3.], [1., 2., 3.]])))


if __name__ == "__main__":
    main()
