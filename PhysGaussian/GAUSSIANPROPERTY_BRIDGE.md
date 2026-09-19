# GaussianProperty bridge

GaussianProperty is an open-source ICCV 2025 implementation for LMM-based material-property prediction on segmented 3D Gaussians. The upstream code is vendored at `../GaussianProperty`.

`gaussian_property_adapter.py` converts its CSV-like prediction output into JSON records with material, Shore hardness, Young's modulus, Poisson ratio, and density for PhysGaussian preprocessing.

The upstream README releases property-prediction code and marks physical simulation models/configurations as TODO; PhysGaussian remains the simulator.
