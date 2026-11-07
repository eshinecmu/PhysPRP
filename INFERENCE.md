# Single-image inference delivery

Use `run_full_pipeline.py` for the primary `REST3D/` implementation. The historical
`third_party/REST3D` copy is retained for provenance and is not an execution root.
The input is one image plus an explicit scene configuration; this path never reads
later video frames or the old bird calibration/background artifacts.

## Run

Copy `config/runtime.example.json` to an external runtime configuration and set
its interpreter, source and weight paths. Set API secrets in the process environment,
not this file. `LAB_LLM_API_KEY`, `LAB_LLM_BASE_URL`, `LAB_LLM_MODEL` route both VLM
stages (or their documented OpenAI fallback). No keys are saved in reports.

```sh
python run_full_pipeline.py --image /path/to/image.png \
  --config config/single_image_bird.json --runtime /path/to/runtime.json \
  --work_dir outputs/my-run --without_relation
```

Set the object scope through the shell launcher. Interaction mode requires the
exact REST3D mask name for its subject; full mode includes every segmented object.

```sh
MODEL_SCOPE=interaction SUBJECT=white_bird_toy_000 bash run_inference.sh \
  --image /path/to/image.png --config config/single_image_bird.json \
  --runtime /path/to/runtime.json --work_dir outputs/interaction --without_relation
MODEL_SCOPE=full bash run_inference.sh \
  --image /path/to/image.png --config config/single_image_bird.json \
  --runtime /path/to/runtime.json --work_dir outputs/full --without_relation
```

Both modes reuse REST3D masks and its scene tree. The interaction selector keeps
the non-floor support/attachment chain and adds conservative nearby mask-box
candidates; that 2D distance is a recall proxy, not a physical distance. The
selected manifest also drives GaussianProperty, so every selected object's
material is predicted. REST3D movable objects enter MPM as dynamic particles.
Fixed interaction objects need an explicit supported plane collider; arbitrary
mesh contact is not implemented in the current Warp path.

`--without_relation` is an explicit unlearned MPM baseline. It cannot pass learned
Transformer inference. Omit it only with a compatible, traceable checkpoint in the
runtime configuration. `--assets_only` stops after reconstruction, native Gaussian
assembly, first-frame rendering and material inference. Missing dependencies fail
preflight; there is no implicit mesh initialization fallback or package installation.

Use a new empty run directory. `--resume` validates the input/config/source identity
and every completed stage artifact hash. Stage logs and commands are under
`validation/`; `run.json` never equates baseline completion with full acceptance.
The material VLM's parameters are semantic lookup priors, not measured properties
or a hardness inversion. `material_actual.json` records requested vs solver values.

## Scene contract

`scene_scale` is meters per SAM3D source unit, explicitly assumed or measured.
Native Gaussian centers and full covariance undergo the exact object-to-scene
transform; source-relative poses and heights are retained. The common MPM mapping
is only the explicit `mpm_origin` translation. Implicit per-object normalization
and orientation correction are disabled in this route. Camera comes from the
single-image pointmap and is checked by actual Gaussian first-frame rendering.

`objects` maps reconstructed names to `dynamic`, `static`, `floor`, or `background`.
Every object must have a role, with an explicit `default_role` if applicable.
`required_objects` failures stop the adapter. Dynamic objects have explicit velocity
in m/s (zero if omitted, recorded as the release assumption in the example).
Static/floor objects require an explicit plane `collider` in world meters; background
is render-only. Current support is plane colliders, not arbitrary static mesh
collision. Scene-wide constitutive type is shared; per-object E/nu/density are
assigned with a fixed material vocabulary. The example plane and scale are illustrative
assumptions, not inferred physical truth or fitted to the bird video.

## Legacy post-step checkpoint contract (Gaussian rendering route)

`model_state_dict` must exactly match all network layers; partial loading is rejected.
Velocity correction now uses `PostStepRelationMPM` and
`input_schema: post-mpm-velocity-residual-v1`. Its metadata must include exact
`material_vocab`, `relation_types`, `hidden_dim`, `n_materials`,
`output_schema: delta-v-after-mpm-v1`, `position_update: x_base+dt*delta_v`,
`max_tokens`, `max_delta_v`, `local_radius`, and `contact_radius`.
Legacy pair-force checkpoints cannot be used as post-step residual weights.
The loader rejects an unmarked zero output head and nonfinite weights. An explicitly
selected zero-correction candidate must declare that in its metadata. Passing this
check does not prove physical accuracy; the single-jelly experiments and their
limitations are retained under `outputs/uniphy_jelly_*`.

The MPM baseline completes G2P before the velocity corrector reads `x_base/v_base`.
It recomputes signed three-component residuals every substep, using the same token
encoder and particle decoder in training and inference. Velocity becomes
`v_base + delta_v`; position becomes `x_base + dt * delta_v`, without integrating
the baseline velocity twice. Static and actively prescribed particles are masked.
The original sparse encoder is reused; this is not a claim of WorldParticle core
integration. Topology is rebuilt each step rather than caching dynamic residuals.
The split corrects translation only: `C/F_trial` keep the baseline velocity gradient;
spatially varying residuals do not yet provide a consistent deformation correction.
No trained relationship results are claimed for the baseline. The old PAC material
gradient check failed with 22.594% relative error and remains a separate unresolved
training result.

## PhysCoRe simulation training (default)

`train_relation_full.py` now defaults to the pinned official
[PhysCoRe implementation](https://github.com/lunarlab-gatech/PhysCoRe/tree/ba7ddf744c7468369529d3baf996fb829fec30dd).
The official MfM/RfD entrypoints remain available. The default numeric experiment
uses the original **WorldParticle Transformer** and the original PhysCoRe
differentiable MPM. `PhysGaussian/worldparticle_grid_correction.py` directly calls
CConv/LocalFeatureExtractor, k-partite merging, RoPE and SuperParticleDecoder.
The adapter supplies physical features and projects bounded particle corrections
onto the MPM grid. Historical GVC configurations retain the old U-Net.
Keep the bundled licenses.

The upstream full workflow is: material-field augmentation → MfM supervision on material
GT → freeze MfM → RfD reconstruction training. RfD corrects grid velocities after
the grid update and before G2P. Its loss backpropagates through the simulated
positions; x/v/F/C carry between truncated-BPTT segments. There is no offline
particle `target = v_true_next-v_base` in this training path.

The default numeric config now uses **independent elastic video episodes**:
the original NCLaw cube and UniPhy `Elastic_Toy` for training, the archived PAC
compression sequence for validation, and the complete UniPhy `Elastic_Blobby`
sequence for testing. Each source appears in exactly one split. The previously
used single jelly trajectory was an integration probe, not a multivideo dataset.
These are simulation-GT materials and complete particle states, not measured
real-world parameters or RGB-D reconstruction. This remains a **known-material
correction adaptation**, not MfM training or Pixie integration. WorldParticle uses
the existing PyTorch 2.2/Open3D runtime, separately from historical GVC runs:

```sh
source /mnt/workspace/eshine/miniconda3/bin/activate pixie
native="$PWD/outputs/worldparticle_native_integration_20261020_01/runtime_deps"
export PYTHONPATH="$native:$PWD/runtime_deps"
for d in /usr/local/lib/python3.11/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export LD_LIBRARY_PATH="$native/torch/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=2 ATTN_IMPL=sdpa OMP_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/outputs/worldparticle_single_step_20261021/cache"
export PHYSPRP_WARP_CACHE_DIR="$XDG_CACHE_HOME/warp" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
python PhysGaussian/remote_mpm_smoke.py --physcore-splits config/physcore_jelly.yaml --device cuda:0
python train_relation_full.py --config config/physcore_jelly.yaml
python train_relation_full.py --config config/physcore_jelly.yaml \
  --evaluate-only --checkpoint outputs/worldparticle_single_step_20261021/training/checkpoint.pt
```

Omitting `--steps` uses the configured five-epoch pilot budget and consumes every
training episode in full. Each shuffled state pair restarts from complete GT
x/v/F/C and runs one substep. Fifty independent pairs accumulate gradients per
update; no predicted state carries between training pairs. The original
Transformer position head is divided by dt, bounded at .002 m/s per component,
and projected with mass-weighted B-spline P2G. No global momentum projection is
imposed. Training uses mean particle Euclidean position distance, the same
objective as upstream tracked_l2. Selection uses single-step validation MSE,
including zero correction. Training saves selected and final candidate weights
and exits; evaluation is a separate process that compares baseline, selected and
last candidate using both GT-reset single steps and free closed-loop rollouts.
Input/source hashes and architecture are checked strictly. Episode paths, IDs,
file and numerical-content hashes must be disjoint. No RGB-D observation or
video loss is fabricated.

`tools/prepare_uniphy_jelly.py INPUT OUTPUT --format physcore` can export official
episode fields. INPUT is the original NCLaw ZIP or an original UniPhy demo
directory. `--pac-probe` re-exports the archived independent compression scene.
The demo importer checks substep recurrence, removes the unwritten padding row,
and aligns `F_trial[t]=GtF[t+1]`. It applies only a common integer-grid translation
to positions and the sticky ground, with no change in physical scale or material
units. The PAC validation reproduces the archived video positions and exports
real internal states; it is a historical simulation, not a new measured holdout.
The in-memory adapter transforms y-up to z-up with a right-handed rotation, including
F/C, gravity and every supplied plane; the actual solver receives the supplied rest volume,
density and boundaries. It rejects unsupported nonuniform masses, static
particles and multiple objects. Source discretization is provenance: the official
gradient-based MPM is retained, so it is not the MLS-aligned baseline of the earlier
experiment. Checkpoints record this distinction and are strictly loaded for numeric
evaluation; the old `gs_simulation.py` post-step loader cannot consume RfD weights.
UniPhy demo and PAC GT use Neo-Hookean elasticity. The pinned material registry
lacks that law, so the numeric adapter supplies the original Kirchhoff-stress
formula while retaining the official MPM. The NCLaw cube retains original
CorotatedElasticity. Source models are explicit in each episode contract.
Historical GVC inputs use a corotated-stress proxy. The WorldParticle adapter
instead reads F/C and the explicit constitutive flag with dt, gravity, volume,
material embedding and boundary information. Discretization
differences between source and baseline remain. The optional PAC public
position-only GT is separate and is not accepted as complete x/v/F/C data.

For real/augmented episodes, delegate directly to the original stages with explicit
dataset/config paths. The RfD config must identify the frozen MfM checkpoint:

```sh
python train_relation_full.py --stage mfm --config /path/to/train_MfM.yaml
python train_relation_full.py --stage rfd --config /path/to/train_RfD.yaml
python train_relation_full.py --stage rfd --evaluate-only \
  --config /path/to/validate_RfD.yaml --skip-render
python PhysGaussian/remote_mpm_smoke.py --physcore --device cuda:3 \
  --trajectory /path/to/solid_substeps.pt
```

These stages retain upstream material/confidence losses and observation/PID
validation rather than substituting the numeric-probe evaluation. The jelly alone
does not provide the cameras or material variation required to reproduce those
stages. In particular, its `ln(E)=11.5129` exceeds the upstream default MfM
`log_E_max=11`; a real MfM config for this material must raise that bound.
The original MfM trainer accepts only `--config`; put its training parameters in
that YAML. RfD and validation retain their upstream command-line overrides.

## Legacy post-MPM trajectory supervision

```sh
python train_relation_full.py --pipeline post-step --trajectory /path/to/solid_substeps.pt \
  --output outputs/post_step_velocity_20261020/solid_checkpoint.pt \
  --local-radius 0.095 --contact-radius 0.09 --device cuda:0
```

The trajectory dictionary must contain `x/v [T,N,3]`,
`F_trial/C [T,N,3,3]`, floating `time [T]`, persistent integer
`particle_id [N]` or unchanged `[T,N]`, and per-particle
`selection/volume/mass/E/nu/density/object_id/material_id [N]`.
State/property tensors use float32; float64 time avoids substep precision loss.
`selection=0` means simulated and `1` means static. Mass must equal density times
volume. `material_vocab` must exactly equal the ordered inference `MATERIAL_VOCAB`;
optional `object_relations` maps sorted object-ID pairs to `RELATION_TYPES` values.

`mpm_config` must explicitly contain `material: jelly`, `n_grid`, `grid_lim`,
`g`, `rpic_damping`, `grid_v_damping_scale`, `substep_dt`, and
`boundary_conditions`. Every saved interval must be exactly one declared MPM
substep. Only elastic solids and static surface/bounding-box boundaries are
currently restored; plastic/dynamic boundary histories are rejected rather than
filled with defaults. Restore `F_trial`, not just the exported elastic `F`.

Each transition independently rebuilds the actual MPM solver from the true current
state, disables learned correction, and runs one substep. Training inputs are its
`x_base/v_base`; the target is `v_true_next-v_base`, with MSE on the predicted
velocity residual. No correction is learned by subtracting independently rolled
trajectories. The current in-memory transition preparation is intended for small
datasets; large datasets need streamed baseline generation.

`--gt` accepts a precomputed `x_base/v_base/target` sample only with the explicit
`SAMPLE_METADATA` contract in `train_relation_full.py`. Old force/pre-step labels
are rejected. Output checkpoints contain `model_state_dict` and matching
`metadata`; an existing output path is never overwritten.

For the original UniPhy/NCLaw jelly, explicitly use `transfer_scheme: mls` and
`grid_mass_epsilon: 1e-7` in `mpm_config`. Its stress transfer and deformation update
use MLS, while the default `gradient` mode preserves PhysGaussian's B-spline
gradient discretization. The two valid discretizations should not be silently
treated as identical. `tools/prepare_uniphy_jelly.py` now records these settings;
older converted files can use the corresponding trainer CLI overrides.

`--use-state-features --normalize-residual` observes post-G2P `F-I` and `dt*C`
and sets the output/loss scale from training targets only. It uses the distinct
`post-mpm-solid-state-v2` checkpoint schema. Small 1000-particle experiments can
set `--max-tokens 1000` to avoid the index-sampling approximation. The split still
corrects only x/v; no approximate deformation-gradient correction is implied.

`--temporal-split --allow-zero-correction --select-closed-loop` includes the
unchanged MPM as a candidate and requires both one-step and validation rollout
errors to improve. An explicitly selected zero head is recorded as
`zero_correction: true`; it means no useful learned correction was established.
New checkpoints also record their baseline transfer/epsilon settings, and the
inference hook refuses a mismatched solver. Reusing a previously viewed test
segment is a diagnostic comparison, not a fresh holdout.
In a `gs_simulation.py` scene JSON, put `transfer_scheme` and `grid_mass_epsilon`
at the top level (not inside the relation-model settings); loading a checkpoint
does not silently change the solver. Near-roundoff targets after aligning the
teacher do not establish a useful learned correction, even if their relative
validation error changes slightly.

```sh
python PhysGaussian/remote_mpm_smoke.py --device cpu
python PhysGaussian/remote_mpm_smoke.py --device cuda:0
```

These checks use a controlled synthetic teacher and the real Warp solver. They
verify independent baseline resets, signed velocity residuals, position updates,
gradient flow and checkpoint roundtrips, without claiming real-data accuracy.

## Small checks

```sh
python tests/check_pipeline_receipts.py
python tests/test_material_contract.py
python tests/test_first_frame_camera.py
python tests/test_preflight.py
python tests/test_relation_events.py
python tests/check_rest3d_runtime.py
python tests/check_native_scene.py
python tests/test_surface_contacts.py
```

The last two require the existing scientific/PLY and Warp environments. These
checks do not replace actual image reconstruction, kernel rollout, and rendered
asset review. Runtime preflight records code revisions, dirty status, module versions,
weight sizes/SHA256 and declared sources. External cache provenance that is unknown
is recorded as unknown rather than invented. A source snapshot is not itself proof
of clean-checkout full inference; that acceptance remains tied to the run receipts.

The example can explicitly set `ground_from_pointmap` and `gravity_from_ground`.
This fits a plane from the genuine single-image floor mask and reconstruction
pointmap, saves residuals/provenance in `resolved_scene.json`, and derives gravity
orientation from that plane. It is a model estimate with assumed metric scale,
not a measurement or video calibration. Duplicate/copy-fallback floor masks are
rejected before loading SAM3D.

An inspected successful segmentation can be reused with
`--segmentation_cache /path/to/prior-run`; the image, object prompt, backend and all
producer artifact hashes must match. The new receipt records the producer identity.
It is explicitly a cache-backed run, not a fresh VLM invocation. Any newly required
objects and floor integrity are revalidated before reconstruction.

The observed external sources use SAM3 commit
`5dd401d1c5c1d5c3eedff06d41b77af824517619` and SAM3D commit
`e19b1699d492e132892ba6c4c6594e94fbdac8f3`, plus the small patches under
`runtime_patches/`. Apply with `git apply --check` then `git apply` in the matching
external checkout. The runtime uses existing Warp 0.10.1; it exposes `config.version`
rather than `__version__`. Point `pythonpath` to a project-owned directory containing
the declared existing Warp package if the rendering environment lacks it; never
prepend an entire unrelated Python environment's site-packages.

A successful native reconstruction may be reused explicitly with
`--reconstruction_cache /path/to/prior-run`. Its image/runtime identities,
segmentation contents, and the exact complete raw-asset hash set must match.
The new run records the producer source identity; this does not assert a fresh
reconstruction under the consumer source. Use a new work directory for continuation.
SAM3D opacity logits of positive/negative infinity represent exact probability
one/zero. The adapter encodes these at output dtype epsilon, records their count
per object, and leaves raw PLY untouched. NaN opacity and nonfinite geometry/SH
still fail validation.

When both REST stages are explicitly cached, preflight imports the native adapter
modules (numpy/scipy/plyfile) in the REST interpreter instead of unused SAM modules.
It still validates declared paths/weight hashes and imports the material, Warp and
rasterizer dependencies. The report labels this scope; fresh REST reconstruction
retains the full SAM import probe. Cache identities/artifacts are checked separately.
