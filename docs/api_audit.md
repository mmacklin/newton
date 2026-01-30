# Newton API Audit

Based on the design guidelines, this audit identifies specific inconsistencies and recommended changes.

---

## 0. High-Level Module Organization

### Collision Pipeline Architecture

There are two collision pipelines with different design philosophies:

| Class | BroadPhase | Use Case |
|-------|------------|----------|
| `CollisionPipeline` | Pre-computed pairs only | Simple scenes, explicit pair lists |
| `CollisionPipelineUnified` | Dynamic (NXN, SAP, EXPLICIT) | Full-featured, pluggable broad phase |

### BroadPhase Design Issues

**Issue 1: Naming inconsistency** - `BroadPhaseMode` vs `SAPSortType`
- Why `Mode` suffix for one enum and `Type` suffix for another?
- Should standardize: either `BroadPhaseType` + `SAPSortType`, or `BroadPhaseMode` + `SAPSortMode`

**Issue 2: Redundant enum** - Why have `BroadPhaseMode` enum at all when we have concrete types?

Current design:
```python
CollisionPipelineUnified(..., broad_phase_mode: BroadPhaseMode = ...)
```

The enum exists only to select which BroadPhase class to instantiate internally. Better design - pass the object directly:
```python
CollisionPipelineUnified(..., broad_phase: BroadPhaseBase | None = ...)
```
This eliminates the redundant enum entirely and follows the pattern where types ARE the selector.

**Issue 3: Location scattering**

| Symbol | Current Location | Should Be |
|--------|------------------|-----------|
| `BroadPhaseMode` | `newton.BroadPhaseMode` | Remove (use types directly) |
| `BroadPhaseAllPairs` | `newton.geometry` | OK (correct) |
| `BroadPhaseSAP` | `newton.geometry` | OK (correct) |
| `BroadPhaseExplicit` | `newton.geometry` | OK (correct) |
| `SAPSortType` | `newton.SAPSortType` (NOT in geometry!) | `newton.geometry.SAPSortType` or `BroadPhaseSAP.SortType` |

**Recommendation**: Remove `BroadPhaseMode` enum entirely; change API to accept BroadPhase objects directly.

### geometry Module Mixing Concerns

The `newton.geometry` module mixes collision-specific and general mesh utilities:

**Collision-specific (keep in geometry)**:
- `BroadPhase*` classes
- `collide_*` functions
- `SDFData`, `SDFHydroelasticConfig`
- `compute_sdf`
- `compute_shape_inertia`, `transform_inertia`

**Mesh utilities (move to utils)**:
- `heightfield_to_mesh` -> `newton.utils.heightfield_to_mesh`
- `generate_terrain_grid` -> `newton.utils.generate_terrain_grid`
- `remesh_mesh` -> `newton.utils.remesh_mesh`

**Duplicate to remove**:
- `newton.geometry.create_box_mesh` - already exists in `newton.utils.create_box_mesh`

### Proposed Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `newton.geometry` | Collision primitives, broad/narrow phase, SDF, inertia computation |
| `newton.utils` | Mesh creation/manipulation, terrain generation, math helpers, recorders |
| `newton.solvers` | All solver implementations |
| `newton.sensors` | All sensor implementations |
| `newton.viewer` | All viewer implementations |
| `newton.ik` | Inverse kinematics |
| `newton.selection` | Entity selection/views |

---

## 1. Top-Level Namespace Clutter

### Symbols that should NOT be at top-level `newton.*`

| Symbol | Current Location | Issue | Recommendation |
|--------|------------------|-------|----------------|
| `SAPSortType` | `newton.SAPSortType` (but NOT `newton.geometry.SAPSortType`) | Only used by `BroadPhaseSAP`, exported at wrong level | Either move to `newton.geometry` only, or make it `BroadPhaseSAP.SortType` |
| `MESH_MAXHULLVERT` | `newton._src.geometry` | Internal constant | Make private or move to `Mesh` class |
| `test_group_pair` | `newton.geometry` | Internal utility | Prefix with `_` or remove from `__all__` |
| `test_world_and_group_pair` | `newton.geometry` | Internal utility | Prefix with `_` or remove from `__all__` |

### Symbols exported multiple times

| Symbol | Locations | Recommendation |
|--------|-----------|----------------|
| `ik` (module) | `newton.ik`, `newton._src.sim.ik` | Keep only `newton.ik` |
| `color_graph`, `plot_graph` | `newton._src.sim` | Move to `newton.utils` if general, or keep internal |

---

## 2. Naming Convention Issues

### Enum Suffix Inconsistency: `*Mode` vs `*Type`

The API uses both `Mode` and `Type` suffixes inconsistently for enums:

| Current | Suffix | Recommendation |
|---------|--------|----------------|
| `AxisType` | Type | OK Keep |
| `BroadPhaseMode` | Mode | -> Remove (redundant with types) |
| `EqType` | Type | OK Keep |
| `GeoType` | Type | OK Keep |
| `JointType` | Type | OK Keep |
| `SAPSortType` | Type | OK Keep |
| `IKJacobianMode` | Mode | -> `IKJacobianType` (or just `JacobianType`) |

**Recommendation**: Standardize on `*Type` suffix for all enums representing a choice of kind/variant.

### Attribute Naming: Use Consistent Abbreviations

Three different stiffness parameter patterns exist:

| Pattern | Examples | Issue |
|---------|----------|-------|
| Short form | `ke`, `kd`, `kf`, `ka`, `mu` in `ShapeConfig` | OK Good |
| `k_` prefix | `k_mu`, `k_lambda`, `k_damp` in soft body methods | Different convention |
| Mixed | `k_hydro` in `ShapeConfig` | Mixes with pattern 1! |

| Current | Issue | Recommended |
|---------|-------|-------------|
| `shape_material_k_hydro` | Inconsistent with `ke`, `kd`, `kf`, `ka` | `shape_material_kh` |
| `ShapeConfig.k_hydro` | Same | `ShapeConfig.kh` |

### Count Naming: Prefer `*_count` over `num_*`

| Current | Issue | Recommended |
|---------|-------|-------------|
| `Model.num_worlds` | Uses `num_*` pattern | `Model.world_count` |
| `ModelBuilder.num_worlds` | Same | `ModelBuilder.world_count` |
| `num_rigid_contacts_per_world` | Uses `num_*` pattern | `rigid_contact_count_per_world` |
| `num_tris` in `MeshAdjacency()` | Uses `num_*` pattern | `tri_count` |
| `num_cameras` in `SensorTiledCamera()` | Uses `num_*` pattern | `camera_count` |
| `num_axes` in `get_joint_dof_count()` | Uses `num_*` pattern | `axis_count` |

### Capacity vs Count: Use `*_max` for capacity, `*_count` for current

Current usage in `Contacts` is good:
- `rigid_contact_max` (capacity) OK
- `rigid_contact_count` (current count) OK

---

## 2.5 Parameter Naming Inconsistencies in Signatures

### Position/Rotation parameter abbreviations

Inconsistent use of abbreviations vs full words:

| Context | Uses | Should Be |
|---------|------|-----------|
| `add_particle(pos: Vec3, ...)` | `pos` | Keep (common in physics) |
| `add_cloth_grid(pos: Vec3, rot: Quat, ...)` | `pos`, `rot` | Keep |
| `SensorRaycast(camera_position: ...)` | `camera_position` | `camera_pos` (or stay full - but be consistent) |
| `IKPositionObjective(target_positions, ...)` | `target_positions` | OK (plural makes sense) |
| `JointDofConfig(target_pos: float, ...)` | `target_pos` | OK |

**Issue**: Mix of `pos`/`position` and `rot`/`rotation`. Pick one pattern.

### Doubled prefixes

| Current | Issue | Recommended |
|---------|-------|-------------|
| `BroadPhaseAllPairs.launch(..., shape_shape_world, ...)` | Double "shape" | `shape_world` |
| `BroadPhaseSAP(..., shape_shape_world, ...)` | Double "shape" | `shape_world` |

### Half-extent parameterization inconsistency

| Function | Parameters | Issue |
|----------|------------|-------|
| `geometry.create_box_mesh(half_extents, ...)` | Single `half_extents` | Uses combined |
| `utils.create_box_mesh(extents)` | Single `extents` | Different name! |
| `add_shape_box(hx, hy, hz, ...)` | Separate `hx`, `hy`, `hz` | Different style |

**Critical Issue**: Two `create_box_mesh` functions with DIFFERENT parameter names:
- `newton.geometry.create_box_mesh(half_extents, ...)`
- `newton.utils.create_box_mesh(extents)`

This is confusing - which is half and which is full? Consolidate into one.

### Mesh hull vertex limit parameter

| Location | Parameter Name |
|----------|---------------|
| `Mesh()` constructor | `maxhullvert` |
| `add_mjcf()`, `add_urdf()`, `add_usd()` | `mesh_maxhullvert` |

Inconsistent prefix usage.


## 2.6 Viewer API Inconsistencies

### log_lines parameter names differ across viewers

| Viewer | Parameters | Issue |
|--------|------------|-------|
| `ViewerFile.log_lines()` | `line_begins`, `line_ends` | Uses `line_` prefix |
| `ViewerGL.log_lines()` | `starts`, `ends` | No prefix |
| `ViewerNull.log_lines()` | `starts`, `ends` | No prefix |
| `ViewerRerun.log_lines()` | `starts`, `ends` | No prefix |

**Recommendation**: Standardize on `starts`, `ends` (shorter, cleaner).

### log_points parameters inconsistent

| Viewer | Parameters | Issue |
|--------|------------|-------|
| `ViewerGL.log_points()` | `radii` | Uses `radii` |
| `ViewerNull.log_points()` | `radii`, `width` | Has BOTH `radii` and `width`! |
| `ViewerRerun.log_points()` | `radii` | Uses `radii` |

**Issue**: `ViewerNull.log_points()` has both `radii` and `width` parameters - redundant.

---

## 3. IK Module Naming (Violates "Prefer common prefixes")

Current IK classes have inconsistent prefix grouping:

### Objectives (should group together when typing `IKObjective...`)
| Current | Recommended |
|---------|-------------|
| `IKObjective` (base) | `IKObjective` (base) OK |
| `IKPositionObjective` | `IKObjectivePosition` |
| `IKRotationObjective` | `IKObjectiveRotation` |
| `IKJointLimitObjective` | `IKObjectiveJointLimit` |

### Optimizers (should group together when typing `IKOptimizer...`)
| Current | Recommended |
|---------|-------------|
| `IKOptimizer` (base) | `IKOptimizer` (base) OK |
| `IKLBFGSOptimizer` | `IKOptimizerLBFGS` |
| `IKLMOptimizer` | `IKOptimizerLM` |

### Already correct
| Current | Status |
|---------|--------|
| `IKJacobianMode` | X Should be `IKJacobianType` to match `*Type` pattern |
| `IKSampler` | OK |
| `IKSolver` | OK |

---

## 4. General-Purpose Names (Violates "Avoid general purpose names")

| Current | Issue | Recommended |
|---------|-------|-------------|
| `RecorderBasic` | Generic name, suffix pattern | `ModelRecorder` (single class, optionally records state) |
| `RecorderModelAndState` | Verbose, suffix pattern | Merge into `ModelRecorder` with optional state recording |

---

## 5. Enum vs Constants (Violates "Prefer Enum over constants")

| Current | Type | Recommendation |
|---------|------|----------------|
| `MAXVAL` | Constant | Keep (used in Warp kernels, enum won't work) |
| `MESH_MAXHULLVERT` | Constant | Move to `Mesh.MAX_HULL_VERTICES` class attribute |

---

## 6. Solver Internal Methods (Should be private)

These methods appear internal and should be prefixed with `_`:

### SolverFeatherPGS
```
allocate_augmented_joint_buffers -> _allocate_augmented_joint_buffers
allocate_model_aux_vars -> _allocate_model_aux_vars
allocate_pgs_buffers -> _allocate_pgs_buffers
allocate_state_aux_vars -> _allocate_state_aux_vars
apply_augmented_joint_tau -> _apply_augmented_joint_tau
build_augmented_joint_targets -> _build_augmented_joint_targets
build_body_maps -> _build_body_maps
compute_articulation_indices -> _compute_articulation_indices
solve_contacts_pgs -> _solve_contacts_pgs
```

### SolverFeatherstone
```
allocate_model_aux_vars -> _allocate_model_aux_vars
allocate_state_aux_vars -> _allocate_state_aux_vars
compute_articulation_indices -> _compute_articulation_indices
```

### SolverVBD
```
collision_detection_penetration_free -> _collision_detection_penetration_free
compute_force_element_adjacency -> _compute_force_element_adjacency
count_num_adjacent_* -> _count_num_adjacent_*
fill_adjacent_* -> _fill_adjacent_*
simulate_one_step_* -> _simulate_one_step_*
```

### SolverMuJoCo (most should be private)
```
apply_mjc_control -> _apply_mjc_control
close_mujoco_viewer -> keep public (user-facing)
color_collision_shapes -> _color_collision_shapes
convert_contacts_to_mjwarp -> _convert_contacts_to_mjwarp
expand_model_fields -> _expand_model_fields
find_body_collision_filter_pairs -> _find_body_collision_filter_pairs
import_mujoco -> keep public (initialization)
mujoco_warp_step -> _mujoco_warp_step
render_mujoco_viewer -> keep public (user-facing)
update_geom_properties -> _update_geom_properties
update_joint_* -> _update_joint_*
update_mjc_data -> _update_mjc_data
update_model_* -> _update_model_*
update_newton_state -> _update_newton_state
```

---

## 7. One-Off Patterns (API Gaps / Inconsistencies)

These are symbols that stand out as unique - often indicating missing API or inconsistent design.

### ModelBuilder default configs use inconsistent implementation

| Attribute | Implementation | Type |
|-----------|----------------|------|
| `default_site_cfg` | @property (computed) | Returns a new ShapeConfig each call |
| `default_shape_cfg` | instance attribute | ShapeConfig stored in `__init__` |
| `default_joint_cfg` | instance attribute | JointDofConfig stored in `__init__` |
| `default_body_armature` | instance attribute | float |
| `default_edge_ke`, `default_edge_kd` | instance attributes | float |
| `default_spring_ke`, `default_spring_kd` | instance attributes | float |
| `default_tri_ke`, `default_tri_ka`, etc. | instance attributes | float |
| `default_particle_radius` | instance attribute | float |

**Issue**: `default_site_cfg` is a @property while all other defaults are instance attributes. This means:
- `default_site_cfg` returns a NEW object each time (can't be modified persistently)
- `default_shape_cfg` is mutable (changes persist)

**Recommendation**: Make all defaults consistent - either all properties or all instance attributes.

### `add_link` vs `add_body` - identical signatures

```python
add_body(xform, armature, com, I_m, mass, key, custom_attributes, lock_inertia) -> int
add_link(xform, armature, com, I_m, mass, key, custom_attributes, lock_inertia) -> int
```

**Question**: Are these aliases? If so, deprecate one. If different, document the difference.

### `Axis` class AND `AxisType` enum both at top level

| Symbol | Type |
|--------|------|
| `Axis` | Class |
| `AxisType` | Enum |

**Question**: What's the relationship? Is `Axis` an instance and `AxisType` a selector? Confusing.

### `Model.set_gravity()` - only setter

Model has no other `set_*` methods. Why is gravity special? Other properties are set via attributes or builder.

**Suggestion**: Either add more setters or make gravity consistent with other properties.

### Style3DModelBuilder has methods ModelBuilder lacks

| Method | In Style3DModelBuilder | In ModelBuilder |
|--------|------------------------|-----------------|
| `sew_close_vertices()` | Yes | No |
| `add_aniso_cloth_grid()` | Yes | No |
| `add_aniso_cloth_mesh()` | Yes | No |
| `add_aniso_edges()` | Yes | No |
| `add_aniso_triangles()` | Yes | No |

**Question**: Is Style3DModelBuilder a subclass or separate? Should these be in base class?

### Viewer API asymmetry

| Feature | ViewerFile | ViewerGL | ViewerNull | ViewerRerun | ViewerUSD | ViewerViser |
|---------|------------|----------|------------|-------------|-----------|-------------|
| `get_recorder()` | Yes | No | No | No | No | No |
| `save_recording()` | Yes | No | No | No | No | Yes |
| `load_recording()` | Yes | No | No | No | No | No |
| `url` property | No | No | No | No | No | Yes |
| `show_notebook()` | No | No | No | Yes | No | Yes |
| `get_frame()` | No | Yes | No | No | No | No |
| `register_ui_callback()` | No | Yes | No | No | No | No |

**Issue**: Viewers have very different capabilities. Consider a base interface with optional capabilities.

### SensorTiledCamera has unique nested classes

```python
SensorTiledCamera:
    .Options(...)        # Nested config class
    .RenderContext(...)  # Nested class
    .RenderLightType     # Nested enum
    .RenderShapeType     # Nested enum
```

No other sensor has this level of nesting. Creates inconsistent API surface.

### SolverImplicitMPM has unique nested classes

```python
SolverImplicitMPM:
    .Model(...)   # Nested class
    .Options(...) # Nested config class
```

Other solvers take options as constructor params. Inconsistent pattern.

### Solver method asymmetry

| Method | Featherstone | VBD | XPBD | Style3D | MuJoCo | ImplicitMPM |
|--------|--------------|-----|------|---------|--------|-------------|
| `step()` | Yes | Yes | Yes | Yes | Yes | Yes |
| `rebuild_bvh()` | No | Yes | No | Yes | No | No |
| `precompute()` | No | No | No | Yes | No | No |
| `enrich_state()` | No | No | No | No | No | Yes |
| `project_outside()` | No | No | No | No | No | Yes |
| `update_particle_frames()` | No | No | No | No | No | Yes |

**Issue**: Solvers have widely varying public APIs. Hard to swap solvers.

### `populate_contacts()` - only free function in sensors

All other symbols in `newton.sensors` are classes. This one function stands out.

**Suggestion**: Make it a method on `Contacts` or `SensorContact`.

### `MatchKind` enum - only used by SensorContact

One-off enum with very limited scope. Consider making it `SensorContact.MatchKind`.

### Top-level free functions

```python
newton.count_rigid_contact_points()
newton.eval_fk()
newton.eval_ik()
newton.get_joint_constraint_count()
newton.get_joint_dof_count()
```

These are the only free functions at the newton top level (besides module-level constants).

**Questions**:
- Should `eval_fk`/`eval_ik` be methods on `Model` or `State`?
- Should `count_rigid_contact_points` be on `Model` or `Contacts`?
- Should `get_joint_*_count` be on `Model` or `JointType`?

---

## 8. Good Examples to Follow

### Sensor Naming (Follows "Prefer common prefixes")
```
SensorContact
SensorFrameTransform
SensorIMU
SensorRaycast
SensorTiledCamera
```

### Solver Naming (Follows "Prefer common prefixes")
```
SolverBase
SolverFeatherstone
SolverImplicitMPM
SolverMuJoCo
SolverSemiImplicit
SolverStyle3D
SolverVBD
SolverXPBD
```

### Viewer Naming (Follows "Prefer common prefixes")
```
ViewerFile
ViewerGL
ViewerNull
ViewerRerun
ViewerUSD
ViewerViser
```

### IK Naming (after proposed changes)
```
IKJacobianType        (was IKJacobianMode)
IKObjective           (base - keep)
IKObjectivePosition   (was IKPositionObjective)
IKObjectiveRotation   (was IKRotationObjective)
IKObjectiveJointLimit (was IKJointLimitObjective)
IKOptimizer           (base - keep)
IKOptimizerLBFGS      (was IKLBFGSOptimizer)
IKOptimizerLM         (was IKLMOptimizer)
IKSampler             (keep)
IKSolver              (keep)
```

---

## 9. Summary of Priority Changes

### Critical (Breaking but high-value)
1. **Remove `BroadPhaseMode` enum entirely** - redundant with BroadPhase* classes
   - Change `CollisionPipelineUnified(broad_phase_mode=...)` to `CollisionPipelineUnified(broad_phase=...)`
   - Accept `BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit | None` directly
2. **Fix duplicate `create_box_mesh`** - two functions with different parameter names!
   - `geometry.create_box_mesh(half_extents)` vs `utils.create_box_mesh(extents)`
   - Keep one in `utils`, remove from `geometry`

### High Priority (Consistency/Guidelines)
3. Rename `shape_material_k_hydro` -> `shape_material_kh` (and `ShapeConfig.k_hydro` -> `kh`)
4. Rename `num_worlds` -> `world_count` (and all other `num_*` -> `*_count`)
5. Move `SAPSortType` from `newton.*` to `newton.geometry` only
6. Make solver internal methods private (`_` prefix)
7. Rename IK classes for prefix grouping:
   - `IKPositionObjective` -> `IKObjectivePosition`
   - `IKRotationObjective` -> `IKObjectiveRotation`
   - `IKJointLimitObjective` -> `IKObjectiveJointLimit`
   - `IKLBFGSOptimizer` -> `IKOptimizerLBFGS`
   - `IKLMOptimizer` -> `IKOptimizerLM`
   - `IKJacobianMode` -> `IKJacobianType`
8. Fix doubled parameter: `shape_shape_world` -> `shape_world` in BroadPhase classes

### Medium Priority (API Polish)
9. Move mesh utilities from `newton.geometry` to `newton.utils`:
   - `heightfield_to_mesh`
   - `generate_terrain_grid`
   - `remesh_mesh`
10. Remove internal utilities from `__all__` (`test_group_pair`, etc.)
11. Consolidate `RecorderBasic` + `RecorderModelAndState` -> `ModelRecorder`
12. Fix viewer parameter inconsistencies:
    - `line_begins`/`line_ends` -> `starts`/`ends` in `ViewerFile`
    - Remove redundant `width` from `ViewerNull.log_points()`
13. Resolve `Axis` class vs `AxisType` enum confusion
14. Clarify `add_body` vs `add_link` (deprecate one or document difference)

### Low Priority (Cosmetic)
15. Move `MESH_MAXHULLVERT` to `Mesh.MAX_HULL_VERTICES`
16. Standardize mesh hull parameter: `maxhullvert` vs `mesh_maxhullvert`
17. Consider standardizing `pos`/`position` usage across API
18. Consider moving `populate_contacts()` to be a method
19. Consider moving `MatchKind` to `SensorContact.MatchKind`

---

## Appendix A: Full Model Attribute Inventory

### Shape Material Attributes (all use abbreviated form except one)
```
shape_material_ke     OK (elastic stiffness)
shape_material_kd     OK (damping)
shape_material_kf     OK (friction stiffness)
shape_material_ka     OK (adhesion)
shape_material_mu     OK (friction coefficient)
shape_material_restitution         OK (full word - acceptable)
shape_material_torsional_friction  OK (full word - acceptable)
shape_material_rolling_friction    OK (full word - acceptable)
shape_material_k_hydro             X -> shape_material_kh
```

### Count Attributes (should use `*_count` consistently)
```
particle_count     OK
body_count         OK
shape_count        OK
joint_count        OK
tri_count          OK
tet_count          OK
edge_count         OK
spring_count       OK
muscle_count       OK
articulation_count OK
joint_dof_count    OK
joint_coord_count  OK
num_worlds         X -> world_count
```

---

## Appendix B: Complete Parameter Naming Audit

### All `num_*` parameters that should be `*_count`
```
ModelBuilder.replicate(num_worlds=...)           -> world_count
MeshAdjacency(num_tris)                          -> tri_count
SensorTiledCamera(num_cameras=...)               -> camera_count
get_joint_dof_count(num_axes=...)                -> axis_count
get_joint_constraint_count(num_axes=...)         -> axis_count
```

### Doubled prefix parameters
```
BroadPhaseAllPairs.launch(shape_shape_world=...) -> shape_world
BroadPhaseSAP(shape_shape_world=...)             -> shape_world
```

### Enum naming (Mode vs Type)
```
BroadPhaseMode     X Remove entirely (redundant with types)
IKJacobianMode     X -> IKJacobianType

AxisType           OK
EqType             OK
GeoType            OK
JointType          OK
SAPSortType        OK
```

### Good naming patterns already in use
```
Sensors:  SensorContact, SensorFrameTransform, SensorIMU, SensorRaycast, SensorTiledCamera
Solvers:  SolverBase, SolverFeatherstone, SolverMuJoCo, SolverVBD, SolverXPBD, ...
Viewers:  ViewerFile, ViewerGL, ViewerNull, ViewerRerun, ViewerUSD, ViewerViser
Builder:  add_body(), add_joint(), add_shape_*(), add_particle(), ...
Factory:  CollisionPipeline.from_model(), Style3DModel.from_model()
State:    state_in/state_out pattern in solver methods
```
