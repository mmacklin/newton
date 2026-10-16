# NIMBLE Attribution And Permission

Native geometry and the pose subspace are derived from NIMBLE:

Yuwei Li, Longwen Zhang, Zesong Qiu, Yingwenqi Jiang, Nianyi Li, Yuexin Ma,
Yuyao Zhang, Lan Xu, and Jingyi Yu. **NIMBLE: A Non-Rigid Hand Model with
Bones and Muscles.** ACM Transactions on Graphics 41(4), SIGGRAPH 2022.

- Repository: https://github.com/reyuwei/NIMBLE_model
- Project: https://liyuwei.cc/proj/nimble
- Paper: https://doi.org/10.1145/3528223.3530079

On 2026-10-15 the user confirmed permission to publish attributed derived ROMs
and rendered media, while excluding original NIMBLE model files from Newton
and report bundles. This uses that explicit authorization, not an inference
that a software license resolves every separately downloaded asset's terms.

This revision uses a conditioned 48-mode subspace of a coherent 64-mode POD
trained on native independent-finger curl families and the 160 native poses,
with rest pose 77 and unchanged closed surface topology. Five bounded
whole-surface target shapes are retained exactly alongside 43 complementary
POD directions. The 10 MPa/1 mm shell, 0.35 kg lumped
mass, and damping are assumed engineering parameters, not anatomical tissue
calibration. Offline NIMBLE-informed targets are fitted within surface-strain
bounds. At runtime, prescribed modal forces shift the elastic rest coordinate;
Newton solves the hand's floating frame and all modal coordinates. The wrist
support is kinematic. The old contact probe is parked out of reach and is not
rendered. No finger position or modal coordinate is overwritten after solve.

There is no anatomical skeleton, muscle model, nonlinear strain barrier, or
self-contact force. Surface strain is a kinematic diagnostic, not calibrated
stress. Original dictionaries, checkpoints, textures, raw native pose datasets,
upstream code downloads, and original model parameters are not included.
