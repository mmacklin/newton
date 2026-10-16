# Conditioned48 Release50 Provenance

Model revision: `larger-motion-hand-release50-20261016`.
Portable base: `372ba98d5c511bb1537488f9607b300f6932bdb1`.

The source is the accepted release50 derived-only bundle, not a failed trial or
the independent repeat. SHA256 identities:

| Input | SHA256 |
| --- | --- |
| Accepted bundle `SEAL.json` | `1e22abcccb87e7fa7329e7698230c4ea0cd25c0f7b45afd792cfbc5844cbec24` |
| Accepted result seal | `b7cdbbc79d058bc30cf87b3f99012c8adf198a6d548a84cfab8353cb91043dbd` |
| Derived `assets/rom-48.npz` | `526d8f3d1f9ffc2e2979429036250b7bfaaca6f81923f9c850b3e45d4944daea` |
| Derived `assets/targets.npz` | `7eafaf8c51a4992d137d38fb874fea24004e54ec0c04ba15c89d79828f641610` |
| Accepted physical `results.json` | `e8e2d57e1a7d6fabce30335ba33a452d821f87caa7b6890344da60f27f0c98fb` |
| Independent repeat seal | `1a40af8d703920432f02bc3491862a08485e1b61fda67a04dc16644fb7d64dc2` |

The derived arrays are copied byte-for-byte. The 48-dimensional span retains
five strain-bounded whole-surface targets from the coherent64 POD and 43
complementary directions. Setup diagonalization uses the unchanged material;
this is not a visually amplified old basis. No raw NIMBLE model or native pose
dataset is included. See [ATTRIBUTION.md](ATTRIBUTION.md).

The source transfer preserves portable asset defaults, current Warp annotations
and nonzero exit on either physical or paired-final failure. It adds equilibrium
checks, separate centroid/maximum travel labels and a complete saved-force audit.
That audit was moved from the worker test into `audit_equilibrium.py` so the
entry point does not import a test module. It also supports explicitly selected
substep rates while checking every stored force and the full released intervals.
`sources.json` records original and bundled source hashes; the original audit
and release tests were adapted together. Other hand physics helpers are unchanged.

The accepted trajectory and its bitwise-identical repeat used a separately
optimized common runtime, manifest SHA256
`c1b1288f5c05326a3fc2b14812c271982e159219a7868ac44104b144a4d85800`.
Their pass and timing must not be assumed to transfer to this checkout's runtime.
A fresh full positive/control validation is required; no runtime patches or
timing claim are included in this portable update. The 15% principal-strain gate
is unchanged despite the accepted run's narrow margin.
