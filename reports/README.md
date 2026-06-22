# Reduced Elastic Report Utilities

These scripts reproduce local report assets for the reduced elastic examples.
Generated HTML, videos, screenshots, CSVs, and JSON summaries are intentionally
ignored by git; publish them to the reports host or another artifact store.

For ViewerGL captures on Linux, use the NVIDIA GLX backend in the same shell:

```bash
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

Render reduced elastic videos and screenshots:

```bash
uv run --extra examples python reports/render_reduced_elastic_videos.py
```

If you already have a local report HTML file, pass it to update cache-busting
query strings and mode-count rows:

```bash
uv run --extra examples python reports/render_reduced_elastic_videos.py \
    --report reports/reduced_elastic_links_implementation.html
```

Render the reduced elastic coupling videos (floating-frame inertial coupling and
joint rotational clamp moment) and refresh each report's cache-busting query
strings. With no `--report`, every known report present in `reports/` is updated:

```bash
uv run --extra examples python reports/render_reduced_elastic_coupling_videos.py
```

Pass `--report` (repeatable) to target specific reports:

```bash
uv run --extra examples python reports/render_reduced_elastic_coupling_videos.py \
    --report reports/reduced_elastic_rotational_coupling.html
```

Run the reduced elastic iteration sweep:

```bash
uv run --extra examples python reports/sweep_reduced_elastic_iterations.py \
    --output reports/assets/reduced_elastic_iteration_sweep.csv
```

Generate the dipper static sag diagnostic report:

```bash
uv run --extra examples python reports/run_dipper_static_sag_experiment.py
```
