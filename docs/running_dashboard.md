# Running the dashboard

## Start the server

From the repository root, with Python 3.9+:

```bash
python3 dashboard/serve.py
```

Open **http://127.0.0.1:8080**. Stop the server with `Ctrl+C`. Use another port if needed:

```bash
python3 dashboard/serve.py --port 8081
```

The server defaults to `127.0.0.1`; `--host` changes the bind address. It uses only the Python standard library. No frontend build, model weights, GPU or generated manifest is required.

## Results location

| Dashboard source | Directory |
| --- | --- |
| Paper results | `analysis/results/` |

The dashboard reads results without running experiments or rewriting files. Click **Refresh** after generating or changing analysis outputs. The source path and load time identify the displayed data. Local reruns share the same output tree; they are not presented as separate runs.

## Explore the research questions

- **Model robustness (RQ1):** select a dataset, comparison and AP/AR metric. Expand the untransformed baseline if needed. **Explore a transformation** provides a transformation selector and tabs for parameter response, retained/lost detections, confidence shifts and cone colours. The cone-colour tab shows its combined figure independently of the selected transformation. There is no robustness summary table.
- **Brightness tuning (RQ2):** switch between search efficiency, brightness scoring and accuracy distributions. Select a model and, where available, a search method. Holdout validation loads metrics and figures from `validation/<dataset>/<model>/filtered_ap_0.1/`; select the validation dataset to compare saved datasets.
- **Fewer test runs (RQ3):** inspect the actual sample size, equivalence outcomes and response curves. Choose transformation, AP/AR and consistency or accuracy degradation; source JSON and report links provide supporting details.

Plots link to full-resolution images. RQ1 generates a response chart from metrics when the saved parameter plot is absent. Other missing figures and empty result sources are shown explicitly.

## Troubleshooting and verification

If **Paper results** is empty, generate outputs with the [workflow runners](running_end_to_end.md). If a figure is missing, run its corresponding [analysis script](running_individual_scripts.md) and refresh. Keep the standard output directory layout: the loader scans known paths rather than every file. Unreadable JSON files are listed in the dashboard's warning details.

The server only exposes its static frontend, the results API (`/api/paper`) and supported result files beneath `analysis/results`. There is no `/api/models` endpoint.

Run the dashboard checks from the repository root:

```bash
python3 -m unittest discover -s dashboard -p 'test_*.py'
```

The HTTP test needs permission to bind a local socket. See [paper-result interpretation](research_paper_results.md) and the [dashboard implementation notes](../dashboard/README.md) for data semantics.
