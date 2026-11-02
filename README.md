# YOLO Metamorphic Testing

Evaluates YOLO detection robustness under various metamorphic tests including brightness, blur, resolution, stretch, grayscale, hue and cone-colour transformations.

The model registry currently enables `yolov5`, `yolov8`, `yolov12` and `yolov26_v2`. Registered weights and datasets must be available locally to run inference.

## Inspect the research results

From the repository root:

```bash
python3 dashboard/serve.py
```

Open **http://127.0.0.1:8080** to view **Paper results**. Viewing the dashboard requires Python 3.9+ and its standard library.

The dashboard reads paper analysis from `analysis/results/`. See [running the dashboard](docs/running_dashboard.md).

## Documentation

| Task | Guide |
| --- | --- |
| Add a model or weights variant | [Adding models](docs/adding_models.md) |
| Inspect saved paper evidence and understand metrics | [Research paper results](docs/research_paper_results.md) |
| Start and use the local dashboard | [Running the dashboard](docs/running_dashboard.md) |
| Prepare data and run RQ1, RQ2 or RQ3 | [End-to-end workflows](docs/running_end_to_end.md) |
| Run transformations, inference, evaluation or analysis separately | [Individual scripts](docs/running_individual_scripts.md) |

[Documentation index](docs/README.md)

## Run experiments

Use Bash and a Python environment compatible with the pinned [requirements](requirements.txt), including PyTorch, Ultralytics, OpenCV and pycocotools. The dashboard's Python minimum does not establish compatibility with the full experiment dependency set. CUDA is recommended for inference; dependency and device compatibility depend on your environment.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 utils/model_registry.py --list
python3 utils/model_registry.py --validate yolov8
```

After preparing images, labels and weights, the main entry points are:

```bash
# RQ1: transformations, inference, evaluation and plots
bash end_to_end_scripts/run_rq1.sh --dataset test --model v8

# RQ2: both brightness searches, scoring and performance distributions
bash end_to_end_scripts/run_rq2.sh --dataset test --models yolov8

# RQ3: bootstrap equivalence using existing test detections and RQ1 metrics
bash end_to_end_scripts/run_rq3.sh --models yolov8
```


## Research questions

- **RQ1 : Model robustness:** compare original and transformed detections, accuracy against ground truth, parameter responses, detection overlap, confidence shifts and cone colours.
- **RQ2 : Brightness tuning:**  Examine brightness scores and separate holdout validation (2b), and compare original/transformed accuracy distributions.
- **RQ3 : Fewer test runs:** bootstrap equivalence between a fixed subsample and the full test dataset. We use 50% in our experiments. We also investigate a more efficient search algorithm to identify the brightness operating range.


## Repository layout

```text
config/                    Model registry, transformation variants, cone classes
models/                    Inference adapters and local model weights
utils/                     Transformations, segmentation, registry and drawing tools
data/base_images/          Images, labels and COCO annotations
data/transformed_images/   Generated transformed images
results/                   Local inference outputs, created by experiments
original_results/          Archived inference outputs
analysis/
  coco_eval.py             COCO evaluation
  utils/                   Conversion, overlap and equivalence analysis
  visualisation/           Plot and report generation
  scripts/                 RQ1 analysis runners
  results/                 Paper analysis used by the dashboard
end_to_end_scripts/        RQ1, RQ2 and RQ3 workflow wrappers
scripts/                   Standalone transformation runners and rq2/ programs
dashboard/                 Standard-library server and static frontend
docs/                      Workflow guides
```
