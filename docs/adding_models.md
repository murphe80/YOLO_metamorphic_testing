# Adding a model

Models are registered in [config/models.json](../config/models.json). The current entries are `yolov5`, `yolov8`, `yolov12`, `yolov26` and `yolov26_v2`; YOLOv26 is already integrated. Use a new name for another architecture or weights variant.

## 1. Add weights and an inference adapter

Create `models/<name>/`, place the weights there and implement an adapter based on an existing `infer_*.py` file. For example, use `yolov8_custom` with version `v8_custom`. Change the output directory and model labels along with the weight-loading logic.

Existing adapters still reference the removed `utils.visualise` module. A new adapter must use the current import:

```python
from utils.visualise_detections import visualize_detections as visualise
```

The RQ1 runner invokes adapters with `--weights`, `--image` and `--output-subdir`. Existing adapters also expose `--conf_thres`, `--iou_thres`, `--device` and `--save-viz`. Preserve these interfaces when adapting one.

Write one `<image-stem>.txt` file per image, including an empty file for no detections. Each row must be:

```text
class_id confidence x1 y1 x2 y2
```

Coordinates are pixel corners, not normalized YOLO training labels. Map class IDs consistently to the dataset labels and [cone-class configuration](../config/cone_classes.json).

Keep the adapter's base output directory at `results/<name>` and join it with `--output-subdir`. The RQ1 wrappers currently pass a value such as `../test/yolov8_custom/original`, which resolves to `results/test/yolov8_custom/original`. Writing every invocation to a fixed `original/` directory would break transformed-result separation.

## 2. Register the model

Append an object to the existing `models` array; do not replace the other entries:

```json
{
  "name": "yolov8_custom",
  "display_name": "YOLOv8 custom",
  "version": "v8_custom",
  "weights": "models/yolov8_custom/weights.pt",
  "inference_script": "models/yolov8_custom/infer_custom.py",
  "results_dir": "results/yolov8_custom",
  "enabled": true,
  "color": {
    "rgb": "rgb(0, 121, 107)",
    "hex": "#00796B"
  }
}
```

Paths are relative to the repository root. `name` and `version` must be unique. RQ1 uses the version (`v8_custom`); RQ2 uses the name (`yolov8_custom`). `enabled: false` excludes the entry from default enabled-model listings. If changing `default_model`, it must name an enabled entry. Registry fields are consumed by individual tools; do not assume arbitrary per-model inference options are automatically forwarded.

## 3. Validate and smoke-test

```bash
python3 utils/model_registry.py --list
python3 utils/model_registry.py --list-versions
python3 utils/model_registry.py --get yolov8_custom
python3 utils/model_registry.py --validate yolov8_custom

# Replace /path/to/image.jpg with a real image
python3 models/yolov8_custom/infer_custom.py \
  --weights models/yolov8_custom/weights.pt \
  --image /path/to/image.jpg --device cpu --output-subdir smoke
```

Inspect `results/yolov8_custom/smoke/<image-stem>.txt` for valid confidence values, pixel boxes and class IDs. Registry validation only checks that weights and script files exist; the smoke run checks loading and output behavior.

Then run a prepared small dataset through the [base workflow](running_end_to_end.md):

```bash
bash end_to_end_scripts/run_rq1_base_experiment.sh --dataset test --model v8_custom
```

## 4. Check each integration

- **RQ1:** model selection and analysis runners use the registry. Confirm detections, COCO conversion and per-variant `stats.json` appear under the new name.
- **RQ2:** search programs read registry weights but load them directly with `ultralytics.YOLO`; they do not invoke the registered inference script. A model requiring another runtime needs corresponding loader support before RQ2 can use it.
- **RQ3:** pass `--models yolov8_custom` once its RQ1 inputs exist. The default model tuple is fixed in `calculate_sample_equivalence.py`.
- **Dashboard:** models are discovered from saved analysis products. Refresh **My experiments** after generating results. The dashboard does not have a model-registry API or promise to use the registry's display colours.

For Python integration, [utils/model_registry.py](../utils/model_registry.py) provides `get_models`, `get_model_by_name`, `get_model_by_version`, `get_model_names` and `validate_model_files`. Shell wrappers use [utils/model_registry.sh](../utils/model_registry.sh).
