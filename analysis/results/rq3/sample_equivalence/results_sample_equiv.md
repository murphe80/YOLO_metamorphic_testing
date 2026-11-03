# Test-set sample-size metamorphic consistency equivalence

## Research question

If only 50% of the test dataset were used for metamorphic assessment, would (a) the observed original-vs-transformed consistency and (b) the observed ground-truth accuracy degradation caused by each transformation be equivalent to using the full test dataset?

## Design

The analysis covers 4 models, 6 rules, and 51 settings, drawn entirely from the `test` dataset (363 images).

The `sample` arm uses a single fixed, reproducibly-drawn subsample containing 182 images (50% of 363, rounded to the nearest image). The same subsample is reused across every model, rule, setting, and analysis below.

Two independent statistics are tested for equivalence between the two arms:

- **Original-vs-transformed consistency** (original_vs_transformed_consistency): raw COCO AP/AR with each model's original detections as the reference and the transformed detections as the prediction (no baseline subtraction).
- **Ground-truth accuracy degradation from transformation** (gt_accuracy_degradation): original_vs_gt AP/AR (untransformed detections vs. real ground truth; a per-model baseline, source-specific for hue mappings) minus transformed_vs_gt AP/AR (transformed detections vs. real ground truth).

Each arm is independently resampled 2,000 times with replacement from within its own population: the `full` arm resamples from all 363 images, while the `sample` arm resamples only from its 182-image subsample. Results use percentile 90% confidence intervals and a symmetric +/-0.05 equivalence margin. No p-values or multiple-testing corrections are used.


Classification rules:

- **Equivalent:** the complete 90% interval lies within [-0.05, +0.05].
- **Meaningfully different:** the complete interval lies beyond one equivalence boundary.
- **Inconclusive:** the interval overlaps an equivalence boundary.

A positive difference means the full-dataset statistic is greater than the 50% subsample; a negative difference means the subsample statistic is greater.


## Original-vs-transformed consistency

Statistic: raw COCO AP/AR with each model's original detections as the reference and the transformed detections as the prediction (no baseline subtraction).

### Key findings

- At parameter level, 332 of 408 comparisons (81.4%) were equivalent, 76 were inconclusive, and 0 were meaningfully different.
- At rule-curve level, 40 of 40 AUC comparisons were equivalent; 0 were inconclusive and 0 were meaningfully different.
- Every AUC comparison was equivalent for YOLOv5, YOLOv8, YOLOv12, YOLO26. Their overall parameter-response patterns therefore transfer from the full test set to the 50% subsample within the +/-0.05 margin.

### Parameter-level overview

| Comparison | Metric | Settings | Equivalent | Inconclusive | Different | Equivalent % |
| --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AP | 204 | 167 | 37 | 0 | 81.9% |
| Test subsample | AR | 204 | 165 | 39 | 0 | 80.9% |



### Sample-size equivalence summary

| Model | Sample AP | Sample AR | All equivalent AUCs |
| --- | --- | --- | --- |
| YOLOv5 | 5/5 | 5/5 | 10/10 |
| YOLOv8 | 5/5 | 5/5 | 10/10 |
| YOLOv12 | 5/5 | 5/5 | 10/10 |
| YOLO26 | 5/5 | 5/5 | 10/10 |


## Ground-truth accuracy degradation from transformation

Statistic: original_vs_gt AP/AR (untransformed detections vs. real ground truth; a per-model baseline, source-specific for hue mappings) minus transformed_vs_gt AP/AR (transformed detections vs. real ground truth).

### Key findings

- At parameter level, 342 of 408 comparisons (83.8%) were equivalent, 66 were inconclusive, and 0 were meaningfully different.
- At rule-curve level, 40 of 40 AUC comparisons were equivalent; 0 were inconclusive and 0 were meaningfully different.

### Parameter-level overview

| Comparison | Metric | Settings | Equivalent | Inconclusive | Different | Equivalent % |
| --- | --- | --- | --- | --- | --- | --- |
| Test subsample | AP | 204 | 166 | 38 | 0 | 81.4% |
| Test subsample | AR | 204 | 176 | 28 | 0 | 86.3% |

