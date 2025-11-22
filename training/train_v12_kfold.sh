#!/usr/bin/env bash

FOLDS=5
KFOLD="~/metamorphic_testing/YOLO_metamorphic_testing/training/dataset/fsoco/kfold"

for i in $(seq 1 $FOLDS); do
    yolo detect train \
      data="$KFOLD/fold${i}/fold${i}.yml" \
      model=yolov12s.pt \
      epochs=200 \
      patience=50 \
      imgsz=640 \
      batch=16 \
      project="~/metamorphic_testing/YOLO_metamorphic_testing/training/kfold_results/v12" \
      name="fold_${i}"
done

